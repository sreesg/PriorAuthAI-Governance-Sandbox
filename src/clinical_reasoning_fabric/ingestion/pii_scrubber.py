"""PII Scrubber with HIPAA Safe Harbor compliance.

Implements regex-based detection and removal of all 18 HIPAA Safe Harbor
identifier categories. Returns scrubbed text with PII replaced by
category-tagged redaction tokens.

Requirements:
    1.2: Remove all HIPAA Safe Harbor identifiers before vector storage
    1.8: Halt processing and prevent storage on scrubbing failure
"""

import logging
import re
from typing import NamedTuple

from clinical_reasoning_fabric.models.exceptions import PIIScrubError

logger = logging.getLogger(__name__)


class PIIPattern(NamedTuple):
    """A compiled regex pattern with its redaction category tag."""
    category: str
    pattern: re.Pattern
    description: str


# Pre-compiled regex patterns for all 18 HIPAA Safe Harbor identifier categories.
# Order matters: more specific patterns should come before general ones.
_PII_PATTERNS: list[PIIPattern] = [
    # 18. Other unique identifying numbers/codes (catch-all for labeled IDs)
    # Must come before NAME pattern to avoid "Member ID" being caught as a name
    PIIPattern(
        category="UNIQUE_ID",
        pattern=re.compile(
            r"(?i)(?:member\s*id|patient\s*id|subscriber\s*id|group\s*(?:number|#|no\.?|id)|"
            r"claim\s*(?:number|#|no\.?|id)|case\s*(?:number|#|no\.?|id)|"
            r"enrollment\s*(?:number|#|no\.?|id))[:\s]*[A-Za-z0-9\-]{3,20}"
        ),
        description="Other unique identifying numbers or codes",
    ),
    # 8. Medical Record Numbers (MRN patterns)
    PIIPattern(
        category="MEDICAL_RECORD_NUMBER",
        pattern=re.compile(
            r"(?i)(?:MRN|medical\s*record\s*(?:number|#|no\.?)?)[:\s]*[A-Za-z0-9\-]{4,20}"
        ),
        description="Medical record numbers",
    ),
    # 9. Health Plan Beneficiary Numbers
    PIIPattern(
        category="HEALTH_PLAN_NUMBER",
        pattern=re.compile(
            r"(?i)(?:health\s*plan|insurance|beneficiary|policy)\s*(?:number|#|no\.?|id)[:\s]*[A-Za-z0-9\-]{4,20}"
        ),
        description="Health plan beneficiary numbers",
    ),
    # 10. Account numbers
    PIIPattern(
        category="ACCOUNT_NUMBER",
        pattern=re.compile(
            r"(?i)(?:account|acct)\s*(?:number|#|no\.?)?[:\s]*[A-Za-z0-9\-]{4,17}"
        ),
        description="Account numbers",
    ),
    # 11. Certificate/License numbers
    PIIPattern(
        category="LICENSE_NUMBER",
        pattern=re.compile(
            r"(?i)(?:license|certificate|cert)\s*(?:number|#|no\.?)?[:\s]*[A-Za-z0-9\-]{4,20}"
        ),
        description="Certificate/license numbers",
    ),
    # 12. Device identifiers/serial numbers
    PIIPattern(
        category="DEVICE_IDENTIFIER",
        pattern=re.compile(
            r"(?i)(?:device|serial|UDI)\s*(?:identifier|id|number|#|no\.?)?[:\s]*[A-Za-z0-9\-]{6,30}"
        ),
        description="Device identifiers and serial numbers",
    ),
    # 13. Biometric identifiers (fingerprint, retinal, voice references)
    PIIPattern(
        category="BIOMETRIC_ID",
        pattern=re.compile(
            r"(?i)(?:fingerprint|retinal?\s*scan|voice\s*print|iris\s*scan|"
            r"biometric|facial\s*recognition|palm\s*print|dna\s*profile)"
            r"\s*(?:id|identifier|data|hash|record)?[:\s]*[A-Za-z0-9+/=\-]{4,}"
        ),
        description="Biometric identifiers",
    ),
    # 14. Photographic images references
    PIIPattern(
        category="PHOTO_ID",
        pattern=re.compile(
            r"(?i)(?:photo(?:graph)?|image|picture|portrait|headshot|face\s*image)"
            r"\s*(?:id|identifier|file|reference|ref)?[:\s]*[A-Za-z0-9_\-./]{4,}"
        ),
        description="Photographic image references",
    ),
    # 6. Fax numbers (labeled as fax) - Must come BEFORE phone to catch "Fax: number"
    PIIPattern(
        category="FAX",
        pattern=re.compile(
            r"(?i)(?:fax|facsimile)[:\s]*(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        ),
        description="Fax numbers",
    ),
    # 2. Email addresses
    PIIPattern(
        category="EMAIL",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
        description="Email addresses",
    ),
    # 3. URLs (web addresses)
    PIIPattern(
        category="URL",
        pattern=re.compile(
            r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
        ),
        description="Web URLs",
    ),
    # 4. IP addresses (IPv4)
    PIIPattern(
        category="IP_ADDRESS",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        description="IP addresses",
    ),
    # 1. SSNs (Social Security Numbers) - XXX-XX-XXXX format specifically
    # Use strict format with required separators to avoid matching ZIP+4 or phone numbers
    PIIPattern(
        category="SSN",
        pattern=re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b"
        ),
        description="Social Security Numbers (dash format)",
    ),
    # SSN without separators (9 consecutive digits not part of longer number)
    PIIPattern(
        category="SSN",
        pattern=re.compile(
            r"(?<!\d)\d{9}(?!\d)"
        ),
        description="Social Security Numbers (no separator)",
    ),
    # 5. Phone numbers (US formats)
    PIIPattern(
        category="PHONE",
        pattern=re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        description="Phone numbers",
    ),
    # 7. VINs (Vehicle Identification Numbers) - 17 alphanumeric chars excluding I, O, Q
    PIIPattern(
        category="VIN",
        pattern=re.compile(
            r"\b[A-HJ-NPR-Z0-9]{17}\b"
        ),
        description="Vehicle Identification Numbers",
    ),
    # 15. Dates (except year alone) - MM/DD/YYYY, MM-DD-YYYY, Month DD YYYY, etc.
    PIIPattern(
        category="DATE",
        pattern=re.compile(
            r"\b(?:"
            # MM/DD/YYYY or MM-DD-YYYY or MM.DD.YYYY
            r"(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.]\d{4}"
            r"|"
            # YYYY/MM/DD or YYYY-MM-DD
            r"\d{4}[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])"
            r"|"
            # Month DD, YYYY or Month DD YYYY
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|"
            r"Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}"
            r"|"
            # DD Month YYYY
            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
            r"Sep|Oct|Nov|Dec)\.?\s+\d{4}"
            r")\b"
        ),
        description="Dates (except year alone)",
    ),
    # 16. Geographic data smaller than state (ZIP codes)
    PIIPattern(
        category="GEOGRAPHIC",
        pattern=re.compile(
            r"\b\d{5}(?:-\d{4})?\b"  # ZIP codes (5 digits, optionally followed by -4 digits)
        ),
        description="Geographic data (ZIP codes)",
    ),
    # 17. Names (common name patterns - title/label + capitalized words)
    # Requires at least one capitalized first+last name after the label
    PIIPattern(
        category="NAME",
        pattern=re.compile(
            r"(?:(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})"
            r"|(?:(?:[Pp]atient|[Mm]ember|[Nn]ame)[:\s]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})"
        ),
        description="Person names",
    ),
]


class PIIScrubber:
    """Scrubs PII from text using HIPAA Safe Harbor methodology.

    Detects and removes all 18 HIPAA Safe Harbor identifier categories
    by replacing them with category-tagged redaction tokens like
    [REDACTED_SSN], [REDACTED_EMAIL], etc.

    Raises PIIScrubError if the scrubbing process encounters any error,
    halting document processing and preventing storage (Requirement 1.8).
    """

    def __init__(self, patterns: list[PIIPattern] | None = None):
        """Initialize the scrubber with PII patterns.

        Args:
            patterns: Optional custom patterns. Defaults to the full
                     HIPAA Safe Harbor pattern set.
        """
        self._patterns = patterns if patterns is not None else _PII_PATTERNS

    @property
    def patterns(self) -> list[PIIPattern]:
        """Return the active PII patterns."""
        return self._patterns

    def scrub(self, text: str) -> str:
        """Remove all HIPAA Safe Harbor identifiers from text.

        Replaces detected PII with category-tagged tokens like
        [REDACTED_SSN], [REDACTED_EMAIL], etc.

        Args:
            text: The input text to scrub.

        Returns:
            The scrubbed text with all PII replaced by redaction tokens.

        Raises:
            PIIScrubError: If scrubbing fails or cannot be completed.
                          This halts document processing and prevents storage.
        """
        if not isinstance(text, str):
            raise PIIScrubError(
                reason="Input must be a string",
                details={"input_type": str(type(text).__name__)},
            )

        try:
            scrubbed = text
            for pii_pattern in self._patterns:
                redaction_token = f"[REDACTED_{pii_pattern.category}]"
                scrubbed = pii_pattern.pattern.sub(redaction_token, scrubbed)
            return scrubbed
        except re.error as e:
            raise PIIScrubError(
                reason=f"Regex processing error during PII scrubbing: {e}",
                details={"pattern_error": str(e)},
            )
        except (MemoryError, RecursionError) as e:
            raise PIIScrubError(
                reason=f"Resource exhaustion during PII scrubbing: {e}",
                details={"error_type": type(e).__name__},
            )
        except Exception as e:
            raise PIIScrubError(
                reason=f"Unexpected error during PII scrubbing: {e}",
                details={"error_type": type(e).__name__, "error_msg": str(e)},
            )

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII in text without removing it.

        Useful for auditing and logging what was found.

        Args:
            text: The input text to scan.

        Returns:
            Dictionary mapping category names to lists of matched strings.

        Raises:
            PIIScrubError: If detection fails.
        """
        if not isinstance(text, str):
            raise PIIScrubError(
                reason="Input must be a string",
                details={"input_type": str(type(text).__name__)},
            )

        try:
            findings: dict[str, list[str]] = {}
            for pii_pattern in self._patterns:
                matches = pii_pattern.pattern.findall(text)
                if matches:
                    findings[pii_pattern.category] = matches
            return findings
        except Exception as e:
            raise PIIScrubError(
                reason=f"Error during PII detection: {e}",
                details={"error_type": type(e).__name__, "error_msg": str(e)},
            )
