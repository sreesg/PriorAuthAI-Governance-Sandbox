"""Property-based tests for PII Scrubbing Completeness.

**Validates: Requirements 1.2, 5.1**

Property 1: PII Scrubbing Completeness
- For any generated text containing HIPAA identifiers, scrubbing removes all
  identifier patterns while preserving non-PII clinical content.
- Uses Hypothesis strategies to generate SSNs, phone numbers, emails, dates, MRNs, etc.
"""

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from clinical_reasoning_fabric.ingestion.pii_scrubber import PIIScrubber


# --- Custom Hypothesis strategies for PII generation ---

def ssn_strategy() -> st.SearchStrategy[str]:
    """Generate SSNs in format XXX-XX-XXXX with valid ranges."""
    return st.builds(
        lambda a, b, c: f"{a:03d}-{b:02d}-{c:04d}",
        st.integers(min_value=100, max_value=999),
        st.integers(min_value=10, max_value=99),
        st.integers(min_value=1000, max_value=9999),
    )


def phone_parenthesized_strategy() -> st.SearchStrategy[str]:
    """Generate phone numbers in format (XXX) XXX-XXXX."""
    return st.builds(
        lambda a, b, c: f"({a:03d}) {b:03d}-{c:04d}",
        st.integers(min_value=200, max_value=999),
        st.integers(min_value=200, max_value=999),
        st.integers(min_value=1000, max_value=9999),
    )


def phone_dashed_strategy() -> st.SearchStrategy[str]:
    """Generate phone numbers in format XXX-XXX-XXXX."""
    return st.builds(
        lambda a, b, c: f"{a:03d}-{b:03d}-{c:04d}",
        st.integers(min_value=200, max_value=999),
        st.integers(min_value=200, max_value=999),
        st.integers(min_value=1000, max_value=9999),
    )


def email_strategy() -> st.SearchStrategy[str]:
    """Generate realistic email addresses."""
    local_part = st.from_regex(r"[a-z][a-z0-9._%+]{2,14}", fullmatch=True)
    domain = st.from_regex(r"[a-z][a-z0-9]{2,8}", fullmatch=True)
    tld = st.sampled_from(["com", "org", "net", "edu", "gov", "health"])
    return st.builds(
        lambda l, d, t: f"{l}@{d}.{t}",
        local_part,
        domain,
        tld,
    )


def date_mm_dd_yyyy_strategy() -> st.SearchStrategy[str]:
    """Generate dates in MM/DD/YYYY format."""
    return st.builds(
        lambda m, d, y: f"{m:02d}/{d:02d}/{y}",
        st.integers(min_value=1, max_value=12),
        st.integers(min_value=1, max_value=28),
        st.integers(min_value=1950, max_value=2024),
    )


def date_iso_strategy() -> st.SearchStrategy[str]:
    """Generate dates in YYYY-MM-DD format."""
    return st.builds(
        lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
        st.integers(min_value=1950, max_value=2024),
        st.integers(min_value=1, max_value=12),
        st.integers(min_value=1, max_value=28),
    )


def date_written_strategy() -> st.SearchStrategy[str]:
    """Generate dates in 'Month DD, YYYY' format."""
    months = st.sampled_from([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ])
    return st.builds(
        lambda month, day, year: f"{month} {day}, {year}",
        months,
        st.integers(min_value=1, max_value=28),
        st.integers(min_value=1950, max_value=2024),
    )


def mrn_strategy() -> st.SearchStrategy[str]:
    """Generate Medical Record Numbers in format MRN: XXXXXXXX."""
    return st.builds(
        lambda n: f"MRN: {n:08d}",
        st.integers(min_value=10000000, max_value=99999999),
    )


def ip_address_strategy() -> st.SearchStrategy[str]:
    """Generate IPv4 addresses."""
    return st.builds(
        lambda a, b, c, d: f"{a}.{b}.{c}.{d}",
        st.integers(min_value=1, max_value=254),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=1, max_value=254),
    )


def url_strategy() -> st.SearchStrategy[str]:
    """Generate URLs."""
    protocol = st.sampled_from(["http://", "https://"])
    domain = st.from_regex(r"[a-z][a-z0-9]{2,8}", fullmatch=True)
    tld = st.sampled_from(["com", "org", "net", "gov"])
    path = st.from_regex(r"/[a-z]{3,8}", fullmatch=True)
    return st.builds(
        lambda p, d, t, pa: f"{p}{d}.{t}{pa}",
        protocol,
        domain,
        tld,
        path,
    )


# --- Non-PII clinical content ---

CLINICAL_TERMS = [
    "chronic lower back pain",
    "type 2 diabetes mellitus",
    "hypertension stage II",
    "metformin 500mg twice daily",
    "lumbar MRI indicated",
    "ejection fraction 45 percent",
    "hemoglobin A1c elevated",
    "bilateral knee osteoarthritis",
    "physical therapy recommended",
    "follow up in 6 weeks",
    "diagnosis confirmed",
    "patient presents with symptoms",
    "no acute distress observed",
    "vital signs stable",
    "prescription renewed",
]


def clinical_content_strategy() -> st.SearchStrategy[str]:
    """Generate non-PII clinical sentences."""
    return st.sampled_from(CLINICAL_TERMS)


def mixed_text_strategy() -> st.SearchStrategy[tuple[str, list[str], list[str]]]:
    """Generate text mixing PII and non-PII content.

    Returns (full_text, list_of_pii_values, list_of_clinical_terms).
    """
    pii_strategy = st.one_of(
        ssn_strategy(),
        phone_parenthesized_strategy(),
        phone_dashed_strategy(),
        email_strategy(),
        date_mm_dd_yyyy_strategy(),
        date_iso_strategy(),
        date_written_strategy(),
        mrn_strategy(),
        ip_address_strategy(),
        url_strategy(),
    )

    @st.composite
    def build_mixed(draw):
        # Draw 1-3 PII values
        pii_values = draw(st.lists(pii_strategy, min_size=1, max_size=3))
        # Draw 1-3 clinical terms
        clinical_terms = draw(
            st.lists(clinical_content_strategy(), min_size=1, max_size=3)
        )

        # Interleave PII and clinical content
        parts = []
        all_items = [(pii, True) for pii in pii_values] + [
            (term, False) for term in clinical_terms
        ]
        # Use a simple deterministic interleaving
        ordering = draw(st.permutations(range(len(all_items))))
        for idx in ordering:
            parts.append(all_items[idx][0])

        full_text = ". ".join(parts) + "."
        return full_text, pii_values, clinical_terms

    return build_mixed()


# --- Property tests ---


@pytest.mark.property
class TestPIIScrubCompleteness:
    """Property 1: PII Scrubbing Completeness.

    **Validates: Requirements 1.2, 5.1**

    For any generated text containing HIPAA identifiers, scrubbing removes
    all identifier patterns while preserving non-PII content.
    """

    scrubber = PIIScrubber()

    @given(ssn=ssn_strategy(), clinical=clinical_content_strategy())
    def test_ssn_always_removed(self, ssn: str, clinical: str):
        """SSNs in XXX-XX-XXXX format are always scrubbed from output."""
        text = f"Patient data: {clinical}. SSN is {ssn}. Diagnosis confirmed."
        result = self.scrubber.scrub(text)
        assert ssn not in result, f"SSN {ssn} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(phone=phone_parenthesized_strategy(), clinical=clinical_content_strategy())
    def test_parenthesized_phone_always_removed(self, phone: str, clinical: str):
        """Phone numbers in (XXX) XXX-XXXX format are always scrubbed."""
        text = f"Contact phone: {phone}. Diagnosis: {clinical}."
        result = self.scrubber.scrub(text)
        assert phone not in result, f"Phone {phone} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(phone=phone_dashed_strategy(), clinical=clinical_content_strategy())
    def test_dashed_phone_always_removed(self, phone: str, clinical: str):
        """Phone numbers in XXX-XXX-XXXX format are always scrubbed."""
        text = f"Call {phone} for {clinical}."
        result = self.scrubber.scrub(text)
        assert phone not in result, f"Phone {phone} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(email=email_strategy(), clinical=clinical_content_strategy())
    def test_email_always_removed(self, email: str, clinical: str):
        """Email addresses are always scrubbed from output."""
        text = f"Send results to {email}. Assessment: {clinical}."
        result = self.scrubber.scrub(text)
        assert email not in result, f"Email {email} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(date=date_mm_dd_yyyy_strategy(), clinical=clinical_content_strategy())
    def test_date_mm_dd_yyyy_always_removed(self, date: str, clinical: str):
        """Dates in MM/DD/YYYY format are always scrubbed."""
        text = f"DOB: {date}. Condition: {clinical}."
        result = self.scrubber.scrub(text)
        assert date not in result, f"Date {date} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(date=date_iso_strategy(), clinical=clinical_content_strategy())
    def test_date_iso_always_removed(self, date: str, clinical: str):
        """Dates in YYYY-MM-DD format are always scrubbed."""
        text = f"Admitted: {date}. Reason: {clinical}."
        result = self.scrubber.scrub(text)
        assert date not in result, f"Date {date} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(date=date_written_strategy(), clinical=clinical_content_strategy())
    def test_date_written_always_removed(self, date: str, clinical: str):
        """Dates in 'Month DD, YYYY' format are always scrubbed."""
        text = f"Procedure scheduled for {date}. For {clinical}."
        result = self.scrubber.scrub(text)
        assert date not in result, f"Date '{date}' was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(mrn=mrn_strategy(), clinical=clinical_content_strategy())
    def test_mrn_always_removed(self, mrn: str, clinical: str):
        """Medical Record Numbers (MRN: XXXXXXXX) are always scrubbed."""
        text = f"Record: {mrn}. Diagnosis: {clinical}."
        result = self.scrubber.scrub(text)
        # Extract the numeric part after "MRN: "
        mrn_number = mrn.split(": ")[1]
        assert mrn_number not in result, f"MRN number {mrn_number} was not scrubbed"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(ip=ip_address_strategy(), clinical=clinical_content_strategy())
    def test_ip_address_always_removed(self, ip: str, clinical: str):
        """IP addresses are always scrubbed from output."""
        text = f"Access from {ip}. Patient has {clinical}."
        result = self.scrubber.scrub(text)
        assert ip not in result, f"IP address {ip} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(url=url_strategy(), clinical=clinical_content_strategy())
    def test_url_always_removed(self, url: str, clinical: str):
        """URLs are always scrubbed from output."""
        text = f"Portal: {url} for {clinical}."
        result = self.scrubber.scrub(text)
        assert url not in result, f"URL {url} was not scrubbed from output"
        assert clinical in result, f"Clinical content '{clinical}' was incorrectly removed"

    @given(data=mixed_text_strategy())
    def test_mixed_pii_all_removed_clinical_preserved(
        self, data: tuple[str, list[str], list[str]]
    ):
        """For text with mixed PII and clinical content, all PII is removed
        and all clinical content is preserved."""
        text, pii_values, clinical_terms = data
        result = self.scrubber.scrub(text)

        # All PII values must be absent from the output
        for pii in pii_values:
            # For MRN entries, check the numeric portion
            if pii.startswith("MRN: "):
                pii_check = pii.split(": ")[1]
            else:
                pii_check = pii
            assert pii_check not in result, (
                f"PII value '{pii_check}' was not scrubbed from output"
            )

        # All clinical terms must be present in the output
        for term in clinical_terms:
            assert term in result, (
                f"Clinical term '{term}' was incorrectly removed during scrubbing"
            )

    @given(clinical=clinical_content_strategy())
    def test_pure_clinical_text_unchanged(self, clinical: str):
        """Text containing only clinical content passes through unchanged."""
        text = f"Assessment: {clinical}. Plan: continue monitoring."
        result = self.scrubber.scrub(text)
        assert clinical in result, (
            f"Clinical content '{clinical}' was incorrectly modified"
        )
        # The entire text structure should remain intact
        assert "Assessment:" in result
        assert "Plan: continue monitoring." in result

    @given(ssn=ssn_strategy())
    def test_scrubbed_output_contains_redaction_token(self, ssn: str):
        """After scrubbing PII, the output contains a REDACTED token."""
        text = f"SSN: {ssn}"
        result = self.scrubber.scrub(text)
        assert "[REDACTED_" in result, (
            "Scrubbed output must contain a redaction token"
        )
        assert ssn not in result

    @given(
        ssn=ssn_strategy(),
        email=email_strategy(),
        date=date_mm_dd_yyyy_strategy(),
    )
    def test_multiple_pii_types_all_scrubbed(
        self, ssn: str, email: str, date: str
    ):
        """Multiple different PII types in same text are all scrubbed."""
        text = (
            f"Patient record - SSN: {ssn}, "
            f"contact: {email}, DOB: {date}. "
            f"Diagnosis: chronic lower back pain."
        )
        result = self.scrubber.scrub(text)
        assert ssn not in result, f"SSN {ssn} was not scrubbed"
        assert email not in result, f"Email {email} was not scrubbed"
        assert date not in result, f"Date {date} was not scrubbed"
        assert "chronic lower back pain" in result
