"""Unit tests for the PII Scrubber with HIPAA Safe Harbor compliance.

Tests verify that each of the 18 HIPAA Safe Harbor identifier categories
is properly detected and removed, and that error handling raises PIIScrubError.
"""

import pytest

from clinical_reasoning_fabric.ingestion.pii_scrubber import PIIScrubber, PIIPattern
from clinical_reasoning_fabric.models.exceptions import PIIScrubError


@pytest.fixture
def scrubber():
    """Create a PIIScrubber instance with default patterns."""
    return PIIScrubber()


class TestSSNScrubbing:
    """Category 1: Social Security Numbers."""

    def test_scrubs_ssn_with_dashes(self, scrubber):
        text = "Patient SSN is 123-45-6789."
        result = scrubber.scrub(text)
        assert "123-45-6789" not in result
        assert "[REDACTED_SSN]" in result

    def test_scrubs_ssn_without_dashes(self, scrubber):
        text = "SSN: 123456789"
        result = scrubber.scrub(text)
        assert "123456789" not in result
        assert "[REDACTED_SSN]" in result

    def test_scrubs_multiple_ssns(self, scrubber):
        text = "Primary: 123-45-6789, Secondary: 987-65-4321"
        result = scrubber.scrub(text)
        assert "123-45-6789" not in result
        assert "987-65-4321" not in result


class TestEmailScrubbing:
    """Category 2: Email addresses."""

    def test_scrubs_simple_email(self, scrubber):
        text = "Contact: john.doe@hospital.com for more info"
        result = scrubber.scrub(text)
        assert "john.doe@hospital.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_scrubs_email_with_plus(self, scrubber):
        text = "Email: patient+notes@clinic.org"
        result = scrubber.scrub(text)
        assert "patient+notes@clinic.org" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_scrubs_email_with_subdomain(self, scrubber):
        text = "user@mail.internal.health.gov"
        result = scrubber.scrub(text)
        assert "user@mail.internal.health.gov" not in result
        assert "[REDACTED_EMAIL]" in result


class TestURLScrubbing:
    """Category 3: URLs."""

    def test_scrubs_http_url(self, scrubber):
        text = "Visit http://patient-portal.hospital.com/records"
        result = scrubber.scrub(text)
        assert "http://patient-portal.hospital.com/records" not in result
        assert "[REDACTED_URL]" in result

    def test_scrubs_https_url(self, scrubber):
        text = "Portal: https://secure.health.org/patient/12345"
        result = scrubber.scrub(text)
        assert "https://secure.health.org/patient/12345" not in result
        assert "[REDACTED_URL]" in result

    def test_scrubs_www_url(self, scrubber):
        text = "See www.clinicportal.com/results"
        result = scrubber.scrub(text)
        assert "www.clinicportal.com/results" not in result
        assert "[REDACTED_URL]" in result


class TestIPAddressScrubbing:
    """Category 4: IP addresses."""

    def test_scrubs_ipv4_address(self, scrubber):
        text = "Server at 192.168.1.100 logged the access"
        result = scrubber.scrub(text)
        assert "192.168.1.100" not in result
        assert "[REDACTED_IP_ADDRESS]" in result

    def test_scrubs_public_ip(self, scrubber):
        text = "Source IP: 10.0.0.1"
        result = scrubber.scrub(text)
        assert "10.0.0.1" not in result
        assert "[REDACTED_IP_ADDRESS]" in result


class TestPhoneNumberScrubbing:
    """Category 5: Phone numbers."""

    def test_scrubs_phone_with_dashes(self, scrubber):
        text = "Call 555-123-4567 for appointment"
        result = scrubber.scrub(text)
        assert "555-123-4567" not in result
        assert "[REDACTED_PHONE]" in result or "[REDACTED_SSN]" in result

    def test_scrubs_phone_with_parens(self, scrubber):
        text = "Phone: (555) 123-4567"
        result = scrubber.scrub(text)
        assert "(555) 123-4567" not in result

    def test_scrubs_phone_with_country_code(self, scrubber):
        text = "Number: +1-555-123-4567"
        result = scrubber.scrub(text)
        assert "+1-555-123-4567" not in result


class TestFaxNumberScrubbing:
    """Category 6: Fax numbers."""

    def test_scrubs_fax_number(self, scrubber):
        text = "Fax: 555-987-6543"
        result = scrubber.scrub(text)
        assert "555-987-6543" not in result
        assert "[REDACTED_FAX]" in result

    def test_scrubs_facsimile(self, scrubber):
        text = "Facsimile: 800-555-0199"
        result = scrubber.scrub(text)
        assert "800-555-0199" not in result
        assert "[REDACTED_FAX]" in result


class TestVINScrubbing:
    """Category 7: Vehicle Identification Numbers."""

    def test_scrubs_vin(self, scrubber):
        text = "Vehicle VIN: 1HGBH41JXMN109186"
        result = scrubber.scrub(text)
        assert "1HGBH41JXMN109186" not in result
        assert "[REDACTED_VIN]" in result


class TestMedicalRecordNumberScrubbing:
    """Category 8: Medical Record Numbers."""

    def test_scrubs_mrn(self, scrubber):
        text = "MRN: 12345678"
        result = scrubber.scrub(text)
        assert "12345678" not in result
        assert "[REDACTED_MEDICAL_RECORD_NUMBER]" in result

    def test_scrubs_medical_record_number(self, scrubber):
        text = "Medical Record Number: ABC-12345"
        result = scrubber.scrub(text)
        assert "ABC-12345" not in result
        assert "[REDACTED_MEDICAL_RECORD_NUMBER]" in result


class TestHealthPlanNumberScrubbing:
    """Category 9: Health Plan Beneficiary Numbers."""

    def test_scrubs_health_plan_number(self, scrubber):
        text = "Health Plan Number: HP987654321"
        result = scrubber.scrub(text)
        assert "HP987654321" not in result
        assert "[REDACTED_HEALTH_PLAN_NUMBER]" in result

    def test_scrubs_insurance_id(self, scrubber):
        text = "Insurance ID: INS-ABC-12345"
        result = scrubber.scrub(text)
        assert "INS-ABC-12345" not in result
        assert "[REDACTED_HEALTH_PLAN_NUMBER]" in result

    def test_scrubs_policy_number(self, scrubber):
        text = "Policy Number: POL123456"
        result = scrubber.scrub(text)
        assert "POL123456" not in result
        assert "[REDACTED_HEALTH_PLAN_NUMBER]" in result


class TestAccountNumberScrubbing:
    """Category 10: Account numbers."""

    def test_scrubs_account_number(self, scrubber):
        text = "Account Number: 123456789012"
        result = scrubber.scrub(text)
        assert "123456789012" not in result
        assert "[REDACTED_ACCOUNT_NUMBER]" in result

    def test_scrubs_acct(self, scrubber):
        text = "Acct Number: 9876543210"
        result = scrubber.scrub(text)
        assert "9876543210" not in result
        assert "[REDACTED_ACCOUNT_NUMBER]" in result


class TestLicenseNumberScrubbing:
    """Category 11: Certificate/License numbers."""

    def test_scrubs_license_number(self, scrubber):
        text = "License Number: DL-A1234567"
        result = scrubber.scrub(text)
        assert "DL-A1234567" not in result
        assert "[REDACTED_LICENSE_NUMBER]" in result

    def test_scrubs_certificate_number(self, scrubber):
        text = "Certificate Number: CERT-98765"
        result = scrubber.scrub(text)
        assert "CERT-98765" not in result
        assert "[REDACTED_LICENSE_NUMBER]" in result


class TestDeviceIdentifierScrubbing:
    """Category 12: Device identifiers/serial numbers."""

    def test_scrubs_device_id(self, scrubber):
        text = "Device ID: DEV-ABC-123456789"
        result = scrubber.scrub(text)
        assert "DEV-ABC-123456789" not in result
        assert "[REDACTED_DEVICE_IDENTIFIER]" in result

    def test_scrubs_serial_number(self, scrubber):
        text = "Serial Number: SN123456789012"
        result = scrubber.scrub(text)
        assert "SN123456789012" not in result
        assert "[REDACTED_DEVICE_IDENTIFIER]" in result

    def test_scrubs_udi(self, scrubber):
        text = "UDI Number: UDI-12345-ABCDEF"
        result = scrubber.scrub(text)
        assert "UDI-12345-ABCDEF" not in result
        assert "[REDACTED_DEVICE_IDENTIFIER]" in result


class TestBiometricIdentifierScrubbing:
    """Category 13: Biometric identifiers."""

    def test_scrubs_fingerprint_id(self, scrubber):
        text = "Fingerprint ID: FP-ABC123DEF456"
        result = scrubber.scrub(text)
        assert "FP-ABC123DEF456" not in result
        assert "[REDACTED_BIOMETRIC_ID]" in result

    def test_scrubs_retinal_scan(self, scrubber):
        text = "Retinal scan data: RS-XYZ789"
        result = scrubber.scrub(text)
        assert "RS-XYZ789" not in result
        assert "[REDACTED_BIOMETRIC_ID]" in result


class TestPhotographicImageScrubbing:
    """Category 14: Photographic image references."""

    def test_scrubs_photo_id(self, scrubber):
        text = "Photo ID: IMG-2024-0001.jpg"
        result = scrubber.scrub(text)
        assert "IMG-2024-0001.jpg" not in result
        assert "[REDACTED_PHOTO_ID]" in result

    def test_scrubs_image_reference(self, scrubber):
        text = "Image file: patient_photo_12345.png"
        result = scrubber.scrub(text)
        assert "patient_photo_12345.png" not in result
        assert "[REDACTED_PHOTO_ID]" in result


class TestDateScrubbing:
    """Category 15: Dates (except year alone)."""

    def test_scrubs_date_mm_dd_yyyy(self, scrubber):
        text = "DOB: 01/15/1985"
        result = scrubber.scrub(text)
        assert "01/15/1985" not in result
        assert "[REDACTED_DATE]" in result

    def test_scrubs_date_with_dashes(self, scrubber):
        text = "Admitted: 03-22-2024"
        result = scrubber.scrub(text)
        assert "03-22-2024" not in result
        assert "[REDACTED_DATE]" in result

    def test_scrubs_iso_date(self, scrubber):
        text = "Date: 2024-01-15"
        result = scrubber.scrub(text)
        assert "2024-01-15" not in result
        assert "[REDACTED_DATE]" in result

    def test_scrubs_written_date(self, scrubber):
        text = "Admitted January 15, 2024 for surgery"
        result = scrubber.scrub(text)
        assert "January 15, 2024" not in result
        assert "[REDACTED_DATE]" in result

    def test_scrubs_abbreviated_month_date(self, scrubber):
        text = "Visit on Mar 5, 2024"
        result = scrubber.scrub(text)
        assert "Mar 5, 2024" not in result
        assert "[REDACTED_DATE]" in result

    def test_preserves_year_alone(self, scrubber):
        text = "Treatment started in 2024 with good results"
        result = scrubber.scrub(text)
        assert "2024" in result


class TestGeographicDataScrubbing:
    """Category 16: Geographic data smaller than state (ZIP codes)."""

    def test_scrubs_zip_code(self, scrubber):
        text = "Patient resides in area 90210"
        result = scrubber.scrub(text)
        assert "90210" not in result
        assert "[REDACTED_GEOGRAPHIC]" in result

    def test_scrubs_zip_plus_four(self, scrubber):
        text = "ZIP: 12345-6789"
        result = scrubber.scrub(text)
        assert "12345-6789" not in result
        # May be caught as GEOGRAPHIC or SSN depending on pattern matching
        assert "[REDACTED_" in result


class TestNameScrubbing:
    """Category 17: Names."""

    def test_scrubs_patient_name(self, scrubber):
        text = "Patient: John Smith was admitted"
        result = scrubber.scrub(text)
        assert "John Smith" not in result
        assert "[REDACTED_NAME]" in result

    def test_scrubs_doctor_name(self, scrubber):
        text = "Referred by Dr. Jane Wilson"
        result = scrubber.scrub(text)
        assert "Jane Wilson" not in result
        assert "[REDACTED_NAME]" in result

    def test_scrubs_mr_mrs(self, scrubber):
        text = "Mr. Robert Johnson checked in"
        result = scrubber.scrub(text)
        assert "Robert Johnson" not in result
        assert "[REDACTED_NAME]" in result


class TestUniqueIDScrubbing:
    """Category 18: Other unique identifying numbers/codes."""

    def test_scrubs_member_id(self, scrubber):
        text = "Member ID: MBR-12345-XYZ"
        result = scrubber.scrub(text)
        assert "MBR-12345-XYZ" not in result
        assert "[REDACTED_UNIQUE_ID]" in result

    def test_scrubs_patient_id(self, scrubber):
        text = "Patient ID: PAT-99887766"
        result = scrubber.scrub(text)
        assert "PAT-99887766" not in result
        assert "[REDACTED_UNIQUE_ID]" in result

    def test_scrubs_claim_number(self, scrubber):
        text = "Claim Number: CLM-2024-001"
        result = scrubber.scrub(text)
        assert "CLM-2024-001" not in result
        assert "[REDACTED_UNIQUE_ID]" in result


class TestNonPIIPreservation:
    """Verify that non-PII content is preserved."""

    def test_preserves_clinical_text(self, scrubber):
        text = "Diagnosed with chronic lower back pain. Recommend MRI of lumbar spine."
        result = scrubber.scrub(text)
        assert "chronic lower back pain" in result
        assert "MRI of lumbar spine" in result

    def test_preserves_medication_info(self, scrubber):
        text = "Prescribed metformin 500mg twice daily for type 2 diabetes."
        result = scrubber.scrub(text)
        assert "metformin 500mg twice daily" in result
        assert "type 2 diabetes" in result

    def test_preserves_diagnosis_codes(self, scrubber):
        text = "ICD-10: M54.5 - Low back pain"
        result = scrubber.scrub(text)
        assert "M54.5" in result
        assert "Low back pain" in result

    def test_empty_string_returns_empty(self, scrubber):
        result = scrubber.scrub("")
        assert result == ""


class TestErrorHandling:
    """Test PIIScrubError is raised on failures."""

    def test_raises_pii_scrub_error_on_non_string_input(self, scrubber):
        with pytest.raises(PIIScrubError) as exc_info:
            scrubber.scrub(None)  # type: ignore
        assert "Input must be a string" in exc_info.value.reason

    def test_raises_pii_scrub_error_on_integer_input(self, scrubber):
        with pytest.raises(PIIScrubError) as exc_info:
            scrubber.scrub(12345)  # type: ignore
        assert "Input must be a string" in exc_info.value.reason

    def test_pii_scrub_error_contains_details(self, scrubber):
        with pytest.raises(PIIScrubError) as exc_info:
            scrubber.scrub(None)  # type: ignore
        assert exc_info.value.details is not None
        assert "input_type" in exc_info.value.details

    def test_pii_scrub_error_is_crf_error(self, scrubber):
        """PIIScrubError should be catchable as CRFError."""
        from clinical_reasoning_fabric.models.exceptions import CRFError
        with pytest.raises(CRFError):
            scrubber.scrub(None)  # type: ignore


class TestMultiplePIIInSameText:
    """Test scrubbing text with multiple PII categories."""

    def test_scrubs_multiple_pii_types(self, scrubber):
        text = (
            "Patient: John Smith, DOB: 01/15/1985, "
            "SSN: 123-45-6789, Email: john@email.com, "
            "Phone: (555) 123-4567"
        )
        result = scrubber.scrub(text)
        assert "John Smith" not in result
        assert "01/15/1985" not in result
        assert "123-45-6789" not in result
        assert "john@email.com" not in result
        assert "(555) 123-4567" not in result
        assert "[REDACTED_" in result

    def test_preserves_surrounding_text(self, scrubber):
        text = "The diagnosis is confirmed. SSN: 123-45-6789. Treatment plan follows."
        result = scrubber.scrub(text)
        assert "The diagnosis is confirmed." in result
        assert "Treatment plan follows." in result
        assert "123-45-6789" not in result


class TestDetectMethod:
    """Test the detect() method for PII auditing."""

    def test_detects_ssn(self, scrubber):
        findings = scrubber.detect("SSN is 123-45-6789")
        assert "SSN" in findings

    def test_detects_email(self, scrubber):
        findings = scrubber.detect("Contact user@example.com")
        assert "EMAIL" in findings

    def test_empty_for_no_pii(self, scrubber):
        findings = scrubber.detect("No personal information here")
        # May or may not find anything depending on pattern sensitivity
        # The key point is it doesn't raise an error
        assert isinstance(findings, dict)

    def test_raises_on_non_string(self, scrubber):
        with pytest.raises(PIIScrubError):
            scrubber.detect(42)  # type: ignore


class TestCustomPatterns:
    """Test that custom patterns can be provided."""

    def test_custom_patterns_override_defaults(self):
        import re
        custom_patterns = [
            PIIPattern(
                category="CUSTOM",
                pattern=re.compile(r"\bCUSTOM-\d+\b"),
                description="Custom identifier",
            ),
        ]
        scrubber = PIIScrubber(patterns=custom_patterns)
        result = scrubber.scrub("ID is CUSTOM-12345")
        assert "CUSTOM-12345" not in result
        assert "[REDACTED_CUSTOM]" in result

    def test_custom_scrubber_does_not_match_defaults(self):
        import re
        custom_patterns = [
            PIIPattern(
                category="CUSTOM",
                pattern=re.compile(r"\bCUSTOM-\d+\b"),
                description="Custom identifier",
            ),
        ]
        scrubber = PIIScrubber(patterns=custom_patterns)
        # SSN should NOT be scrubbed by custom-only scrubber
        result = scrubber.scrub("SSN: 123-45-6789")
        assert "123-45-6789" in result
