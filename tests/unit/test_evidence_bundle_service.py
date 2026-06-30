"""Unit tests for EvidenceBundleService (Evidence Bundle Producer).

Tests schema validation, lineage assembly, execution trace attachment,
and validation error handling for Evidence Bundle production.

Validates:
    - Valid bundles are produced with all required fields
    - Missing fields raise BundleValidationError with correct field lists
    - Empty lineage_trail raises BundleValidationError
    - Empty document_signatures raises BundleValidationError
    - Invalid field types are detected and reported
    - Execution trace is attached when provided
    - Validation errors halt decision and identify missing/invalid fields

Requirements referenced: 10.1, 10.2, 10.3, 10.4, 10.5, 8.3
"""

from datetime import datetime, timezone

import pytest

from clinical_reasoning_fabric.beacon.evidence_bundle_service import (
    DefaultSchemaValidator,
    EvidenceBundleService,
    SchemaValidator,
)
from clinical_reasoning_fabric.models.core import (
    EvidenceBundle,
    KMSSignature,
    LineageEntry,
    TraceCategory,
    TraceEntry,
)
from clinical_reasoning_fabric.models.exceptions import BundleValidationError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def service():
    """EvidenceBundleService with default schema validator."""
    return EvidenceBundleService()


@pytest.fixture
def valid_lineage_trail():
    """A valid lineage trail with one entry."""
    return [
        LineageEntry(
            conclusion="Patient meets medical necessity criteria for dupilumab",
            evidence_id="chunk-abc-001",
            retrieval_timestamp=datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        )
    ]


@pytest.fixture
def valid_signatures():
    """A valid list of KMS signatures."""
    return [
        KMSSignature(
            key_id="arn:aws:kms:us-east-1:123456789:key/abc-123",
            signature="YmFzZTY0c2lnbmF0dXJl",
            algorithm="RSASSA_PKCS1_V1_5_SHA_256",
            signed_at=datetime(2024, 6, 14, 8, 0, 0, tzinfo=timezone.utc),
        )
    ]


@pytest.fixture
def valid_execution_trace():
    """A valid execution trace for attaching to bundle."""
    return [
        TraceEntry(
            sequence_number=1,
            timestamp="2024-06-15T10:30:45.123Z",
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.CONTEXT_RETRIEVAL,
            details={"event": "evidence_retrieval"},
        ),
        TraceEntry(
            sequence_number=2,
            timestamp="2024-06-15T10:30:46.456Z",
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.DECISION_STEP,
            details={"event": "criteria_evaluation"},
        ),
    ]


# =============================================================================
# Test: Valid Bundle Production (Requirement 10.1)
# =============================================================================


class TestValidBundleProduction:
    """Tests that valid inputs produce a correct EvidenceBundle."""

    def test_produces_valid_bundle_with_all_fields(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Requirement 10.1: Produces bundle with all required fields."""
        bundle = service.produce_bundle(
            execution_id="exec-001",
            decision="approve",
            reason="All clinical necessity criteria met",
            lineage_trail=valid_lineage_trail,
            document_signatures=valid_signatures,
        )

        assert isinstance(bundle, EvidenceBundle)
        assert bundle.execution_id == "exec-001"
        assert bundle.decision == "approve"
        assert bundle.reason == "All clinical necessity criteria met"

    def test_lineage_trail_is_preserved_in_order(
        self, service, valid_signatures
    ):
        """Requirement 10.2: Lineage trail maintains ordering."""
        trail = [
            LineageEntry(
                conclusion="First conclusion",
                evidence_id="ev-001",
                retrieval_timestamp=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
            ),
            LineageEntry(
                conclusion="Second conclusion",
                evidence_id="ev-002",
                retrieval_timestamp=datetime(2024, 6, 15, 10, 1, 0, tzinfo=timezone.utc),
            ),
            LineageEntry(
                conclusion="Third conclusion",
                evidence_id="ev-003",
                retrieval_timestamp=datetime(2024, 6, 15, 10, 2, 0, tzinfo=timezone.utc),
            ),
        ]

        bundle = service.produce_bundle(
            execution_id="exec-002",
            decision="escalate",
            reason="Insufficient evidence for criterion 2",
            lineage_trail=trail,
            document_signatures=valid_signatures,
        )

        assert len(bundle.lineage_trail) == 3
        assert bundle.lineage_trail[0].conclusion == "First conclusion"
        assert bundle.lineage_trail[1].conclusion == "Second conclusion"
        assert bundle.lineage_trail[2].conclusion == "Third conclusion"

    def test_document_signatures_preserved(
        self, service, valid_lineage_trail
    ):
        """Requirement 10.3: KMS signatures for all referenced documents."""
        signatures = [
            KMSSignature(
                key_id="key-1",
                signature="sig1base64",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 6, 14, 8, 0, 0, tzinfo=timezone.utc),
            ),
            KMSSignature(
                key_id="key-2",
                signature="sig2base64",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 6, 14, 9, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        bundle = service.produce_bundle(
            execution_id="exec-003",
            decision="approve",
            reason="Criteria satisfied",
            lineage_trail=valid_lineage_trail,
            document_signatures=signatures,
        )

        assert len(bundle.original_document_signatures) == 2
        assert bundle.original_document_signatures[0].key_id == "key-1"
        assert bundle.original_document_signatures[1].key_id == "key-2"

    def test_execution_trace_attached_when_provided(
        self, service, valid_lineage_trail, valid_signatures, valid_execution_trace
    ):
        """Requirement 8.3: Attach execution trace to Evidence_Bundle."""
        bundle = service.produce_bundle(
            execution_id="exec-004",
            decision="approve",
            reason="All criteria met",
            lineage_trail=valid_lineage_trail,
            document_signatures=valid_signatures,
            execution_trace=valid_execution_trace,
        )

        assert bundle.execution_trace is not None
        assert len(bundle.execution_trace) == 2
        assert bundle.execution_trace[0].sequence_number == 1
        assert bundle.execution_trace[1].sequence_number == 2

    def test_execution_trace_none_when_not_provided(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Execution trace is None when not provided."""
        bundle = service.produce_bundle(
            execution_id="exec-005",
            decision="approve",
            reason="All criteria met",
            lineage_trail=valid_lineage_trail,
            document_signatures=valid_signatures,
        )

        assert bundle.execution_trace is None


# =============================================================================
# Test: Missing Fields Validation (Requirement 10.4, 10.5)
# =============================================================================


class TestMissingFieldsValidation:
    """Tests that missing fields raise BundleValidationError."""

    def test_missing_execution_id_raises_error(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Requirement 10.5: Missing execution_id halts decision."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id=None,
                decision="approve",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert "execution_id" in exc_info.value.missing_fields

    def test_missing_decision_raises_error(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Requirement 10.5: Missing decision halts decision."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision=None,
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert "decision" in exc_info.value.missing_fields

    def test_missing_reason_raises_error(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Requirement 10.5: Missing reason halts decision."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason=None,
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert "reason" in exc_info.value.missing_fields

    def test_missing_lineage_trail_raises_error(
        self, service, valid_signatures
    ):
        """Requirement 10.5: Missing lineage_trail halts decision."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="All criteria met",
                lineage_trail=None,
                document_signatures=valid_signatures,
            )

        assert "lineage_trail" in exc_info.value.missing_fields

    def test_missing_signatures_raises_error(
        self, service, valid_lineage_trail
    ):
        """Requirement 10.5: Missing signatures halts decision."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=None,
            )

        assert "original_document_signatures" in exc_info.value.missing_fields

    def test_multiple_missing_fields_reported_together(
        self, service
    ):
        """Requirement 10.5: All missing fields identified in single error."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id=None,
                decision=None,
                reason=None,
                lineage_trail=None,
                document_signatures=None,
            )

        error = exc_info.value
        assert len(error.missing_fields) == 5
        assert "execution_id" in error.missing_fields
        assert "decision" in error.missing_fields
        assert "reason" in error.missing_fields
        assert "lineage_trail" in error.missing_fields
        assert "original_document_signatures" in error.missing_fields


# =============================================================================
# Test: Empty/Invalid Fields (Requirement 10.4)
# =============================================================================


class TestEmptyAndInvalidFields:
    """Tests that empty or invalid typed fields raise BundleValidationError."""

    def test_empty_execution_id_raises_error(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Empty string execution_id is invalid."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="",
                decision="approve",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert any("execution_id" in f for f in exc_info.value.invalid_fields)

    def test_whitespace_only_execution_id_raises_error(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Whitespace-only execution_id is invalid."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="   ",
                decision="approve",
                reason="Criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert any("execution_id" in f for f in exc_info.value.invalid_fields)

    def test_empty_decision_raises_error(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Empty string decision is invalid."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert any("decision" in f for f in exc_info.value.invalid_fields)

    def test_empty_reason_raises_error(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Empty string reason is invalid."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert any("reason" in f for f in exc_info.value.invalid_fields)

    def test_empty_lineage_trail_raises_error(
        self, service, valid_signatures
    ):
        """Empty lineage_trail (0 entries) is invalid."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="All criteria met",
                lineage_trail=[],
                document_signatures=valid_signatures,
            )

        assert any("lineage_trail" in f for f in exc_info.value.invalid_fields)

    def test_empty_signatures_raises_error(
        self, service, valid_lineage_trail
    ):
        """Empty signatures (0 entries) is invalid."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=[],
            )

        assert any(
            "original_document_signatures" in f for f in exc_info.value.invalid_fields
        )


# =============================================================================
# Test: Lineage Entry Validation (Requirement 10.2)
# =============================================================================


class TestLineageEntryValidation:
    """Tests validation of individual lineage trail entries."""

    def test_lineage_entry_with_empty_conclusion_raises_error(
        self, service, valid_signatures
    ):
        """Each lineage entry must have non-empty conclusion."""
        trail = [
            LineageEntry(
                conclusion="  ",  # whitespace only — invalid
                evidence_id="ev-001",
                retrieval_timestamp=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
            )
        ]

        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="Criteria met",
                lineage_trail=trail,
                document_signatures=valid_signatures,
            )

        assert any("conclusion" in f for f in exc_info.value.invalid_fields)

    def test_lineage_entry_with_empty_evidence_id_raises_error(
        self, service, valid_signatures
    ):
        """Each lineage entry must have non-empty evidence_id."""
        trail = [
            LineageEntry(
                conclusion="Patient meets criteria",
                evidence_id="  ",  # whitespace only — invalid
                retrieval_timestamp=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
            )
        ]

        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="Criteria met",
                lineage_trail=trail,
                document_signatures=valid_signatures,
            )

        assert any("evidence_id" in f for f in exc_info.value.invalid_fields)

    def test_multiple_lineage_entries_with_one_invalid(
        self, service, valid_signatures
    ):
        """If one entry in a multi-entry trail is invalid, error is raised."""
        trail = [
            LineageEntry(
                conclusion="Valid conclusion",
                evidence_id="ev-001",
                retrieval_timestamp=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
            ),
            LineageEntry(
                conclusion="  ",  # invalid
                evidence_id="ev-002",
                retrieval_timestamp=datetime(2024, 6, 15, 10, 1, 0, tzinfo=timezone.utc),
            ),
        ]

        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="approve",
                reason="Criteria met",
                lineage_trail=trail,
                document_signatures=valid_signatures,
            )

        assert any("lineage_trail[1].conclusion" in f for f in exc_info.value.invalid_fields)


# =============================================================================
# Test: BundleValidationError Structure (Requirement 10.5)
# =============================================================================


class TestBundleValidationErrorStructure:
    """Tests that BundleValidationError carries correct diagnostic info."""

    def test_error_contains_reason_message(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Error reason describes the validation failure."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id=None,
                decision="approve",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert "validation failed" in exc_info.value.reason.lower()

    def test_error_has_missing_fields_list(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Missing fields are reported as a list."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id=None,
                decision=None,
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert isinstance(exc_info.value.missing_fields, list)
        assert len(exc_info.value.missing_fields) == 2

    def test_error_has_invalid_fields_list(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """Invalid fields are reported as a list."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="   ",
                decision="",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert isinstance(exc_info.value.invalid_fields, list)
        assert len(exc_info.value.invalid_fields) >= 2

    def test_error_serializable_via_to_dict(
        self, service, valid_lineage_trail, valid_signatures
    ):
        """BundleValidationError can be serialized for audit logging."""
        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id=None,
                decision="approve",
                reason="All criteria met",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        error_dict = exc_info.value.to_dict()
        assert "error_type" in error_dict
        assert error_dict["error_type"] == "BundleValidationError"
        assert "missing_fields" in error_dict
        assert "reason" in error_dict


# =============================================================================
# Test: Custom Schema Validator (Requirement 10.4)
# =============================================================================


class TestCustomSchemaValidator:
    """Tests that a custom SchemaValidator can be injected."""

    def test_custom_validator_errors_raise_bundle_error(
        self, valid_lineage_trail, valid_signatures
    ):
        """Custom validator returning errors halts bundle production."""

        class StrictValidator:
            def validate_bundle(self, bundle: EvidenceBundle) -> list[str]:
                return ["Custom rule: decision must be 'approve' or 'escalate'"]

        service = EvidenceBundleService(schema_validator=StrictValidator())

        with pytest.raises(BundleValidationError) as exc_info:
            service.produce_bundle(
                execution_id="exec-001",
                decision="deny",  # our custom validator doesn't actually check this
                reason="Invalid decision",
                lineage_trail=valid_lineage_trail,
                document_signatures=valid_signatures,
            )

        assert any("Custom rule" in f for f in exc_info.value.invalid_fields)

    def test_custom_validator_passes_returns_bundle(
        self, valid_lineage_trail, valid_signatures
    ):
        """Custom validator returning empty list allows bundle creation."""

        class PermissiveValidator:
            def validate_bundle(self, bundle: EvidenceBundle) -> list[str]:
                return []

        service = EvidenceBundleService(schema_validator=PermissiveValidator())

        bundle = service.produce_bundle(
            execution_id="exec-001",
            decision="approve",
            reason="All good",
            lineage_trail=valid_lineage_trail,
            document_signatures=valid_signatures,
        )

        assert isinstance(bundle, EvidenceBundle)


# =============================================================================
# Test: Default Schema Validator
# =============================================================================


class TestDefaultSchemaValidator:
    """Tests for the DefaultSchemaValidator class."""

    def test_valid_bundle_passes_validation(
        self, valid_lineage_trail, valid_signatures
    ):
        """A valid bundle returns no errors from default validator."""
        validator = DefaultSchemaValidator()
        bundle = EvidenceBundle(
            execution_id="exec-001",
            decision="approve",
            reason="All criteria met",
            lineage_trail=valid_lineage_trail,
            original_document_signatures=valid_signatures,
        )

        errors = validator.validate_bundle(bundle)
        assert errors == []
