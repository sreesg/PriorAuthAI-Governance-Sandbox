"""Unit tests for the CRF custom exception hierarchy."""

import re
from datetime import datetime, timezone

import pytest

from clinical_reasoning_fabric.models.exceptions import (
    CRFError,
    BundleValidationError,
    InferenceTimeoutError,
    IngestionError,
    InvalidNamespaceError,
    KMSUnavailableError,
    MemberNotFoundError,
    PIIScrubError,
    TraceRecordingError,
    ToolValidationError,
    UnauthorizedError,
    UnmappableRecordError,
)


ISO_8601_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$"
)


class TestCRFErrorBase:
    """Tests for the base CRFError class."""

    def test_creation_with_reason_only(self):
        err = CRFError(reason="Something went wrong")
        assert err.reason == "Something went wrong"
        assert err.document_id is None
        assert err.identity is None
        assert err.details is None
        assert ISO_8601_PATTERN.match(err.timestamp)

    def test_creation_with_all_fields(self):
        err = CRFError(
            reason="Test error",
            document_id="doc-123",
            identity="user-456",
            details={"key": "value"},
        )
        assert err.reason == "Test error"
        assert err.document_id == "doc-123"
        assert err.identity == "user-456"
        assert err.details == {"key": "value"}

    def test_timestamp_is_utc_iso8601(self):
        before = datetime.now(timezone.utc)
        err = CRFError(reason="test")
        after = datetime.now(timezone.utc)

        ts = datetime.fromisoformat(err.timestamp)
        assert ts.tzinfo is not None
        assert before <= ts <= after

    def test_is_exception(self):
        err = CRFError(reason="an error")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(CRFError) as exc_info:
            raise CRFError(reason="test raise")
        assert exc_info.value.reason == "test raise"

    def test_to_dict_serialization(self):
        err = CRFError(
            reason="serialize me",
            document_id="doc-001",
            identity="agent-x",
        )
        data = err.to_dict()
        assert data["reason"] == "serialize me"
        assert data["document_id"] == "doc-001"
        assert data["identity"] == "agent-x"
        assert data["error_type"] == "CRFError"
        assert "timestamp" in data

    def test_str_representation(self):
        err = CRFError(reason="oops", document_id="doc-1", identity="user-1")
        s = str(err)
        assert "CRFError" in s
        assert "oops" in s
        assert "doc-1" in s
        assert "user-1" in s

    def test_str_without_optional_fields(self):
        err = CRFError(reason="minimal")
        s = str(err)
        assert "CRFError: minimal" in s
        assert "document_id" not in s


class TestIngestionError:
    def test_inherits_crf_error(self):
        err = IngestionError(reason="corrupted PDF", document_id="doc-bad")
        assert isinstance(err, CRFError)
        assert isinstance(err, Exception)

    def test_captures_context(self):
        err = IngestionError(
            reason="unsupported format: TIFF",
            document_id="doc-999",
            identity="ingest-service",
        )
        assert err.reason == "unsupported format: TIFF"
        assert err.document_id == "doc-999"
        data = err.to_dict()
        assert data["error_type"] == "IngestionError"


class TestPIIScrubError:
    def test_inherits_crf_error(self):
        err = PIIScrubError(reason="scrubbing failed")
        assert isinstance(err, CRFError)

    def test_captures_document_context(self):
        err = PIIScrubError(
            reason="regex timeout on large document",
            document_id="doc-large",
        )
        data = err.to_dict()
        assert data["error_type"] == "PIIScrubError"
        assert data["document_id"] == "doc-large"


class TestKMSUnavailableError:
    def test_inherits_crf_error(self):
        err = KMSUnavailableError(reason="KMS service timeout")
        assert isinstance(err, CRFError)

    def test_serialization(self):
        err = KMSUnavailableError(
            reason="connection refused",
            document_id="doc-sign-fail",
            details={"endpoint": "kms.us-east-1.amazonaws.com"},
        )
        data = err.to_dict()
        assert data["error_type"] == "KMSUnavailableError"
        assert data["details"]["endpoint"] == "kms.us-east-1.amazonaws.com"


class TestUnauthorizedError:
    def test_inherits_crf_error(self):
        err = UnauthorizedError(reason="access denied")
        assert isinstance(err, CRFError)

    def test_captures_operation_and_permission(self):
        err = UnauthorizedError(
            reason="insufficient permissions",
            identity="user-no-access",
            operation="read_clinical_data",
            missing_permission="clinical:read",
        )
        assert err.operation == "read_clinical_data"
        assert err.missing_permission == "clinical:read"
        data = err.to_dict()
        assert data["operation"] == "read_clinical_data"
        assert data["missing_permission"] == "clinical:read"
        assert data["error_type"] == "UnauthorizedError"

    def test_no_clinical_data_in_error(self):
        """UnauthorizedError must not expose clinical data."""
        err = UnauthorizedError(
            reason="access denied",
            identity="rogue-user",
            operation="view_patient",
            missing_permission="phi:access",
        )
        data = err.to_dict()
        # Ensure no clinical-sounding fields leaked
        for key in data:
            assert "clinical" not in str(data[key]).lower() or key == "operation"


class TestTraceRecordingError:
    def test_inherits_crf_error(self):
        err = TraceRecordingError(reason="storage full")
        assert isinstance(err, CRFError)

    def test_captures_request_id(self):
        err = TraceRecordingError(
            reason="write failure",
            request_id="req-abc-123",
        )
        assert err.request_id == "req-abc-123"
        data = err.to_dict()
        assert data["request_id"] == "req-abc-123"
        assert data["error_type"] == "TraceRecordingError"


class TestMemberNotFoundError:
    def test_inherits_crf_error(self):
        err = MemberNotFoundError(reason="member not in graph")
        assert isinstance(err, CRFError)

    def test_captures_member_id(self):
        err = MemberNotFoundError(
            reason="no matching member node",
            member_id="MBR-12345",
        )
        assert err.member_id == "MBR-12345"
        data = err.to_dict()
        assert data["member_id"] == "MBR-12345"
        assert data["error_type"] == "MemberNotFoundError"


class TestBundleValidationError:
    def test_inherits_crf_error(self):
        err = BundleValidationError(reason="schema mismatch")
        assert isinstance(err, CRFError)

    def test_captures_field_details(self):
        err = BundleValidationError(
            reason="missing required fields",
            missing_fields=["execution_id", "lineage_trail"],
            invalid_fields=["decision"],
        )
        assert err.missing_fields == ["execution_id", "lineage_trail"]
        assert err.invalid_fields == ["decision"]
        data = err.to_dict()
        assert data["missing_fields"] == ["execution_id", "lineage_trail"]
        assert data["invalid_fields"] == ["decision"]
        assert data["error_type"] == "BundleValidationError"


class TestUnmappableRecordError:
    def test_inherits_crf_error(self):
        err = UnmappableRecordError(reason="unknown entity type")
        assert isinstance(err, CRFError)

    def test_captures_source_info(self):
        err = UnmappableRecordError(
            reason="entity_type 'FooBar' not in mapping",
            source_record_id="src-rec-789",
            entity_type="FooBar",
        )
        assert err.source_record_id == "src-rec-789"
        assert err.entity_type == "FooBar"
        data = err.to_dict()
        assert data["source_record_id"] == "src-rec-789"
        assert data["entity_type"] == "FooBar"
        assert data["error_type"] == "UnmappableRecordError"


class TestToolValidationError:
    def test_inherits_crf_error(self):
        err = ToolValidationError(reason="tool not in catalog")
        assert isinstance(err, CRFError)

    def test_captures_tool_details(self):
        err = ToolValidationError(
            reason="invalid parameters",
            identity="agent-001",
            tool_name="unapproved_tool",
            validation_errors=["param 'x' is required", "param 'y' has wrong type"],
        )
        assert err.tool_name == "unapproved_tool"
        assert len(err.validation_errors) == 2
        data = err.to_dict()
        assert data["tool_name"] == "unapproved_tool"
        assert data["error_type"] == "ToolValidationError"


class TestInvalidNamespaceError:
    def test_inherits_crf_error(self):
        err = InvalidNamespaceError(reason="invalid format")
        assert isinstance(err, CRFError)

    def test_captures_namespace(self):
        err = InvalidNamespaceError(
            reason="namespace contains special characters",
            namespace="bad namespace!@#",
        )
        assert err.namespace == "bad namespace!@#"
        data = err.to_dict()
        assert data["namespace"] == "bad namespace!@#"
        assert data["error_type"] == "InvalidNamespaceError"


class TestInferenceTimeoutError:
    def test_inherits_crf_error(self):
        err = InferenceTimeoutError(reason="exceeded 15s limit")
        assert isinstance(err, CRFError)

    def test_captures_snippet_and_timeout(self):
        err = InferenceTimeoutError(
            reason="inference timed out",
            snippet_id="chunk-abc-123",
            timeout_seconds=15.0,
        )
        assert err.snippet_id == "chunk-abc-123"
        assert err.timeout_seconds == 15.0
        data = err.to_dict()
        assert data["snippet_id"] == "chunk-abc-123"
        assert data["timeout_seconds"] == 15.0
        assert data["error_type"] == "InferenceTimeoutError"

    def test_default_timeout(self):
        err = InferenceTimeoutError(reason="timeout")
        assert err.timeout_seconds == 15.0


class TestExceptionHierarchyCatchAll:
    """Verify all exceptions can be caught as CRFError."""

    def test_all_exceptions_caught_by_base(self):
        exceptions = [
            IngestionError(reason="test"),
            PIIScrubError(reason="test"),
            KMSUnavailableError(reason="test"),
            UnauthorizedError(reason="test"),
            TraceRecordingError(reason="test"),
            MemberNotFoundError(reason="test"),
            BundleValidationError(reason="test"),
            UnmappableRecordError(reason="test"),
            ToolValidationError(reason="test"),
            InvalidNamespaceError(reason="test"),
            InferenceTimeoutError(reason="test"),
        ]
        for exc in exceptions:
            with pytest.raises(CRFError):
                raise exc

    def test_all_exceptions_have_timestamp(self):
        exceptions = [
            IngestionError(reason="test"),
            PIIScrubError(reason="test"),
            KMSUnavailableError(reason="test"),
            UnauthorizedError(reason="test"),
            TraceRecordingError(reason="test"),
            MemberNotFoundError(reason="test"),
            BundleValidationError(reason="test"),
            UnmappableRecordError(reason="test"),
            ToolValidationError(reason="test"),
            InvalidNamespaceError(reason="test"),
            InferenceTimeoutError(reason="test"),
        ]
        for exc in exceptions:
            assert ISO_8601_PATTERN.match(exc.timestamp), (
                f"{type(exc).__name__} has invalid timestamp: {exc.timestamp}"
            )
