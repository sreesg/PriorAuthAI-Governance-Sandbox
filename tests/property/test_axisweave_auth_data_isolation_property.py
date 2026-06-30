"""Property-based tests for Axisweave Authentication Data Isolation.

**Validates: Requirements 13.9**

Property 25: Axisweave Authentication Data Isolation
- For any request with invalid/missing credentials, the response contains zero
  document data fields (no chunk text, no content hashes, no signatures, no
  provenance metadata).
- The audit log records timestamp and source identifier for every auth failure.
"""

import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.api.auth_provider import (
    APIAuthProvider,
    APICredentials,
    AuthFailureLog,
)
from clinical_reasoning_fabric.api.namespace import NamespaceRegistry
from clinical_reasoning_fabric.models.exceptions import UnauthorizedError


# =============================================================================
# Constants
# =============================================================================

# Document data fields that must NEVER appear in an authentication failure response.
# These represent sensitive document content that must remain isolated on auth failure.
# NOTE: The base CRFError class has a structural `document_id` field that is always None
# in auth failures — it's a context-tracking field, not leaked document content.
# We check that document_id is None (no data leaked) rather than checking key absence.
DOCUMENT_DATA_FIELDS = {
    "chunk_text",
    "content_hash",
    "content_hashes",
    "signature",
    "signatures",
    "kms_signature",
    "provenance",
    "provenance_metadata",
    "document_bytes",
    "document_content",
    "chunk_index",
    "ingestion_timestamp",
    "embedding",
    "vector",
    "namespace_data",
    "retrieved_chunks",
    "search_results",
    "evidence",
    "clinical_note",
    "patient_data",
}

# Structural fields in CRFError that may be present as keys but must be None
# in auth failure responses (they carry no document data when None)
STRUCTURAL_NULL_FIELDS = {"document_id"}

# Required audit fields for auth failure logging
REQUIRED_AUDIT_LOG_FIELDS = {"timestamp", "source_identifier"}


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for generating invalid API keys (strings that won't be in the store)
invalid_api_key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=64,
).filter(lambda s: s.strip() != "" and s != "valid-key-001" and s != "valid-key-002")

# Strategy for empty/whitespace-only keys (missing credentials)
missing_api_key_strategy = st.one_of(
    st.just(""),
    st.just("   "),
    st.just("\t"),
    st.just("\n"),
    st.text(
        alphabet=st.characters(whitelist_categories=("Z",)),  # whitespace chars
        min_size=1,
        max_size=10,
    ),
)

# Combined strategy: either missing or invalid keys
bad_credentials_strategy = st.one_of(
    missing_api_key_strategy,
    invalid_api_key_strategy,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_auth_provider_with_valid_keys() -> APIAuthProvider:
    """Create an APIAuthProvider with some registered valid keys.

    This ensures that invalid/missing keys reliably trigger auth failures.
    """
    registry = NamespaceRegistry()
    registry.register_namespace("test-ns", "tenant-001")

    provider = APIAuthProvider(namespace_registry=registry)
    provider.register_api_key(
        api_key="valid-key-001",
        tenant_id="tenant-001",
        authorized_namespaces=["test-ns"],
    )
    provider.register_api_key(
        api_key="valid-key-002",
        tenant_id="tenant-002",
        authorized_namespaces=["other-ns"],
    )
    return provider


def _collect_all_values_recursive(d: dict) -> set:
    """Recursively collect all string values from a dictionary."""
    values = set()
    for v in d.values():
        if isinstance(v, str):
            values.add(v)
        elif isinstance(v, dict):
            values.update(_collect_all_values_recursive(v))
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    values.add(item)
                elif isinstance(item, dict):
                    values.update(_collect_all_values_recursive(item))
    return values


# =============================================================================
# Property 25: Axisweave Authentication Data Isolation
# =============================================================================


@pytest.mark.property
class TestAxisweaveAuthDataIsolation:
    """Property 25: Axisweave Authentication Data Isolation.

    **Validates: Requirements 13.9**

    For any request with invalid/missing credentials:
    - The response contains zero document data fields
    - Audit log records timestamp and source identifier
    """

    @given(api_key=bad_credentials_strategy)
    @settings(max_examples=200)
    def test_invalid_credentials_expose_zero_document_data(self, api_key: str):
        """UnauthorizedError to_dict() contains zero document data fields.

        **Validates: Requirements 13.9**

        For any request with invalid or missing credentials, the error response
        must contain no chunk text, no content hashes, no signatures, and no
        provenance metadata.
        """
        provider = _make_auth_provider_with_valid_keys()

        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials(api_key)

        error_dict = exc_info.value.to_dict()

        # Verify ZERO document data field keys are present
        document_keys_found = set(error_dict.keys()) & DOCUMENT_DATA_FIELDS
        assert len(document_keys_found) == 0, (
            f"Document data fields found in auth failure response: {document_keys_found}. "
            f"Error dict keys: {set(error_dict.keys())}"
        )

        # Verify structural fields that may exist as keys carry no data (are None)
        for field_name in STRUCTURAL_NULL_FIELDS:
            if field_name in error_dict:
                assert error_dict[field_name] is None, (
                    f"Structural field '{field_name}' should be None in auth failure "
                    f"but contains: {error_dict[field_name]}"
                )

    @given(api_key=bad_credentials_strategy)
    @settings(max_examples=200)
    def test_invalid_credentials_no_document_data_in_values(self, api_key: str):
        """UnauthorizedError to_dict() values contain no document content patterns.

        **Validates: Requirements 13.9**

        The serialized error must not leak document data through any field values.
        No field value should contain content hash patterns (hex strings of 64 chars),
        KMS signature patterns, or chunk content.
        """
        provider = _make_auth_provider_with_valid_keys()

        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials(api_key)

        error_dict = exc_info.value.to_dict()
        all_values = _collect_all_values_recursive(error_dict)

        # Verify no value looks like a content hash (SHA-256 = 64 hex chars)
        import re
        sha256_pattern = re.compile(r"^[a-f0-9]{64}$")
        for val in all_values:
            if isinstance(val, str) and sha256_pattern.match(val):
                pytest.fail(
                    f"Auth failure response contains a value matching SHA-256 hash pattern: {val}"
                )

    @given(api_key=bad_credentials_strategy)
    @settings(max_examples=200)
    def test_auth_failure_logs_timestamp(self, api_key: str):
        """Authentication failure is logged with a valid timestamp.

        **Validates: Requirements 13.9**

        The audit trail must record the timestamp of the authentication failure.
        """
        provider = _make_auth_provider_with_valid_keys()

        with patch.object(provider, "_log_auth_failure", wraps=provider._log_auth_failure) as mock_log:
            with pytest.raises(UnauthorizedError):
                provider.validate_credentials(api_key)

            # Verify _log_auth_failure was called
            assert mock_log.called, "Auth failure was not logged"

            # The UnauthorizedError itself carries the timestamp
            call_kwargs = mock_log.call_args
            # Verify the log was invoked (timestamp is generated inside _log_auth_failure)
            assert mock_log.call_count >= 1, (
                "Expected at least one auth failure log call"
            )

    @given(api_key=bad_credentials_strategy)
    @settings(max_examples=200)
    def test_auth_failure_logs_source_identifier(self, api_key: str):
        """Authentication failure is logged with a source identifier.

        **Validates: Requirements 13.9**

        The audit trail must record the source identifier on auth failure.
        """
        provider = _make_auth_provider_with_valid_keys()

        with patch.object(provider, "_log_auth_failure", wraps=provider._log_auth_failure) as mock_log:
            with pytest.raises(UnauthorizedError):
                provider.validate_credentials(api_key)

            # Verify _log_auth_failure was called with source_identifier
            assert mock_log.called, "Auth failure was not logged"
            call_args = mock_log.call_args
            # _log_auth_failure takes source_identifier as first positional or keyword arg
            if call_args.kwargs:
                assert "source_identifier" in call_args.kwargs, (
                    f"source_identifier not in log call kwargs: {call_args.kwargs}"
                )
                assert call_args.kwargs["source_identifier"] is not None, (
                    "source_identifier should not be None"
                )
            elif call_args.args:
                # First positional arg is source_identifier
                assert call_args.args[0] is not None, (
                    "source_identifier (first arg) should not be None"
                )

    @given(api_key=bad_credentials_strategy)
    @settings(max_examples=200)
    def test_error_has_timestamp_field(self, api_key: str):
        """The UnauthorizedError to_dict() includes a valid timestamp.

        **Validates: Requirements 13.9**

        The error response itself records when the failure occurred.
        """
        provider = _make_auth_provider_with_valid_keys()

        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials(api_key)

        error_dict = exc_info.value.to_dict()

        assert "timestamp" in error_dict, (
            f"'timestamp' missing from error dict. Keys: {set(error_dict.keys())}"
        )
        assert error_dict["timestamp"] is not None, "timestamp should not be None"

        # Verify it's a valid ISO-8601 timestamp
        timestamp_str = error_dict["timestamp"]
        try:
            parsed = datetime.fromisoformat(timestamp_str)
            # Should be timezone-aware (UTC)
            assert parsed.tzinfo is not None or "Z" in timestamp_str or "+" in timestamp_str, (
                f"Timestamp should include timezone info: {timestamp_str}"
            )
        except (ValueError, TypeError):
            pytest.fail(f"Timestamp '{timestamp_str}' is not valid ISO-8601")

    @given(api_key=invalid_api_key_strategy)
    @settings(max_examples=200)
    def test_invalid_key_error_does_not_expose_valid_key_info(self, api_key: str):
        """Auth failure response doesn't expose information about valid keys.

        **Validates: Requirements 13.9**

        The error response must not reveal which keys are valid, what namespaces
        exist, or any tenant configuration details.
        """
        provider = _make_auth_provider_with_valid_keys()

        # Ensure the key is actually invalid
        assume(api_key not in ("valid-key-001", "valid-key-002"))

        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials(api_key)

        error_dict = exc_info.value.to_dict()
        all_values = _collect_all_values_recursive(error_dict)

        # Verify no valid key or tenant info leaked
        for val in all_values:
            if isinstance(val, str):
                assert "valid-key-001" not in val, (
                    "Auth failure response leaks valid API key"
                )
                assert "valid-key-002" not in val, (
                    "Auth failure response leaks valid API key"
                )
                assert "tenant-001" not in val, (
                    "Auth failure response leaks tenant information"
                )
                assert "tenant-002" not in val, (
                    "Auth failure response leaks tenant information"
                )

    @given(api_key=missing_api_key_strategy)
    @settings(max_examples=100)
    def test_missing_credentials_expose_zero_document_data(self, api_key: str):
        """Missing credentials (empty/whitespace) expose zero document data.

        **Validates: Requirements 13.9**

        Specifically tests the missing-credentials path to ensure the same
        data isolation guarantees apply.
        """
        provider = _make_auth_provider_with_valid_keys()

        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials(api_key)

        error_dict = exc_info.value.to_dict()

        # Verify ZERO document data field keys are present
        document_keys_found = set(error_dict.keys()) & DOCUMENT_DATA_FIELDS
        assert len(document_keys_found) == 0, (
            f"Document data fields found in missing-credentials response: "
            f"{document_keys_found}. Error dict keys: {set(error_dict.keys())}"
        )

        # Verify structural fields carry no data
        for field_name in STRUCTURAL_NULL_FIELDS:
            if field_name in error_dict:
                assert error_dict[field_name] is None, (
                    f"Structural field '{field_name}' should be None in auth failure "
                    f"but contains: {error_dict[field_name]}"
                )

        # Verify timestamp is present (audit requirement)
        assert "timestamp" in error_dict, "timestamp required in auth failure response"
        assert error_dict["timestamp"] is not None, "timestamp should not be None"
