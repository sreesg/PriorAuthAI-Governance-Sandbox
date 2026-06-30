"""Unit tests for API authentication and tenant isolation.

Tests APICredentials dataclass, APIAuthProvider key validation,
tenant_id extraction, namespace access control, and failure logging.

Validates:
    - APICredentials stores api_key, tenant_id, authorized_namespaces
    - validate_credentials() accepts valid registered API keys
    - validate_credentials() rejects missing/empty/invalid API keys
    - validate_credentials() exposes zero document data on failure
    - check_namespace_access() allows access to own namespace
    - check_namespace_access() allows cross-namespace access with explicit grant
    - check_namespace_access() denies access to unauthorized namespaces
    - Authentication failures are logged with timestamp and source identifier

Requirements referenced: 13.5, 13.9
"""

import logging
from datetime import datetime, timezone

import pytest

from clinical_reasoning_fabric.api.auth_provider import (
    APIAuthProvider,
    APICredentials,
    AuthFailureLog,
)
from clinical_reasoning_fabric.api.namespace import NamespaceRegistry
from clinical_reasoning_fabric.models.exceptions import UnauthorizedError


# =============================================================================
# APICredentials Tests
# =============================================================================


class TestAPICredentials:
    """Tests for the APICredentials dataclass."""

    def test_create_credentials_with_defaults(self):
        """Create credentials with default empty authorized_namespaces."""
        creds = APICredentials(api_key="key-123", tenant_id="tenant-001")

        assert creds.api_key == "key-123"
        assert creds.tenant_id == "tenant-001"
        assert creds.authorized_namespaces == []

    def test_create_credentials_with_authorized_namespaces(self):
        """Create credentials with explicit authorized namespaces."""
        creds = APICredentials(
            api_key="key-456",
            tenant_id="tenant-002",
            authorized_namespaces=["ns-a", "ns-b", "ns-c"],
        )

        assert creds.api_key == "key-456"
        assert creds.tenant_id == "tenant-002"
        assert creds.authorized_namespaces == ["ns-a", "ns-b", "ns-c"]

    def test_credentials_are_independent_instances(self):
        """Each credentials instance has its own authorized_namespaces list."""
        creds1 = APICredentials(api_key="key-1", tenant_id="t1")
        creds2 = APICredentials(api_key="key-2", tenant_id="t2")

        creds1.authorized_namespaces.append("ns-x")
        assert "ns-x" not in creds2.authorized_namespaces


# =============================================================================
# APIAuthProvider - Registration Tests
# =============================================================================


class TestAPIAuthProviderRegistration:
    """Tests for API key registration."""

    @pytest.fixture
    def registry(self):
        """Fresh NamespaceRegistry."""
        return NamespaceRegistry()

    @pytest.fixture
    def provider(self, registry):
        """Fresh APIAuthProvider."""
        return APIAuthProvider(namespace_registry=registry)

    def test_register_api_key(self, provider):
        """Register an API key with tenant and namespaces."""
        creds = provider.register_api_key(
            api_key="ak-valid-key-12345",
            tenant_id="tenant-001",
            authorized_namespaces=["ns-a"],
        )

        assert isinstance(creds, APICredentials)
        assert creds.api_key == "ak-valid-key-12345"
        assert creds.tenant_id == "tenant-001"
        assert creds.authorized_namespaces == ["ns-a"]

    def test_register_api_key_no_namespaces(self, provider):
        """Register an API key without explicit namespace authorizations."""
        creds = provider.register_api_key(
            api_key="ak-key-no-ns",
            tenant_id="tenant-002",
        )

        assert creds.authorized_namespaces == []

    def test_register_empty_key_raises_error(self, provider):
        """Registering an empty API key raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            provider.register_api_key(api_key="", tenant_id="tenant-001")

    def test_register_whitespace_key_raises_error(self, provider):
        """Registering a whitespace-only API key raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            provider.register_api_key(api_key="   ", tenant_id="tenant-001")

    def test_register_duplicate_key_raises_error(self, provider):
        """Registering the same key twice raises ValueError."""
        provider.register_api_key(api_key="ak-dup", tenant_id="tenant-001")

        with pytest.raises(ValueError, match="already registered"):
            provider.register_api_key(api_key="ak-dup", tenant_id="tenant-002")


# =============================================================================
# APIAuthProvider - Validation Tests
# =============================================================================


class TestAPIAuthProviderValidation:
    """Tests for validate_credentials()."""

    @pytest.fixture
    def registry(self):
        return NamespaceRegistry()

    @pytest.fixture
    def provider(self, registry):
        provider = APIAuthProvider(namespace_registry=registry)
        provider.register_api_key(
            api_key="ak-valid-key-12345678",
            tenant_id="tenant-001",
            authorized_namespaces=["pa-workflow", "hedis-ns"],
        )
        return provider

    def test_validate_valid_key(self, provider):
        """Valid API key returns correct credentials."""
        creds = provider.validate_credentials("ak-valid-key-12345678")

        assert creds.api_key == "ak-valid-key-12345678"
        assert creds.tenant_id == "tenant-001"
        assert creds.authorized_namespaces == ["pa-workflow", "hedis-ns"]

    def test_validate_missing_key_raises_unauthorized(self, provider):
        """Empty string API key raises UnauthorizedError."""
        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials("")

        assert "missing" in exc_info.value.reason.lower() or "empty" in exc_info.value.reason.lower()

    def test_validate_none_key_raises_unauthorized(self, provider):
        """None API key raises UnauthorizedError."""
        with pytest.raises(UnauthorizedError):
            provider.validate_credentials(None)

    def test_validate_whitespace_key_raises_unauthorized(self, provider):
        """Whitespace-only API key raises UnauthorizedError."""
        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials("   ")

        assert "missing" in exc_info.value.reason.lower() or "empty" in exc_info.value.reason.lower()

    def test_validate_invalid_key_raises_unauthorized(self, provider):
        """Unrecognized API key raises UnauthorizedError."""
        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials("ak-wrong-key-invalid")

        assert "invalid" in exc_info.value.reason.lower()

    def test_unauthorized_exposes_no_document_data(self, provider):
        """UnauthorizedError on invalid key contains zero document data."""
        with pytest.raises(UnauthorizedError) as exc_info:
            provider.validate_credentials("ak-bad-key")

        error = exc_info.value
        error_dict = error.to_dict()

        # Ensure no document-related data leaks
        assert error.document_id is None
        assert "chunk" not in str(error_dict).lower()
        assert "content_hash" not in str(error_dict).lower()
        assert "signature" not in str(error_dict).lower()
        assert "provenance" not in str(error_dict).lower()

    def test_validate_logs_failure_on_invalid_key(self, provider, caplog):
        """Authentication failure is logged with timestamp and source identifier."""
        with caplog.at_level(logging.WARNING):
            with pytest.raises(UnauthorizedError):
                provider.validate_credentials("ak-invalid-key-xyz")

        assert len(caplog.records) >= 1
        log_message = caplog.records[0].message
        assert "timestamp=" in log_message
        assert "source=" in log_message

    def test_validate_logs_failure_on_empty_key(self, provider, caplog):
        """Authentication failure is logged for empty keys."""
        with caplog.at_level(logging.WARNING):
            with pytest.raises(UnauthorizedError):
                provider.validate_credentials("")

        assert len(caplog.records) >= 1
        log_message = caplog.records[0].message
        assert "Authentication failure" in log_message
        assert "timestamp=" in log_message


# =============================================================================
# APIAuthProvider - Namespace Access Tests
# =============================================================================


class TestAPIAuthProviderNamespaceAccess:
    """Tests for check_namespace_access()."""

    @pytest.fixture
    def registry(self):
        registry = NamespaceRegistry()
        registry.register_namespace("pa-workflow", "tenant-001")
        registry.register_namespace("hedis-ns", "tenant-002")
        registry.register_namespace("shared-ns", "tenant-003")
        return registry

    @pytest.fixture
    def provider(self, registry):
        provider = APIAuthProvider(namespace_registry=registry)
        # Key with explicit namespace authorizations
        provider.register_api_key(
            api_key="ak-tenant-001-key",
            tenant_id="tenant-001",
            authorized_namespaces=["pa-workflow"],
        )
        # Key with cross-namespace grant (via explicit scope)
        provider.register_api_key(
            api_key="ak-tenant-001-cross",
            tenant_id="tenant-001",
            authorized_namespaces=["pa-workflow", "shared-ns"],
        )
        # Key with no explicit namespaces (relies on registry ownership)
        provider.register_api_key(
            api_key="ak-tenant-002-key",
            tenant_id="tenant-002",
            authorized_namespaces=[],
        )
        return provider

    def test_access_own_namespace_via_explicit_scope(self, provider):
        """Caller can access namespace listed in their authorized_namespaces."""
        creds = provider.validate_credentials("ak-tenant-001-key")
        assert provider.check_namespace_access(creds, "pa-workflow") is True

    def test_access_cross_namespace_via_explicit_scope(self, provider):
        """Caller can access cross-namespace if explicitly in their scope."""
        creds = provider.validate_credentials("ak-tenant-001-cross")
        assert provider.check_namespace_access(creds, "shared-ns") is True

    def test_access_own_namespace_via_registry_ownership(self, provider):
        """Caller can access namespace they own in the registry."""
        creds = provider.validate_credentials("ak-tenant-002-key")
        # tenant-002 owns hedis-ns in the registry
        assert provider.check_namespace_access(creds, "hedis-ns") is True

    def test_access_via_registry_cross_namespace_grant(self, provider, registry):
        """Caller can access namespace via cross-namespace grant in registry."""
        # Grant tenant-002 (owner of hedis-ns) access to shared-ns
        registry.grant_cross_namespace_access("hedis-ns", "shared-ns")

        creds = provider.validate_credentials("ak-tenant-002-key")
        assert provider.check_namespace_access(creds, "shared-ns") is True

    def test_deny_access_to_unauthorized_namespace(self, provider):
        """Caller without access to a namespace gets UnauthorizedError."""
        creds = provider.validate_credentials("ak-tenant-001-key")

        with pytest.raises(UnauthorizedError) as exc_info:
            provider.check_namespace_access(creds, "hedis-ns")

        assert "not authorized" in exc_info.value.reason.lower() or "denied" in exc_info.value.reason.lower()

    def test_deny_exposes_no_document_data(self, provider):
        """Namespace access denial exposes zero document data."""
        creds = provider.validate_credentials("ak-tenant-001-key")

        with pytest.raises(UnauthorizedError) as exc_info:
            provider.check_namespace_access(creds, "hedis-ns")

        error = exc_info.value
        error_dict = error.to_dict()

        assert error.document_id is None
        assert "chunk" not in str(error_dict).lower()
        assert "content_hash" not in str(error_dict).lower()
        assert "signature" not in str(error_dict).lower()

    def test_deny_logs_failure_with_timestamp_and_source(self, provider, caplog):
        """Namespace access denial is logged with timestamp and source."""
        creds = provider.validate_credentials("ak-tenant-001-key")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(UnauthorizedError):
                provider.check_namespace_access(creds, "hedis-ns")

        assert len(caplog.records) >= 1
        log_message = caplog.records[0].message
        assert "timestamp=" in log_message
        assert "source=" in log_message

    def test_access_check_with_empty_authorized_namespaces(self, provider, registry):
        """Key with no explicit scope falls back to registry ownership check."""
        creds = provider.validate_credentials("ak-tenant-002-key")

        # tenant-002 owns hedis-ns — should succeed via registry
        assert provider.check_namespace_access(creds, "hedis-ns") is True

        # tenant-002 does NOT own pa-workflow and has no grants — should deny
        with pytest.raises(UnauthorizedError):
            provider.check_namespace_access(creds, "pa-workflow")


# =============================================================================
# AuthFailureLog Tests
# =============================================================================


class TestAuthFailureLog:
    """Tests for the AuthFailureLog dataclass."""

    def test_create_failure_log(self):
        """Create a failure log with all fields."""
        log = AuthFailureLog(
            timestamp="2024-01-15T10:30:00+00:00",
            source_identifier="tenant-001",
            reason="Invalid API key",
            api_key_prefix="ak-12345...",
        )

        assert log.timestamp == "2024-01-15T10:30:00+00:00"
        assert log.source_identifier == "tenant-001"
        assert log.reason == "Invalid API key"
        assert log.api_key_prefix == "ak-12345..."

    def test_create_failure_log_without_prefix(self):
        """Create a failure log without api_key_prefix (default None)."""
        log = AuthFailureLog(
            timestamp="2024-01-15T10:30:00+00:00",
            source_identifier="unknown",
            reason="Missing credentials",
        )

        assert log.api_key_prefix is None
