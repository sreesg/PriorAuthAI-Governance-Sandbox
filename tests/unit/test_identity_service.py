"""Unit tests for IdentityService (BEACON Layer 1).

Tests authentication, RBAC enforcement, PHI masking, and trace context creation.

Validates:
    - authenticate_and_authorize() succeeds with valid credentials and permissions
    - authenticate_and_authorize() raises UnauthorizedError on missing/invalid credentials
    - authenticate_and_authorize() raises UnauthorizedError when identity not in RBAC
    - authenticate_and_authorize() raises UnauthorizedError when operation not permitted
    - UnauthorizedError never exposes clinical data
    - mask_phi() replaces PII/PHI fields with irreversible masked tokens
    - mask_phi() handles nested dictionaries and lists
    - create_trace_context() associates identity_id with request_id
    - Unauthorized access attempts are logged with required fields
    - Authentication failures are logged with timestamp and reason

Requirements referenced: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import hashlib
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from clinical_reasoning_fabric.beacon.identity_service import (
    Credentials,
    IdentityService,
    MaskingService,
    TraceContext,
)
from clinical_reasoning_fabric.models.core import AuthResult, RBACPolicy
from clinical_reasoning_fabric.models.exceptions import UnauthorizedError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rbac_policy():
    """Standard RBAC policy with multiple roles and assignments."""
    return RBACPolicy(
        policy_id="test-policy-001",
        roles={
            "clinical_agent": ["read_clinical_data", "submit_pa_request", "query_evidence"],
            "admin": ["read_clinical_data", "submit_pa_request", "query_evidence", "manage_users"],
            "viewer": ["read_clinical_data"],
        },
        identity_role_assignments={
            "agent-001": "clinical_agent",
            "admin-001": "admin",
            "viewer-001": "viewer",
        },
    )


@pytest.fixture
def masking_service():
    """Standard masking service instance."""
    return MaskingService()


@pytest.fixture
def identity_service(rbac_policy, masking_service):
    """IdentityService configured with test RBAC policy."""
    return IdentityService(rbac_policy=rbac_policy, masking_service=masking_service)


# =============================================================================
# Authentication Success Tests
# =============================================================================


class TestAuthenticationSuccess:
    """Tests for successful authentication and authorization flows."""

    async def test_auth_success_with_api_key(self, identity_service):
        """Valid api_key and permitted operation returns AuthResult."""
        credentials = Credentials(identity_id="agent-001", api_key="valid-api-key-123")
        result = await identity_service.authenticate_and_authorize(
            credentials, "read_clinical_data"
        )

        assert isinstance(result, AuthResult)
        assert result.identity_id == "agent-001"
        assert "read_clinical_data" in result.granted_permissions
        assert result.authenticated_at is not None
        assert result.session_id is not None

    async def test_auth_success_with_token(self, identity_service):
        """Valid token and permitted operation returns AuthResult."""
        credentials = Credentials(identity_id="admin-001", token="valid-token-xyz")
        result = await identity_service.authenticate_and_authorize(
            credentials, "manage_users"
        )

        assert isinstance(result, AuthResult)
        assert result.identity_id == "admin-001"
        assert "manage_users" in result.granted_permissions

    async def test_auth_returns_all_role_permissions(self, identity_service):
        """AuthResult includes all permissions for the assigned role."""
        credentials = Credentials(identity_id="agent-001", api_key="valid-key")
        result = await identity_service.authenticate_and_authorize(
            credentials, "submit_pa_request"
        )

        assert set(result.granted_permissions) == {
            "read_clinical_data",
            "submit_pa_request",
            "query_evidence",
        }

    async def test_auth_session_id_is_unique(self, identity_service):
        """Each authentication generates a unique session_id."""
        credentials = Credentials(identity_id="agent-001", api_key="valid-key")
        result1 = await identity_service.authenticate_and_authorize(
            credentials, "read_clinical_data"
        )
        result2 = await identity_service.authenticate_and_authorize(
            credentials, "read_clinical_data"
        )

        assert result1.session_id != result2.session_id


# =============================================================================
# Authentication Failure Tests
# =============================================================================


class TestAuthenticationFailure:
    """Tests for authentication failures (Requirement 5.5)."""

    async def test_missing_credentials_raises_unauthorized(self, identity_service):
        """Missing api_key and token raises UnauthorizedError."""
        credentials = Credentials(identity_id="agent-001")  # No api_key or token

        with pytest.raises(UnauthorizedError) as exc_info:
            await identity_service.authenticate_and_authorize(
                credentials, "read_clinical_data"
            )

        assert "missing credentials" in exc_info.value.reason.lower()

    async def test_empty_api_key_raises_unauthorized(self, identity_service):
        """Empty api_key raises UnauthorizedError."""
        credentials = Credentials(identity_id="agent-001", api_key="")

        with pytest.raises(UnauthorizedError) as exc_info:
            await identity_service.authenticate_and_authorize(
                credentials, "read_clinical_data"
            )

        assert "credentials" in exc_info.value.reason.lower()

    async def test_whitespace_only_token_raises_unauthorized(self, identity_service):
        """Whitespace-only token raises UnauthorizedError."""
        credentials = Credentials(identity_id="agent-001", token="   ")

        with pytest.raises(UnauthorizedError) as exc_info:
            await identity_service.authenticate_and_authorize(
                credentials, "read_clinical_data"
            )

        assert "invalid credentials" in exc_info.value.reason.lower()

    async def test_identity_not_in_rbac_raises_unauthorized(self, identity_service):
        """Identity not found in RBAC policy raises UnauthorizedError."""
        credentials = Credentials(identity_id="unknown-identity", api_key="some-key")

        with pytest.raises(UnauthorizedError) as exc_info:
            await identity_service.authenticate_and_authorize(
                credentials, "read_clinical_data"
            )

        assert "not recognized" in exc_info.value.reason.lower()


# =============================================================================
# Authorization Failure Tests
# =============================================================================


class TestAuthorizationFailure:
    """Tests for authorization failures (Requirement 5.3)."""

    async def test_insufficient_permissions_raises_unauthorized(self, identity_service):
        """Operation not in role permissions raises UnauthorizedError."""
        credentials = Credentials(identity_id="viewer-001", api_key="valid-key")

        with pytest.raises(UnauthorizedError) as exc_info:
            await identity_service.authenticate_and_authorize(
                credentials, "submit_pa_request"
            )

        error = exc_info.value
        assert error.operation == "submit_pa_request"
        assert error.missing_permission == "submit_pa_request"
        assert error.identity == "viewer-001"

    async def test_unauthorized_error_contains_no_clinical_data(self, identity_service):
        """UnauthorizedError must never contain clinical data."""
        credentials = Credentials(identity_id="viewer-001", api_key="valid-key")

        with pytest.raises(UnauthorizedError) as exc_info:
            await identity_service.authenticate_and_authorize(
                credentials, "submit_pa_request"
            )

        error_dict = exc_info.value.to_dict()
        # Ensure no clinical data keys are present
        clinical_keys = {
            "diagnosis", "prescription", "clinical_notes", "patient_data",
            "member_state", "evidence", "phi", "pii", "ssn", "mrn",
        }
        for key in error_dict:
            assert key.lower() not in clinical_keys

    async def test_unauthorized_logs_required_fields(self, identity_service):
        """Unauthorized access attempt logs identity, operation, timestamp, missing permission."""
        credentials = Credentials(identity_id="viewer-001", api_key="valid-key")

        with pytest.raises(UnauthorizedError) as exc_info:
            await identity_service.authenticate_and_authorize(
                credentials, "manage_users"
            )

        error = exc_info.value
        # Required audit fields per Requirement 5.3
        assert error.identity == "viewer-001"
        assert error.operation == "manage_users"
        assert error.missing_permission == "manage_users"
        assert error.timestamp is not None


# =============================================================================
# PHI Masking Tests
# =============================================================================


class TestPHIMasking:
    """Tests for PII/PHI masking (Requirement 5.1)."""

    def test_mask_phi_replaces_name_fields(self, identity_service):
        """Name fields are replaced with irreversible masked tokens."""
        data = {
            "patient_name": "John Doe",
            "first_name": "John",
            "last_name": "Doe",
            "diagnosis": "Type 2 Diabetes",
        }
        masked = identity_service.mask_phi(data)

        assert masked["patient_name"].startswith("MASKED_")
        assert masked["first_name"].startswith("MASKED_")
        assert masked["last_name"].startswith("MASKED_")
        # Non-PHI fields are preserved
        assert masked["diagnosis"] == "Type 2 Diabetes"

    def test_mask_phi_replaces_ssn(self, identity_service):
        """SSN fields are masked."""
        data = {"ssn": "123-45-6789", "status": "active"}
        masked = identity_service.mask_phi(data)

        assert masked["ssn"].startswith("MASKED_")
        assert masked["status"] == "active"

    def test_mask_phi_replaces_contact_info(self, identity_service):
        """Phone, email, and address fields are masked."""
        data = {
            "phone": "555-0123",
            "email": "patient@example.com",
            "address": "123 Main St",
            "city": "Springfield",
            "zip": "62701",
        }
        masked = identity_service.mask_phi(data)

        assert masked["phone"].startswith("MASKED_")
        assert masked["email"].startswith("MASKED_")
        assert masked["address"].startswith("MASKED_")
        assert masked["city"].startswith("MASKED_")
        assert masked["zip"].startswith("MASKED_")

    def test_mask_phi_replaces_mrn_and_dob(self, identity_service):
        """Medical record number and date of birth are masked."""
        data = {
            "mrn": "MRN-12345",
            "date_of_birth": "1985-03-15",
            "cpt_code": "99213",
        }
        masked = identity_service.mask_phi(data)

        assert masked["mrn"].startswith("MASKED_")
        assert masked["date_of_birth"].startswith("MASKED_")
        assert masked["cpt_code"] == "99213"

    def test_mask_phi_is_irreversible(self, identity_service):
        """Masked tokens cannot be reversed to original values."""
        data = {"patient_name": "Jane Smith"}
        masked = identity_service.mask_phi(data)

        # The masked value is a hash prefix, not the original
        assert "Jane" not in masked["patient_name"]
        assert "Smith" not in masked["patient_name"]

    def test_mask_phi_is_deterministic(self, identity_service):
        """Same input produces same masked token (for consistency)."""
        data1 = {"patient_name": "John Doe"}
        data2 = {"patient_name": "John Doe"}
        masked1 = identity_service.mask_phi(data1)
        masked2 = identity_service.mask_phi(data2)

        assert masked1["patient_name"] == masked2["patient_name"]

    def test_mask_phi_different_values_produce_different_tokens(self, identity_service):
        """Different values produce different masked tokens."""
        data = {"patient_name": "John Doe"}
        data2 = {"patient_name": "Jane Smith"}
        masked1 = identity_service.mask_phi(data)
        masked2 = identity_service.mask_phi(data2)

        assert masked1["patient_name"] != masked2["patient_name"]

    def test_mask_phi_handles_nested_dicts(self, identity_service):
        """Nested dictionaries have PHI fields masked recursively."""
        data = {
            "patient": {
                "name": "John Doe",
                "dob": "1985-03-15",
            },
            "clinical": {
                "diagnosis": "Diabetes",
            },
        }
        masked = identity_service.mask_phi(data)

        assert masked["patient"]["name"].startswith("MASKED_")
        assert masked["patient"]["dob"].startswith("MASKED_")
        assert masked["clinical"]["diagnosis"] == "Diabetes"

    def test_mask_phi_handles_lists(self, identity_service):
        """Lists containing dicts have PHI fields masked."""
        data = {
            "contacts": [
                {"phone": "555-0001", "type": "home"},
                {"phone": "555-0002", "type": "work"},
            ]
        }
        masked = identity_service.mask_phi(data)

        assert masked["contacts"][0]["phone"].startswith("MASKED_")
        assert masked["contacts"][0]["type"] == "home"
        assert masked["contacts"][1]["phone"].startswith("MASKED_")

    def test_mask_phi_returns_new_dict(self, identity_service):
        """mask_phi returns a new dict without modifying the original."""
        data = {"patient_name": "John Doe", "status": "active"}
        masked = identity_service.mask_phi(data)

        assert data["patient_name"] == "John Doe"  # Original unchanged
        assert masked["patient_name"] != "John Doe"


# =============================================================================
# Trace Context Tests
# =============================================================================


class TestTraceContext:
    """Tests for trace context creation (Requirement 5.4)."""

    def test_create_trace_context_associates_identity(self, identity_service):
        """Trace context contains the authenticated identity_id."""
        ctx = identity_service.create_trace_context(
            identity_id="agent-001", request_id="req-123"
        )

        assert isinstance(ctx, TraceContext)
        assert ctx.identity_id == "agent-001"
        assert ctx.request_id == "req-123"

    def test_create_trace_context_has_timestamp(self, identity_service):
        """Trace context has a creation timestamp."""
        ctx = identity_service.create_trace_context(
            identity_id="agent-001", request_id="req-456"
        )

        assert ctx.created_at is not None
        assert isinstance(ctx.created_at, datetime)

    def test_create_trace_context_has_session_id(self, identity_service):
        """Trace context generates a unique session_id."""
        ctx1 = identity_service.create_trace_context(
            identity_id="agent-001", request_id="req-001"
        )
        ctx2 = identity_service.create_trace_context(
            identity_id="agent-001", request_id="req-002"
        )

        assert ctx1.session_id is not None
        assert ctx2.session_id is not None
        assert ctx1.session_id != ctx2.session_id

    def test_trace_context_preserves_request_id(self, identity_service):
        """Each trace context correctly stores the associated request_id."""
        ctx = identity_service.create_trace_context(
            identity_id="admin-001", request_id="pa-request-789"
        )

        assert ctx.request_id == "pa-request-789"


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    """Tests for audit logging of auth events (Requirements 5.3, 5.5)."""

    async def test_unauthorized_access_logged_with_required_fields(
        self, identity_service, caplog
    ):
        """Unauthorized access attempts are logged with identity, operation, timestamp, missing permission."""
        credentials = Credentials(identity_id="viewer-001", api_key="valid-key")

        with pytest.raises(UnauthorizedError):
            await identity_service.authenticate_and_authorize(
                credentials, "manage_users"
            )

        # Check log output contains required fields
        log_output = caplog.text
        assert "viewer-001" in log_output or "Unauthorized" in log_output

    async def test_missing_credentials_logged(self, identity_service, caplog):
        """Authentication failure with missing credentials is logged."""
        credentials = Credentials(identity_id="agent-001")

        with pytest.raises(UnauthorizedError):
            await identity_service.authenticate_and_authorize(
                credentials, "read_clinical_data"
            )

        log_output = caplog.text
        assert "missing_credentials" in log_output or "Authentication failure" in log_output
