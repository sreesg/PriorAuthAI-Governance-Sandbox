"""Property-based tests for Identity Isolation and Attribution.

**Validates: Requirements 5.3, 5.4**

Property 8: Unauthorized Response Data Isolation
- For any unauthorized request, the UnauthorizedError's to_dict() contains
  ZERO clinical data fields (no diagnosis, prescription, patient_data, evidence,
  phi, pii, ssn, mrn keys).
- The error's to_dict() contains required audit fields: identity, operation, timestamp.

Property 9: Trace Entry Identity Attribution
- For any authenticated session, every trace context created contains the correct
  identity_id that was passed in.
- All trace contexts have non-null timestamps and session_ids.
"""

import asyncio
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.identity_service import (
    Credentials,
    IdentityService,
    MaskingService,
    TraceContext,
)
from clinical_reasoning_fabric.models.core import RBACPolicy
from clinical_reasoning_fabric.models.exceptions import UnauthorizedError


# =============================================================================
# Helpers
# =============================================================================

# Clinical data field names that must NEVER appear in denial responses
CLINICAL_DATA_FIELDS = {
    "diagnosis",
    "prescription",
    "patient_data",
    "evidence",
    "phi",
    "pii",
    "ssn",
    "mrn",
    "patient_name",
    "medical_record",
    "health_plan",
    "clinical_note",
    "treatment",
    "medication",
    "lab_result",
    "vital_signs",
    "procedure",
    "condition",
}

# Required audit fields that MUST appear in denial responses
REQUIRED_AUDIT_FIELDS = {"identity", "operation", "timestamp"}


def _make_rbac_policy(
    authorized_identities: dict[str, str],
    roles: dict[str, list[str]],
) -> RBACPolicy:
    """Create an RBAC policy with given identity assignments and roles."""
    return RBACPolicy(
        policy_id="test-policy",
        roles=roles,
        identity_role_assignments=authorized_identities,
    )


def _make_identity_service(
    authorized_identities: dict[str, str] | None = None,
    roles: dict[str, list[str]] | None = None,
) -> IdentityService:
    """Create an IdentityService with configurable RBAC."""
    if roles is None:
        roles = {"reader": ["read_clinical_data", "view_patient"]}
    if authorized_identities is None:
        authorized_identities = {}

    policy = _make_rbac_policy(authorized_identities, roles)
    masking_service = MaskingService()
    return IdentityService(rbac_policy=policy, masking_service=masking_service)


# =============================================================================
# Hypothesis strategies
# =============================================================================

# Strategy for identity_ids: non-empty printable strings
identity_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

# Strategy for operations: non-empty strings representing operation names
operation_strategy = st.sampled_from([
    "read_clinical_data",
    "write_clinical_data",
    "delete_patient",
    "admin_access",
    "view_patient",
    "update_prescription",
    "approve_pa",
    "escalate_case",
])

# Strategy for request_ids: UUID-like strings
request_id_strategy = st.uuids().map(str)


@st.composite
def unauthorized_scenario_strategy(draw):
    """Generate scenarios that will result in unauthorized access.

    Possible scenarios:
    1. Identity not in RBAC policy (identity not recognized)
    2. Identity exists but operation not permitted for their role
    3. Missing credentials
    """
    scenario_type = draw(st.sampled_from([
        "identity_not_found",
        "insufficient_permissions",
        "missing_credentials",
    ]))

    identity_id = draw(identity_id_strategy)
    operation = draw(operation_strategy)

    if scenario_type == "identity_not_found":
        # Identity not in the RBAC assignments
        service = _make_identity_service(
            authorized_identities={"other-user": "reader"},
            roles={"reader": ["read_clinical_data"]},
        )
        credentials = Credentials(identity_id=identity_id, api_key="valid-key-123")
        # Make sure identity_id is NOT in authorized identities
        assume(identity_id != "other-user")

    elif scenario_type == "insufficient_permissions":
        # Identity exists but lacks permission for the operation
        service = _make_identity_service(
            authorized_identities={identity_id: "reader"},
            roles={"reader": ["read_clinical_data"]},
        )
        credentials = Credentials(identity_id=identity_id, api_key="valid-key-123")
        # Make sure the operation is NOT in reader permissions
        assume(operation != "read_clinical_data")

    else:  # missing_credentials
        service = _make_identity_service(
            authorized_identities={identity_id: "reader"},
            roles={"reader": ["read_clinical_data"]},
        )
        credentials = Credentials(identity_id=identity_id, api_key=None, token=None)

    return service, credentials, operation, identity_id


# =============================================================================
# Property 8: Unauthorized Response Data Isolation
# =============================================================================


@pytest.mark.property
class TestUnauthorizedResponseDataIsolation:
    """Property 8: Unauthorized Response Data Isolation.

    **Validates: Requirements 5.3**

    Denial responses contain zero clinical data fields; audit log contains
    required fields (identity, operation, timestamp).
    """

    @given(data=unauthorized_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_denial_contains_zero_clinical_data_fields(self, data):
        """UnauthorizedError to_dict() contains ZERO clinical data fields.

        **Validates: Requirements 5.3**

        For any unauthorized request, the error response must never expose
        clinical data such as diagnosis, prescription, patient_data, evidence,
        phi, pii, ssn, or mrn keys.
        """
        service, credentials, operation, identity_id = data

        with pytest.raises(UnauthorizedError) as exc_info:
            await service.authenticate_and_authorize(credentials, operation)

        error_dict = exc_info.value.to_dict()

        # Check that NO clinical data field key is present in the error dict
        clinical_keys_found = set(error_dict.keys()) & CLINICAL_DATA_FIELDS
        assert len(clinical_keys_found) == 0, (
            f"Clinical data fields found in denial response: {clinical_keys_found}. "
            f"Error dict keys: {set(error_dict.keys())}"
        )

    @given(data=unauthorized_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_denial_contains_required_audit_fields(self, data):
        """UnauthorizedError to_dict() contains all required audit fields.

        **Validates: Requirements 5.3**

        Every denial must contain: identity, operation, and timestamp
        for proper audit logging of unauthorized access attempts.
        """
        service, credentials, operation, identity_id = data

        with pytest.raises(UnauthorizedError) as exc_info:
            await service.authenticate_and_authorize(credentials, operation)

        error_dict = exc_info.value.to_dict()

        # Verify required audit fields are present and non-null
        for field in REQUIRED_AUDIT_FIELDS:
            assert field in error_dict, (
                f"Required audit field '{field}' missing from denial response. "
                f"Available keys: {set(error_dict.keys())}"
            )
            assert error_dict[field] is not None, (
                f"Required audit field '{field}' is None in denial response"
            )

    @given(data=unauthorized_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_denial_timestamp_is_valid_iso8601(self, data):
        """UnauthorizedError timestamp is a valid ISO-8601 timestamp.

        **Validates: Requirements 5.3**
        """
        service, credentials, operation, identity_id = data

        with pytest.raises(UnauthorizedError) as exc_info:
            await service.authenticate_and_authorize(credentials, operation)

        error_dict = exc_info.value.to_dict()

        # Verify timestamp is parseable as ISO-8601
        timestamp_str = error_dict["timestamp"]
        assert timestamp_str is not None, "Timestamp should not be None"
        try:
            parsed = datetime.fromisoformat(timestamp_str)
            assert parsed.tzinfo is not None or "Z" in timestamp_str or "+" in timestamp_str, (
                f"Timestamp should include timezone info: {timestamp_str}"
            )
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp_str}' is not valid ISO-8601")

    @given(data=unauthorized_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_denial_identity_matches_requesting_identity(self, data):
        """The identity field in denial matches the requesting identity.

        **Validates: Requirements 5.3**
        """
        service, credentials, operation, identity_id = data

        with pytest.raises(UnauthorizedError) as exc_info:
            await service.authenticate_and_authorize(credentials, operation)

        error_dict = exc_info.value.to_dict()

        assert error_dict["identity"] == identity_id, (
            f"Denial identity '{error_dict['identity']}' does not match "
            f"requesting identity '{identity_id}'"
        )


# =============================================================================
# Property 9: Trace Entry Identity Attribution
# =============================================================================


@pytest.mark.property
class TestTraceEntryIdentityAttribution:
    """Property 9: Trace Entry Identity Attribution.

    **Validates: Requirements 5.4**

    Every trace entry for an authenticated session contains the correct
    identity_id, non-null timestamps, and non-null session_ids.
    """

    @given(
        identity_id=identity_id_strategy,
        request_id=request_id_strategy,
    )
    @settings(max_examples=200)
    def test_trace_context_identity_matches_input(self, identity_id, request_id):
        """Trace context identity_id matches the authenticated identity passed in.

        **Validates: Requirements 5.4**

        Every trace entry for an authenticated session must contain the
        correct identity_id that was used during authentication.
        """
        service = _make_identity_service()
        trace_context = service.create_trace_context(identity_id, request_id)

        assert trace_context.identity_id == identity_id, (
            f"Trace context identity '{trace_context.identity_id}' does not match "
            f"input identity '{identity_id}'"
        )

    @given(
        identity_id=identity_id_strategy,
        request_id=request_id_strategy,
    )
    @settings(max_examples=200)
    def test_trace_context_has_non_null_timestamp(self, identity_id, request_id):
        """Trace context has a non-null created_at timestamp.

        **Validates: Requirements 5.4**
        """
        service = _make_identity_service()
        trace_context = service.create_trace_context(identity_id, request_id)

        assert trace_context.created_at is not None, (
            "Trace context created_at should not be None"
        )
        assert isinstance(trace_context.created_at, datetime), (
            f"Trace context created_at should be datetime, got {type(trace_context.created_at)}"
        )

    @given(
        identity_id=identity_id_strategy,
        request_id=request_id_strategy,
    )
    @settings(max_examples=200)
    def test_trace_context_has_non_null_session_id(self, identity_id, request_id):
        """Trace context has a non-null session_id.

        **Validates: Requirements 5.4**
        """
        service = _make_identity_service()
        trace_context = service.create_trace_context(identity_id, request_id)

        assert trace_context.session_id is not None, (
            "Trace context session_id should not be None"
        )
        assert len(trace_context.session_id) > 0, (
            "Trace context session_id should be non-empty"
        )

    @given(
        identity_id=identity_id_strategy,
        request_id=request_id_strategy,
    )
    @settings(max_examples=200)
    def test_trace_context_request_id_matches_input(self, identity_id, request_id):
        """Trace context request_id matches the input request_id.

        **Validates: Requirements 5.4**
        """
        service = _make_identity_service()
        trace_context = service.create_trace_context(identity_id, request_id)

        assert trace_context.request_id == request_id, (
            f"Trace context request_id '{trace_context.request_id}' does not match "
            f"input request_id '{request_id}'"
        )

    @given(
        identity_ids=st.lists(identity_id_strategy, min_size=2, max_size=10),
        request_id=request_id_strategy,
    )
    @settings(max_examples=100)
    def test_different_identities_produce_distinct_trace_contexts(
        self, identity_ids, request_id
    ):
        """Different identity_ids produce trace contexts with distinct identity_ids.

        **Validates: Requirements 5.4**

        Each trace entry is attributable to a specific identity, so
        distinct identities must produce trace contexts that correctly
        differentiate them.
        """
        assume(len(set(identity_ids)) > 1)  # Ensure at least 2 distinct identities

        service = _make_identity_service()
        contexts = [
            service.create_trace_context(iid, request_id) for iid in identity_ids
        ]

        # Each context's identity_id should match its input
        for iid, ctx in zip(identity_ids, contexts):
            assert ctx.identity_id == iid, (
                f"Context identity '{ctx.identity_id}' does not match input '{iid}'"
            )

    @given(
        identity_id=identity_id_strategy,
        request_ids=st.lists(request_id_strategy, min_size=2, max_size=5),
    )
    @settings(max_examples=100)
    def test_same_identity_different_requests_all_attributed(
        self, identity_id, request_ids
    ):
        """Same identity across multiple requests all have correct attribution.

        **Validates: Requirements 5.4**

        Even when the same identity makes multiple requests, each trace
        context correctly attributes to that identity with unique session_ids.
        """
        service = _make_identity_service()
        contexts = [
            service.create_trace_context(identity_id, rid) for rid in request_ids
        ]

        # All contexts should have the same identity_id
        for ctx in contexts:
            assert ctx.identity_id == identity_id

        # All contexts should have unique session_ids
        session_ids = [ctx.session_id for ctx in contexts]
        assert len(set(session_ids)) == len(session_ids), (
            f"Session IDs should be unique, got duplicates: {session_ids}"
        )
