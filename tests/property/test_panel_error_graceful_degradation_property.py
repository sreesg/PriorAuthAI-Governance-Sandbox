"""Property-based tests for Panel Error Graceful Degradation.

**Validates: Requirements 15.9**

Property 36: Panel Error Graceful Degradation
- When a frontend API endpoint returns an error (404, 500), the response is
  still valid JSON and does not expose internal server details or stack traces.
- Uses Hypothesis to generate random request_ids/member_ids that may or may
  not exist, verifying all error responses are graceful.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest
from fastapi import FastAPI
from hypothesis import given, settings
from hypothesis import strategies as st

from src.clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    create_evidence_graph_router,
    create_frontend_router,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Random request IDs that likely don't exist in the store
random_request_id_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=50,
)

# Random member IDs that likely don't exist in the store
random_member_id_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=50,
)

# Random execution IDs for evidence bundle endpoint
random_execution_id_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=50,
)

# Sensitive patterns that should NEVER appear in error responses
SENSITIVE_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"File \".*\.py\", line \d+",
    r"raise\s+\w+Error",
    r"at 0x[0-9a-fA-F]+",
    r"/usr/.*\.py",
    r"/home/.*\.py",
    r"site-packages/",
    r"internal server error.*traceback",
    r"password",
    r"secret",
    r"api_key",
    r"connection_string",
]


# =============================================================================
# Helpers
# =============================================================================


def _create_app() -> FastAPI:
    """Create a fresh FastAPI app with all frontend routes registered."""
    storage = InMemoryAppendOnlyStorage()
    audit_service = AuditTrailService(storage_backend=storage)
    app = FastAPI()
    router = create_frontend_router(audit_service)
    app.include_router(router)
    evidence_graph_router = create_evidence_graph_router()
    app.include_router(evidence_graph_router)
    return app


def _contains_sensitive_info(text: str) -> list[str]:
    """Check if text contains any sensitive patterns.

    Returns a list of matched patterns (empty if none found).
    """
    found = []
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pattern)
    return found


async def _make_request(app: FastAPI, method: str, path: str) -> httpx.Response:
    """Make an HTTP request to the test app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        if method == "GET":
            return await client.get(path)
        return await client.get(path)


# =============================================================================
# Property 36: Panel Error Graceful Degradation
# =============================================================================


@pytest.mark.property
class TestPanelErrorGracefulDegradation:
    """Property 36: Panel Error Graceful Degradation.

    **Validates: Requirements 15.9**

    When a frontend API endpoint returns an error (404, 500), the response
    is still valid JSON, does not expose internal server details or stack
    traces, and does not affect other endpoints.
    """

    @given(request_id=random_request_id_strategy)
    @settings(max_examples=100)
    def test_beacon_status_error_is_valid_json(self, request_id: str):
        """Error responses from /beacon/status are valid JSON without internals.

        **Validates: Requirements 15.9**

        For any random request_id that may not exist, the endpoint must
        return a valid JSON response (either success or structured error)
        without exposing stack traces or server paths.
        """
        app = _create_app()
        response = asyncio.get_event_loop().run_until_complete(
            _make_request(app, "GET", f"/api/beacon/status/{request_id}")
        )

        # Response must be valid JSON regardless of status code
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            pytest.fail(
                f"Response for request_id='{request_id}' is not valid JSON: {e}"
            )

        # Response body must not contain sensitive info
        body_str = json.dumps(body)
        sensitive = _contains_sensitive_info(body_str)
        assert not sensitive, (
            f"Error response for request_id='{request_id}' exposes "
            f"sensitive info matching patterns: {sensitive}"
        )

    @given(request_id=random_request_id_strategy)
    @settings(max_examples=100)
    def test_axisweave_context_error_is_valid_json(self, request_id: str):
        """Error responses from /axisweave/context are valid JSON without internals.

        **Validates: Requirements 15.9**

        For any random request_id, the Axisweave context endpoint must
        return valid JSON and not expose internal details.
        """
        app = _create_app()
        response = asyncio.get_event_loop().run_until_complete(
            _make_request(app, "GET", f"/api/axisweave/context/{request_id}")
        )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            pytest.fail(
                f"Response for request_id='{request_id}' is not valid JSON: {e}"
            )

        body_str = json.dumps(body)
        sensitive = _contains_sensitive_info(body_str)
        assert not sensitive, (
            f"Axisweave context error for request_id='{request_id}' exposes "
            f"sensitive info matching patterns: {sensitive}"
        )

    @given(member_id=random_member_id_strategy)
    @settings(max_examples=100)
    def test_sdoh_inference_error_is_valid_json(self, member_id: str):
        """Error responses from /inference/sdoh are valid JSON without internals.

        **Validates: Requirements 15.9**

        For any random member_id, the SDOH inference endpoint must return
        valid JSON and not expose server internals.
        """
        app = _create_app()
        response = asyncio.get_event_loop().run_until_complete(
            _make_request(app, "GET", f"/api/inference/sdoh/{member_id}")
        )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            pytest.fail(
                f"Response for member_id='{member_id}' is not valid JSON: {e}"
            )

        body_str = json.dumps(body)
        sensitive = _contains_sensitive_info(body_str)
        assert not sensitive, (
            f"SDOH inference error for member_id='{member_id}' exposes "
            f"sensitive info matching patterns: {sensitive}"
        )

    @given(request_id=random_request_id_strategy)
    @settings(max_examples=100)
    def test_error_response_does_not_crash_other_endpoints(self, request_id: str):
        """An error on one endpoint does not affect other endpoints.

        **Validates: Requirements 15.9**

        After receiving an error response for one endpoint, all other
        endpoints must still be operational and return valid JSON.
        """
        app = _create_app()

        async def _test_isolation():
            # First call might return 404/200 for unknown request_id
            r1 = await _make_request(
                app, "GET", f"/api/beacon/status/{request_id}"
            )
            # Other endpoints must still work
            r2 = await _make_request(
                app, "GET", "/api/md-queue"
            )
            r3 = await _make_request(
                app, "GET", f"/api/axisweave/context/{request_id}"
            )
            return r1, r2, r3

        r1, r2, r3 = asyncio.get_event_loop().run_until_complete(_test_isolation())

        # All responses must be parseable JSON
        for i, response in enumerate([r1, r2, r3]):
            try:
                response.json()
            except (json.JSONDecodeError, ValueError) as e:
                pytest.fail(
                    f"Response {i} not valid JSON after error scenario: {e}"
                )

        # md-queue endpoint (r2) must always work since it doesn't depend on request_id
        assert r2.status_code == 200, (
            f"MD queue endpoint returned {r2.status_code} after beacon "
            f"status error for request_id='{request_id}'"
        )

    @given(execution_id=random_execution_id_strategy)
    @settings(max_examples=100)
    def test_evidence_bundle_error_is_valid_json(self, execution_id: str):
        """Error responses from /evidence-bundle are valid JSON without internals.

        **Validates: Requirements 15.9**

        For any random execution_id that doesn't exist, the evidence bundle
        endpoint must return valid JSON and not expose server internals.
        """
        app = _create_app()
        response = asyncio.get_event_loop().run_until_complete(
            _make_request(app, "GET", f"/api/evidence-bundle/{execution_id}")
        )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            pytest.fail(
                f"Response for execution_id='{execution_id}' is not valid JSON: {e}"
            )

        body_str = json.dumps(body)
        sensitive = _contains_sensitive_info(body_str)
        assert not sensitive, (
            f"Evidence bundle error for execution_id='{execution_id}' exposes "
            f"sensitive info matching patterns: {sensitive}"
        )

    @given(member_id=random_member_id_strategy)
    @settings(max_examples=100)
    def test_graph_member_error_is_valid_json(self, member_id: str):
        """Error responses from /graph/member are valid JSON without internals.

        **Validates: Requirements 15.9**

        For any random member_id that doesn't exist, the graph endpoint
        must return valid JSON and not expose server internals.
        """
        app = _create_app()
        response = asyncio.get_event_loop().run_until_complete(
            _make_request(app, "GET", f"/api/graph/member/{member_id}")
        )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            pytest.fail(
                f"Response for member_id='{member_id}' is not valid JSON: {e}"
            )

        body_str = json.dumps(body)
        sensitive = _contains_sensitive_info(body_str)
        assert not sensitive, (
            f"Graph member error for member_id='{member_id}' exposes "
            f"sensitive info matching patterns: {sensitive}"
        )

    @given(
        request_id=random_request_id_strategy,
        member_id=random_member_id_strategy,
    )
    @settings(max_examples=100)
    def test_error_status_codes_are_appropriate(
        self, request_id: str, member_id: str
    ):
        """Error responses use appropriate HTTP status codes (4xx/2xx, never 5xx).

        **Validates: Requirements 15.9**

        Graceful degradation means the server handles missing data
        as a normal condition (200 with empty data or 404 for not found),
        but never as an unhandled 500 internal server error.
        """
        app = _create_app()

        async def _test_all_endpoints():
            responses = []
            responses.append(
                await _make_request(
                    app, "GET", f"/api/beacon/status/{request_id}"
                )
            )
            responses.append(
                await _make_request(
                    app, "GET", f"/api/axisweave/context/{request_id}"
                )
            )
            responses.append(
                await _make_request(
                    app, "GET", f"/api/inference/sdoh/{member_id}"
                )
            )
            responses.append(
                await _make_request(app, "GET", "/api/md-queue")
            )
            return responses

        responses = asyncio.get_event_loop().run_until_complete(
            _test_all_endpoints()
        )

        for i, response in enumerate(responses):
            # Should never be 500 (unhandled error)
            assert response.status_code != 500, (
                f"Endpoint {i} returned 500 Internal Server Error for "
                f"request_id='{request_id}', member_id='{member_id}'"
            )
            # Valid status codes are 200 (empty data) or 404 (not found)
            assert response.status_code in (200, 404, 422), (
                f"Endpoint {i} returned unexpected status {response.status_code}"
            )
