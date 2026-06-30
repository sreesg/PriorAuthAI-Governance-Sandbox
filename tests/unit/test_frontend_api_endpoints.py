"""Unit tests for frontend API endpoints (BEACON status and Axisweave context).

Tests the GET /api/beacon/status/{request_id} and
GET /api/axisweave/context/{request_id} endpoints using httpx AsyncClient.

Validates: Requirements 15.1, 15.2, 15.3
"""

import pytest
import httpx
from fastapi import FastAPI

from src.clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    BEACON_LAYERS,
    BeaconStatusResponse,
    AxisweaveContextResponse,
    create_frontend_router,
)
from src.clinical_reasoning_fabric.models.core import TraceCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_trail_service():
    """Create an AuditTrailService with in-memory storage."""
    storage = InMemoryAppendOnlyStorage()
    return AuditTrailService(storage_backend=storage)


@pytest.fixture
def app(audit_trail_service):
    """Create a FastAPI app with the frontend router mounted."""
    app = FastAPI()
    router = create_frontend_router(audit_trail_service)
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create an httpx AsyncClient with ASGI transport."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ---------------------------------------------------------------------------
# Helper to record trace entries
# ---------------------------------------------------------------------------


async def _record_entry(
    service: AuditTrailService,
    request_id: str,
    category: TraceCategory,
    identity_id: str = "test-user",
    details: dict = None,
):
    """Helper to record a trace entry."""
    await service.record_entry(
        request_id=request_id,
        identity_id=identity_id,
        category=category,
        details=details,
    )


# ---------------------------------------------------------------------------
# Tests: GET /api/beacon/status/{request_id}
# ---------------------------------------------------------------------------


class TestBeaconStatusEndpoint:
    """Tests for the BEACON status endpoint."""

    @pytest.mark.asyncio
    async def test_returns_all_pending_when_no_trace(self, client):
        """Empty trace returns all 7 layers in pending state."""
        response = await client.get("/api/beacon/status/nonexistent-request")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "nonexistent-request"
        assert len(data["layers"]) == 7
        assert data["current_layer"] == 0
        for layer in data["layers"]:
            assert layer["state"] == "pending"

    @pytest.mark.asyncio
    async def test_returns_seven_layers(self, client):
        """Response always contains exactly 7 BEACON layers."""
        response = await client.get("/api/beacon/status/any-request")
        assert response.status_code == 200
        data = response.json()
        assert len(data["layers"]) == 7
        layer_ids = [l["id"] for l in data["layers"]]
        assert layer_ids == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

    @pytest.mark.asyncio
    async def test_layer_names_match_beacon_spec(self, client):
        """Layer names match the BEACON specification."""
        response = await client.get("/api/beacon/status/any-request")
        data = response.json()
        expected_names = [
            "Identity", "Context", "MCP Gateway", "Sandbox",
            "Verification", "Observability", "Human Gates",
        ]
        actual_names = [l["name"] for l in data["layers"]]
        assert actual_names == expected_names

    @pytest.mark.asyncio
    async def test_identity_layer_passed_after_auth_trace(
        self, audit_trail_service, client
    ):
        """L1 shows 'passed' after authentication trace entry."""
        request_id = "req-auth-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.AGENT_ACTION,
            details={"authentication": "succeeded", "identity": "user-123"},
        )

        response = await client.get(f"/api/beacon/status/{request_id}")
        assert response.status_code == 200
        data = response.json()

        l1 = next(l for l in data["layers"] if l["id"] == "L1")
        assert l1["state"] == "passed"
        assert l1["timestamp"] is not None

    @pytest.mark.asyncio
    async def test_context_layer_passed_after_retrieval_trace(
        self, audit_trail_service, client
    ):
        """L2 shows 'passed' after context retrieval trace entry."""
        request_id = "req-ctx-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={"context_planner": "assembled", "chunks": []},
        )

        response = await client.get(f"/api/beacon/status/{request_id}")
        assert response.status_code == 200
        data = response.json()

        l2 = next(l for l in data["layers"] if l["id"] == "L2")
        assert l2["state"] == "passed"

    @pytest.mark.asyncio
    async def test_layer_shows_failed_on_error_details(
        self, audit_trail_service, client
    ):
        """Layer shows 'failed' when trace entry contains error details."""
        request_id = "req-fail-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.AGENT_ACTION,
            details={"authentication": "attempt", "denied": True},
        )

        response = await client.get(f"/api/beacon/status/{request_id}")
        assert response.status_code == 200
        data = response.json()

        l1 = next(l for l in data["layers"] if l["id"] == "L1")
        assert l1["state"] == "failed"

    @pytest.mark.asyncio
    async def test_multiple_layers_tracked(self, audit_trail_service, client):
        """Multiple trace entries update multiple layers."""
        request_id = "req-multi-001"
        # L1 - Identity
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.AGENT_ACTION,
            details={"authentication": "succeeded"},
        )
        # L2 - Context
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={"briefing_packet": "assembled"},
        )
        # L3 - MCP Gateway
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.TOOL_INVOCATION,
            details={"mcp_gateway": "tool_executed"},
        )

        response = await client.get(f"/api/beacon/status/{request_id}")
        assert response.status_code == 200
        data = response.json()

        l1 = next(l for l in data["layers"] if l["id"] == "L1")
        l2 = next(l for l in data["layers"] if l["id"] == "L2")
        l3 = next(l for l in data["layers"] if l["id"] == "L3")
        assert l1["state"] == "passed"
        assert l2["state"] == "passed"
        assert l3["state"] == "passed"
        # L4+ should still be pending
        l4 = next(l for l in data["layers"] if l["id"] == "L4")
        assert l4["state"] == "pending"

    @pytest.mark.asyncio
    async def test_current_layer_advances(self, audit_trail_service, client):
        """current_layer index advances as layers complete."""
        request_id = "req-advance-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.AGENT_ACTION,
            details={"identity": "ok"},
        )
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={"context_planner": "done"},
        )

        response = await client.get(f"/api/beacon/status/{request_id}")
        data = response.json()
        # L2 (index 1) should be the current layer
        assert data["current_layer"] >= 1

    @pytest.mark.asyncio
    async def test_response_model_validation(self, client):
        """Response conforms to BeaconStatusResponse schema."""
        response = await client.get("/api/beacon/status/test-schema")
        assert response.status_code == 200
        # Should be parseable as BeaconStatusResponse
        data = BeaconStatusResponse(**response.json())
        assert data.request_id == "test-schema"
        assert len(data.layers) == 7


# ---------------------------------------------------------------------------
# Tests: GET /api/axisweave/context/{request_id}
# ---------------------------------------------------------------------------


class TestAxisweaveContextEndpoint:
    """Tests for the Axisweave context endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_chunks_when_no_trace(self, client):
        """Empty trace returns empty chunks array."""
        response = await client.get("/api/axisweave/context/nonexistent-request")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "nonexistent-request"
        assert data["chunks"] == []

    @pytest.mark.asyncio
    async def test_returns_chunks_from_context_retrieval_trace(
        self, audit_trail_service, client
    ):
        """Chunks from context_retrieval trace entries are returned."""
        request_id = "req-chunks-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "context_planner": "retrieval",
                "chunks": [
                    {
                        "chunk_id": "chunk-001",
                        "text": "Patient shows signs of improvement",
                        "document_id": "doc-abc",
                        "content_hash": "a" * 64,
                        "relevance_score": 0.85,
                        "kms_status": "valid",
                        "chunk_index": 0,
                        "ingestion_timestamp": "2024-01-15T10:30:00Z",
                    },
                    {
                        "chunk_id": "chunk-002",
                        "text": "Lab results within normal range",
                        "document_id": "doc-def",
                        "content_hash": "b" * 64,
                        "relevance_score": 0.72,
                        "kms_status": "valid",
                        "chunk_index": 1,
                        "ingestion_timestamp": "2024-01-15T11:00:00Z",
                    },
                ],
            },
        )

        response = await client.get(f"/api/axisweave/context/{request_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == request_id
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_id"] == "chunk-001"
        assert data["chunks"][0]["relevance_score"] == 0.85
        assert data["chunks"][0]["kms_status"] == "valid"
        assert data["chunks"][1]["chunk_id"] == "chunk-002"

    @pytest.mark.asyncio
    async def test_returns_single_chunk_from_details(
        self, audit_trail_service, client
    ):
        """Single chunk stored directly in details is returned."""
        request_id = "req-single-chunk-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "chunk_id": "chunk-solo",
                "text": "CT scan shows no abnormalities",
                "document_id": "doc-solo",
                "content_hash": "c" * 64,
                "score": 0.91,
                "kms_status": "valid",
                "chunk_index": 0,
                "ingestion_timestamp": "2024-02-01T09:00:00Z",
            },
        )

        response = await client.get(f"/api/axisweave/context/{request_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["chunk_id"] == "chunk-solo"
        assert data["chunks"][0]["relevance_score"] == 0.91

    @pytest.mark.asyncio
    async def test_ignores_non_retrieval_trace_entries(
        self, audit_trail_service, client
    ):
        """Only context_retrieval entries are used for chunks."""
        request_id = "req-mixed-001"
        # agent_action entry - should be ignored for chunks
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.AGENT_ACTION,
            details={"identity": "user-1", "chunks": [{"chunk_id": "ignore-me"}]},
        )
        # context_retrieval entry - should be used
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "chunks": [
                    {
                        "chunk_id": "chunk-valid",
                        "text": "Valid clinical data",
                        "document_id": "doc-1",
                        "content_hash": "d" * 64,
                        "relevance_score": 0.65,
                        "kms_status": "valid",
                        "chunk_index": 2,
                        "ingestion_timestamp": "2024-03-01T12:00:00Z",
                    }
                ]
            },
        )

        response = await client.get(f"/api/axisweave/context/{request_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["chunk_id"] == "chunk-valid"

    @pytest.mark.asyncio
    async def test_skips_malformed_chunk_data(
        self, audit_trail_service, client
    ):
        """Malformed chunk data is skipped gracefully."""
        request_id = "req-malformed-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "chunks": [
                    "not-a-dict",  # Invalid: should be a dict
                    {
                        "chunk_id": "chunk-good",
                        "text": "Good data",
                        "document_id": "doc-g",
                        "content_hash": "e" * 64,
                        "relevance_score": 0.77,
                        "kms_status": "valid",
                        "chunk_index": 0,
                        "ingestion_timestamp": "2024-04-01T08:00:00Z",
                    },
                ]
            },
        )

        response = await client.get(f"/api/axisweave/context/{request_id}")
        assert response.status_code == 200
        data = response.json()
        # Only the valid chunk should be returned
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["chunk_id"] == "chunk-good"

    @pytest.mark.asyncio
    async def test_chunk_includes_all_provenance_fields(
        self, audit_trail_service, client
    ):
        """Each chunk in response includes all required provenance fields."""
        request_id = "req-provenance-001"
        await _record_entry(
            audit_trail_service,
            request_id,
            TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "chunks": [
                    {
                        "chunk_id": "chunk-prov",
                        "text": "Evidence text content",
                        "document_id": "doc-prov-123",
                        "content_hash": "f" * 64,
                        "relevance_score": 0.88,
                        "kms_status": "invalid",
                        "chunk_index": 3,
                        "ingestion_timestamp": "2024-05-15T14:30:00Z",
                    }
                ]
            },
        )

        response = await client.get(f"/api/axisweave/context/{request_id}")
        assert response.status_code == 200
        data = response.json()
        chunk = data["chunks"][0]
        # Verify all provenance fields are present
        assert chunk["chunk_id"] == "chunk-prov"
        assert chunk["text"] == "Evidence text content"
        assert chunk["document_id"] == "doc-prov-123"
        assert chunk["content_hash"] == "f" * 64
        assert chunk["relevance_score"] == 0.88
        assert chunk["kms_status"] == "invalid"
        assert chunk["chunk_index"] == 3
        assert chunk["ingestion_timestamp"] == "2024-05-15T14:30:00Z"

    @pytest.mark.asyncio
    async def test_response_model_validation(self, client):
        """Response conforms to AxisweaveContextResponse schema."""
        response = await client.get("/api/axisweave/context/test-schema")
        assert response.status_code == 200
        data = AxisweaveContextResponse(**response.json())
        assert data.request_id == "test-schema"
        assert isinstance(data.chunks, list)


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for graceful error handling."""

    @pytest.mark.asyncio
    async def test_beacon_status_graceful_on_empty_trace(self, client):
        """BEACON status returns 200 with pending layers for unknown request."""
        response = await client.get("/api/beacon/status/unknown-id")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_axisweave_context_graceful_on_empty_trace(self, client):
        """Axisweave context returns 200 with empty chunks for unknown request."""
        response = await client.get("/api/axisweave/context/unknown-id")
        assert response.status_code == 200
        data = response.json()
        assert data["chunks"] == []
