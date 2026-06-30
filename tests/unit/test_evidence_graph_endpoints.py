"""Unit tests for Evidence Bundle and Graph API endpoints.

Tests the GET /api/evidence-bundle/{execution_id} and
GET /api/graph/member/{member_id} endpoints using httpx AsyncClient.

Validates: Requirements 15.4, 15.5
"""

import pytest
import httpx
from fastapi import FastAPI

from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    EvidenceBundleResponse,
    MemberGraphResponse,
    create_evidence_graph_router,
    store_evidence_bundle,
    store_member_graph,
    get_evidence_bundle_store,
    get_member_graph_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_stores():
    """Clear in-memory stores before each test."""
    get_evidence_bundle_store().clear()
    get_member_graph_store().clear()
    yield
    get_evidence_bundle_store().clear()
    get_member_graph_store().clear()


@pytest.fixture
def app():
    """Create a FastAPI app with the evidence/graph router mounted."""
    app = FastAPI()
    router = create_evidence_graph_router()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create an httpx AsyncClient with ASGI transport."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ---------------------------------------------------------------------------
# Tests: GET /api/evidence-bundle/{execution_id}
# ---------------------------------------------------------------------------


class TestEvidenceBundleEndpoint:
    """Tests for the Evidence Bundle endpoint."""

    @pytest.mark.asyncio
    async def test_returns_404_when_bundle_not_found(self, client):
        """Returns 404 when execution_id does not exist."""
        response = await client.get("/api/evidence-bundle/nonexistent-exec")
        assert response.status_code == 404
        data = response.json()
        assert "nonexistent-exec" in data["detail"]

    @pytest.mark.asyncio
    async def test_returns_bundle_with_lineage_trail(self, client):
        """Returns full evidence bundle with lineage trail entries."""
        store_evidence_bundle("exec-001", {
            "decision": "approve",
            "reason": "All clinical criteria met with supporting evidence",
            "lineage_trail": [
                {
                    "conclusion": "Patient meets diagnosis requirement for ICD-10 M17.11",
                    "evidence_id": "chunk-abc-001",
                    "timestamp": "2024-06-15T10:30:00.000Z",
                    "confidence": 0.92,
                },
                {
                    "conclusion": "Conservative therapy documented for 6 months",
                    "evidence_id": "chunk-def-002",
                    "timestamp": "2024-06-15T10:30:01.000Z",
                    "confidence": 0.87,
                },
            ],
            "signatures": [
                {
                    "key_id": "arn:aws:kms:us-east-1:123456789:key/abc-123",
                    "signature": "c2lnbmF0dXJlLWJhc2U2NA==",
                    "algorithm": "RSASSA_PKCS1_V1_5_SHA_256",
                },
            ],
        })

        response = await client.get("/api/evidence-bundle/exec-001")
        assert response.status_code == 200
        data = response.json()

        assert data["execution_id"] == "exec-001"
        assert data["decision"] == "approve"
        assert data["reason"] == "All clinical criteria met with supporting evidence"
        assert len(data["lineage_trail"]) == 2
        assert data["lineage_trail"][0]["conclusion"] == "Patient meets diagnosis requirement for ICD-10 M17.11"
        assert data["lineage_trail"][0]["evidence_id"] == "chunk-abc-001"
        assert data["lineage_trail"][0]["timestamp"] == "2024-06-15T10:30:00.000Z"
        assert data["lineage_trail"][0]["confidence"] == 0.92
        assert data["lineage_trail"][1]["confidence"] == 0.87
        assert len(data["signatures"]) == 1
        assert data["signatures"][0]["key_id"] == "arn:aws:kms:us-east-1:123456789:key/abc-123"

    @pytest.mark.asyncio
    async def test_lineage_entry_without_confidence(self, client):
        """Lineage entries with no confidence return null for that field."""
        store_evidence_bundle("exec-002", {
            "decision": "escalate",
            "reason": "Indeterminate criterion found",
            "lineage_trail": [
                {
                    "conclusion": "Unable to confirm therapy duration",
                    "evidence_id": "chunk-xyz-003",
                    "timestamp": "2024-06-16T08:00:00.000Z",
                },
            ],
            "signatures": [
                {
                    "key_id": "key-001",
                    "signature": "sig-value",
                },
            ],
        })

        response = await client.get("/api/evidence-bundle/exec-002")
        assert response.status_code == 200
        data = response.json()
        assert data["lineage_trail"][0]["confidence"] is None

    @pytest.mark.asyncio
    async def test_multiple_signatures(self, client):
        """Bundle with multiple document signatures returns all."""
        store_evidence_bundle("exec-003", {
            "decision": "approve",
            "reason": "Criteria met",
            "lineage_trail": [
                {
                    "conclusion": "Diagnosis confirmed",
                    "evidence_id": "chunk-001",
                    "timestamp": "2024-07-01T12:00:00.000Z",
                    "confidence": 0.95,
                },
            ],
            "signatures": [
                {"key_id": "key-a", "signature": "sig-a", "algorithm": "RSASSA_PKCS1_V1_5_SHA_256"},
                {"key_id": "key-b", "signature": "sig-b", "algorithm": "RSASSA_PSS_SHA_256"},
            ],
        })

        response = await client.get("/api/evidence-bundle/exec-003")
        assert response.status_code == 200
        data = response.json()
        assert len(data["signatures"]) == 2
        assert data["signatures"][0]["key_id"] == "key-a"
        assert data["signatures"][1]["algorithm"] == "RSASSA_PSS_SHA_256"

    @pytest.mark.asyncio
    async def test_response_model_conformance(self, client):
        """Response conforms to EvidenceBundleResponse schema."""
        store_evidence_bundle("exec-schema", {
            "decision": "approve",
            "reason": "Test schema validation",
            "lineage_trail": [
                {
                    "conclusion": "Test conclusion",
                    "evidence_id": "ev-001",
                    "timestamp": "2024-01-01T00:00:00.000Z",
                    "confidence": 0.5,
                },
            ],
            "signatures": [
                {"key_id": "k1", "signature": "s1"},
            ],
        })

        response = await client.get("/api/evidence-bundle/exec-schema")
        assert response.status_code == 200
        bundle = EvidenceBundleResponse(**response.json())
        assert bundle.execution_id == "exec-schema"
        assert len(bundle.lineage_trail) == 1
        assert len(bundle.signatures) == 1

    @pytest.mark.asyncio
    async def test_default_algorithm_for_signature(self, client):
        """Signature without explicit algorithm defaults to RSASSA_PKCS1_V1_5_SHA_256."""
        store_evidence_bundle("exec-default-algo", {
            "decision": "approve",
            "reason": "Defaults test",
            "lineage_trail": [
                {"conclusion": "c", "evidence_id": "e", "timestamp": "2024-01-01T00:00:00Z"},
            ],
            "signatures": [
                {"key_id": "k1", "signature": "s1"},
            ],
        })

        response = await client.get("/api/evidence-bundle/exec-default-algo")
        assert response.status_code == 200
        data = response.json()
        assert data["signatures"][0]["algorithm"] == "RSASSA_PKCS1_V1_5_SHA_256"


# ---------------------------------------------------------------------------
# Tests: GET /api/graph/member/{member_id}
# ---------------------------------------------------------------------------


class TestMemberGraphEndpoint:
    """Tests for the member graph endpoint."""

    @pytest.mark.asyncio
    async def test_returns_404_when_member_not_found(self, client):
        """Returns 404 when member_id does not exist in graph store."""
        response = await client.get("/api/graph/member/unknown-member")
        assert response.status_code == 404
        data = response.json()
        assert "unknown-member" in data["detail"]

    @pytest.mark.asyncio
    async def test_returns_member_graph_with_nodes_and_edges(self, client):
        """Returns nodes and edges for a stored member graph."""
        store_member_graph("member-001", {
            "nodes": [
                {
                    "id": "member-001",
                    "type": "member",
                    "label": "Patient John Doe",
                    "properties": {"age": 55},
                },
                {
                    "id": "dx-m17-11",
                    "type": "diagnosis",
                    "label": "Primary OA, Right Knee (M17.11)",
                    "properties": {"icd10": "M17.11", "status": "active"},
                },
                {
                    "id": "rx-naproxen",
                    "type": "medication",
                    "label": "Naproxen 500mg",
                    "properties": {"dose": "500mg", "frequency": "BID"},
                },
            ],
            "edges": [
                {
                    "source": "member-001",
                    "target": "dx-m17-11",
                    "type": "HAS_CONDITION",
                    "label": "has condition",
                },
                {
                    "source": "member-001",
                    "target": "rx-naproxen",
                    "type": "IS_PRESCRIBED",
                    "label": "is prescribed",
                },
            ],
        })

        response = await client.get("/api/graph/member/member-001")
        assert response.status_code == 200
        data = response.json()

        assert data["member_id"] == "member-001"
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2

        # Verify node structure
        member_node = next(n for n in data["nodes"] if n["id"] == "member-001")
        assert member_node["type"] == "member"
        assert member_node["label"] == "Patient John Doe"
        assert member_node["properties"]["age"] == 55

        dx_node = next(n for n in data["nodes"] if n["id"] == "dx-m17-11")
        assert dx_node["type"] == "diagnosis"
        assert dx_node["properties"]["icd10"] == "M17.11"

        # Verify edge structure
        condition_edge = next(e for e in data["edges"] if e["type"] == "HAS_CONDITION")
        assert condition_edge["source"] == "member-001"
        assert condition_edge["target"] == "dx-m17-11"
        assert condition_edge["label"] == "has condition"

    @pytest.mark.asyncio
    async def test_graph_with_sdoh_factors(self, client):
        """Graph includes SDOH factor nodes with INFERRED_FROM edges."""
        store_member_graph("member-002", {
            "nodes": [
                {
                    "id": "member-002",
                    "type": "member",
                    "label": "Patient",
                    "properties": {},
                },
                {
                    "id": "sdoh-housing",
                    "type": "sdoh_factor",
                    "label": "Housing Instability",
                    "properties": {"category": "housing_instability", "origin": "inferred", "confidence": 0.72},
                },
                {
                    "id": "ev-chunk-101",
                    "type": "evidence_source",
                    "label": "Clinical Note Snippet",
                    "properties": {"document_id": "doc-555"},
                },
            ],
            "edges": [
                {
                    "source": "sdoh-housing",
                    "target": "ev-chunk-101",
                    "type": "INFERRED_FROM",
                    "label": "inferred from",
                },
            ],
        })

        response = await client.get("/api/graph/member/member-002")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 3
        sdoh_node = next(n for n in data["nodes"] if n["type"] == "sdoh_factor")
        assert sdoh_node["properties"]["origin"] == "inferred"
        assert sdoh_node["properties"]["confidence"] == 0.72

        inferred_edge = next(e for e in data["edges"] if e["type"] == "INFERRED_FROM")
        assert inferred_edge["source"] == "sdoh-housing"
        assert inferred_edge["target"] == "ev-chunk-101"

    @pytest.mark.asyncio
    async def test_graph_with_empty_nodes_and_edges(self, client):
        """Member with no clinical data returns empty arrays."""
        store_member_graph("member-empty", {
            "nodes": [],
            "edges": [],
        })

        response = await client.get("/api/graph/member/member-empty")
        assert response.status_code == 200
        data = response.json()
        assert data["member_id"] == "member-empty"
        assert data["nodes"] == []
        assert data["edges"] == []

    @pytest.mark.asyncio
    async def test_graph_with_policy_rule_nodes(self, client):
        """Graph includes policy rule nodes with GOVERNED_BY edges."""
        store_member_graph("member-003", {
            "nodes": [
                {"id": "member-003", "type": "member", "label": "Patient", "properties": {}},
                {"id": "event-knee-surgery", "type": "diagnosis", "label": "Knee Surgery Request", "properties": {}},
                {"id": "pol-surg-340", "type": "policy_rule", "label": "POL-SURG-340", "properties": {"policy_id": "POL-SURG-340"}},
            ],
            "edges": [
                {"source": "member-003", "target": "event-knee-surgery", "type": "HAS_CONDITION", "label": "has condition"},
                {"source": "event-knee-surgery", "target": "pol-surg-340", "type": "GOVERNED_BY", "label": "governed by"},
            ],
        })

        response = await client.get("/api/graph/member/member-003")
        assert response.status_code == 200
        data = response.json()
        policy_node = next(n for n in data["nodes"] if n["type"] == "policy_rule")
        assert policy_node["properties"]["policy_id"] == "POL-SURG-340"

        governed_edge = next(e for e in data["edges"] if e["type"] == "GOVERNED_BY")
        assert governed_edge["source"] == "event-knee-surgery"
        assert governed_edge["target"] == "pol-surg-340"

    @pytest.mark.asyncio
    async def test_response_model_conformance(self, client):
        """Response conforms to MemberGraphResponse schema."""
        store_member_graph("member-schema", {
            "nodes": [
                {"id": "n1", "type": "member", "label": "Test", "properties": {}},
            ],
            "edges": [],
        })

        response = await client.get("/api/graph/member/member-schema")
        assert response.status_code == 200
        graph = MemberGraphResponse(**response.json())
        assert graph.member_id == "member-schema"
        assert len(graph.nodes) == 1
        assert graph.edges == []

    @pytest.mark.asyncio
    async def test_node_properties_are_flexible_dict(self, client):
        """Node properties support arbitrary key-value pairs."""
        store_member_graph("member-props", {
            "nodes": [
                {
                    "id": "dx-complex",
                    "type": "diagnosis",
                    "label": "Complex Diagnosis",
                    "properties": {
                        "icd10": "M54.5",
                        "onset_date": "2024-01-15",
                        "severity": "moderate",
                        "laterality": "bilateral",
                        "notes_count": 3,
                    },
                },
            ],
            "edges": [],
        })

        response = await client.get("/api/graph/member/member-props")
        assert response.status_code == 200
        data = response.json()
        props = data["nodes"][0]["properties"]
        assert props["icd10"] == "M54.5"
        assert props["severity"] == "moderate"
        assert props["notes_count"] == 3
