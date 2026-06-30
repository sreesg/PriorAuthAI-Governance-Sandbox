"""Unit tests for SDOH Inference and Medical Director Queue API endpoints.

Tests the GET /api/inference/sdoh/{member_id} and
GET /api/md-queue endpoints using httpx AsyncClient.

Validates: Requirements 15.6, 15.7
"""

import pytest
import httpx
from fastapi import FastAPI

from src.clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    SDOHInferenceResponse,
    MDQueueResponse,
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
# Tests: GET /api/inference/sdoh/{member_id}
# ---------------------------------------------------------------------------


class TestSDOHInferenceEndpoint:
    """Tests for the SDOH inference endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_member_entries(self, client):
        """No trace entries for a member returns empty arrays."""
        response = await client.get("/api/inference/sdoh/member-unknown")
        assert response.status_code == 200
        data = response.json()
        assert data["member_id"] == "member-unknown"
        assert data["inferred_facts"] == []
        assert data["explicit_facts"] == []

    @pytest.mark.asyncio
    async def test_returns_inferred_facts_from_trace(
        self, audit_trail_service, client
    ):
        """Inferred SDOH facts from trace entries are returned with full chain."""
        await _record_entry(
            audit_trail_service,
            "req-sdoh-001",
            TraceCategory.AGENT_ACTION,
            details={
                "member_id": "member-123",
                "inferred_facts": [
                    {
                        "fact_id": "fact-001",
                        "type": "sdoh_factor",
                        "category": "housing_instability",
                        "conclusion": "Patient likely has housing instability",
                        "confidence": 0.78,
                        "source_text": "Patient reports frequent moves and difficulty storing medications",
                        "chain": {
                            "chain_id": "chain-001",
                            "hops": [
                                {
                                    "hop_number": 1,
                                    "source_text": "Patient reports frequent moves",
                                    "intermediate_conclusion": "Housing instability indicated",
                                    "confidence": 0.78,
                                }
                            ],
                            "cumulative_confidence": 0.78,
                            "final_conclusion": "Patient likely has housing instability",
                        },
                    }
                ],
            },
        )

        response = await client.get("/api/inference/sdoh/member-123")
        assert response.status_code == 200
        data = response.json()
        assert data["member_id"] == "member-123"
        assert len(data["inferred_facts"]) == 1

        fact = data["inferred_facts"][0]
        assert fact["fact_id"] == "fact-001"
        assert fact["type"] == "sdoh_factor"
        assert fact["category"] == "housing_instability"
        assert fact["conclusion"] == "Patient likely has housing instability"
        assert fact["confidence"] == 0.78
        assert fact["origin"] == "inferred"
        assert fact["source_text"] == "Patient reports frequent moves and difficulty storing medications"

        # Verify chain
        chain = fact["chain"]
        assert chain["chain_id"] == "chain-001"
        assert len(chain["hops"]) == 1
        assert chain["hops"][0]["hop_number"] == 1
        assert chain["cumulative_confidence"] == 0.78

    @pytest.mark.asyncio
    async def test_returns_explicit_facts_from_trace(
        self, audit_trail_service, client
    ):
        """Explicit facts stored in trace entries are returned."""
        await _record_entry(
            audit_trail_service,
            "req-sdoh-002",
            TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "member_id": "member-456",
                "explicit_facts": [
                    {
                        "fact_id": "explicit-001",
                        "type": "sdoh_factor",
                        "category": "food_insecurity",
                        "conclusion": "Documented food insecurity",
                    }
                ],
            },
        )

        response = await client.get("/api/inference/sdoh/member-456")
        assert response.status_code == 200
        data = response.json()
        assert len(data["explicit_facts"]) == 1
        assert data["explicit_facts"][0]["fact_id"] == "explicit-001"
        assert data["explicit_facts"][0]["origin"] == "explicit"
        assert data["explicit_facts"][0]["category"] == "food_insecurity"

    @pytest.mark.asyncio
    async def test_returns_sdoh_factors_from_graph_query(
        self, audit_trail_service, client
    ):
        """SDOH factors from graph query results are correctly categorized."""
        await _record_entry(
            audit_trail_service,
            "req-sdoh-003",
            TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "member_id": "member-789",
                "sdoh_factors": [
                    {
                        "sdoh_id": "sdoh-explicit-01",
                        "sdoh_category": "transportation_barriers",
                        "description": "No reliable transportation",
                        "origin": "explicit",
                    },
                    {
                        "sdoh_id": "sdoh-inferred-01",
                        "sdoh_category": "medication_storage_limitations",
                        "conclusion": "Medication storage issues inferred",
                        "confidence": 0.65,
                        "source_text": "Patient mentions lack of refrigeration at home",
                        "origin": "inferred",
                        "chain": {
                            "chain_id": "chain-002",
                            "hops": [
                                {
                                    "hop_number": 1,
                                    "source_text": "Patient mentions lack of refrigeration",
                                    "intermediate_conclusion": "No cold storage available",
                                    "confidence": 0.81,
                                },
                                {
                                    "hop_number": 2,
                                    "source_text": "No cold storage available",
                                    "intermediate_conclusion": "Medication storage issues",
                                    "confidence": 0.80,
                                },
                            ],
                            "cumulative_confidence": 0.65,
                            "final_conclusion": "Medication storage issues inferred",
                        },
                    },
                ],
            },
        )

        response = await client.get("/api/inference/sdoh/member-789")
        assert response.status_code == 200
        data = response.json()

        # Check explicit factor
        assert len(data["explicit_facts"]) == 1
        explicit = data["explicit_facts"][0]
        assert explicit["fact_id"] == "sdoh-explicit-01"
        assert explicit["type"] == "sdoh_factor"
        assert explicit["category"] == "transportation_barriers"
        assert explicit["origin"] == "explicit"

        # Check inferred factor
        assert len(data["inferred_facts"]) == 1
        inferred = data["inferred_facts"][0]
        assert inferred["fact_id"] == "sdoh-inferred-01"
        assert inferred["type"] == "sdoh_factor"
        assert inferred["category"] == "medication_storage_limitations"
        assert inferred["confidence"] == 0.65
        assert inferred["origin"] == "inferred"
        assert len(inferred["chain"]["hops"]) == 2

    @pytest.mark.asyncio
    async def test_multiple_inference_types(
        self, audit_trail_service, client
    ):
        """Multiple inference types (sdoh, adherence risk, care barrier) are returned."""
        await _record_entry(
            audit_trail_service,
            "req-sdoh-004",
            TraceCategory.AGENT_ACTION,
            details={
                "member_id": "member-multi",
                "inferred_facts": [
                    {
                        "fact_id": "fact-sdoh",
                        "type": "sdoh_factor",
                        "category": "food_insecurity",
                        "conclusion": "Food insecurity likely",
                        "confidence": 0.72,
                        "source_text": "Mentions skipping meals",
                        "chain": {
                            "chain_id": "c1",
                            "hops": [{"hop_number": 1, "source_text": "x", "intermediate_conclusion": "y", "confidence": 0.72}],
                            "cumulative_confidence": 0.72,
                            "final_conclusion": "Food insecurity likely",
                        },
                    },
                    {
                        "fact_id": "fact-adherence",
                        "type": "medication_adherence_risk",
                        "conclusion": "Adherence risk due to cost",
                        "confidence": 0.60,
                        "source_text": "Patient asked about generic alternatives",
                        "chain": {
                            "chain_id": "c2",
                            "hops": [{"hop_number": 1, "source_text": "x", "intermediate_conclusion": "y", "confidence": 0.60}],
                            "cumulative_confidence": 0.60,
                            "final_conclusion": "Adherence risk due to cost",
                        },
                    },
                    {
                        "fact_id": "fact-barrier",
                        "type": "care_access_barrier",
                        "conclusion": "Transportation barrier to clinic",
                        "confidence": 0.55,
                        "source_text": "Patient missed appointment, cites distance",
                        "chain": {
                            "chain_id": "c3",
                            "hops": [{"hop_number": 1, "source_text": "x", "intermediate_conclusion": "y", "confidence": 0.55}],
                            "cumulative_confidence": 0.55,
                            "final_conclusion": "Transportation barrier to clinic",
                        },
                    },
                ],
            },
        )

        response = await client.get("/api/inference/sdoh/member-multi")
        assert response.status_code == 200
        data = response.json()
        assert len(data["inferred_facts"]) == 3

        types = {f["type"] for f in data["inferred_facts"]}
        assert types == {"sdoh_factor", "medication_adherence_risk", "care_access_barrier"}

    @pytest.mark.asyncio
    async def test_source_text_truncated_to_500_chars(
        self, audit_trail_service, client
    ):
        """Source text is truncated to 500 characters maximum."""
        long_text = "A" * 600
        await _record_entry(
            audit_trail_service,
            "req-sdoh-005",
            TraceCategory.AGENT_ACTION,
            details={
                "member_id": "member-long",
                "inferred_facts": [
                    {
                        "fact_id": "fact-long",
                        "type": "sdoh_factor",
                        "category": "housing_instability",
                        "conclusion": "Test",
                        "confidence": 0.5,
                        "source_text": long_text,
                        "chain": {
                            "chain_id": "c1",
                            "hops": [{"hop_number": 1, "source_text": "x", "intermediate_conclusion": "y", "confidence": 0.5}],
                            "cumulative_confidence": 0.5,
                            "final_conclusion": "Test",
                        },
                    }
                ],
            },
        )

        response = await client.get("/api/inference/sdoh/member-long")
        assert response.status_code == 200
        data = response.json()
        assert len(data["inferred_facts"][0]["source_text"]) == 500

    @pytest.mark.asyncio
    async def test_response_model_validation(self, client):
        """Response conforms to SDOHInferenceResponse schema."""
        response = await client.get("/api/inference/sdoh/test-schema")
        assert response.status_code == 200
        data = SDOHInferenceResponse(**response.json())
        assert data.member_id == "test-schema"
        assert isinstance(data.inferred_facts, list)
        assert isinstance(data.explicit_facts, list)

    @pytest.mark.asyncio
    async def test_skips_malformed_inferred_fact_data(
        self, audit_trail_service, client
    ):
        """Malformed inferred fact data is skipped gracefully."""
        await _record_entry(
            audit_trail_service,
            "req-sdoh-malform",
            TraceCategory.AGENT_ACTION,
            details={
                "member_id": "member-bad",
                "inferred_facts": [
                    "not-a-dict",
                    {
                        "fact_id": "good-fact",
                        "type": "sdoh_factor",
                        "category": "food_insecurity",
                        "conclusion": "Valid fact",
                        "confidence": 0.8,
                        "source_text": "Good data",
                        "chain": {
                            "chain_id": "cg",
                            "hops": [{"hop_number": 1, "source_text": "x", "intermediate_conclusion": "y", "confidence": 0.8}],
                            "cumulative_confidence": 0.8,
                            "final_conclusion": "Valid fact",
                        },
                    },
                ],
            },
        )

        response = await client.get("/api/inference/sdoh/member-bad")
        assert response.status_code == 200
        data = response.json()
        assert len(data["inferred_facts"]) == 1
        assert data["inferred_facts"][0]["fact_id"] == "good-fact"


# ---------------------------------------------------------------------------
# Tests: GET /api/md-queue
# ---------------------------------------------------------------------------


class TestMDQueueEndpoint:
    """Tests for the Medical Director queue endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_escalations(self, client):
        """No escalation entries returns empty cases array."""
        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        assert data["cases"] == []

    @pytest.mark.asyncio
    async def test_returns_escalated_case_from_trace(
        self, audit_trail_service, client
    ):
        """Escalated case is returned with all artifact summaries."""
        await _record_entry(
            audit_trail_service,
            "req-esc-001",
            TraceCategory.DECISION_STEP,
            details={
                "escalation": {
                    "case_id": "case-001",
                    "briefing_summary": "PA for MRI lumbar spine, member has chronic back pain",
                    "criteria_assessment": [
                        {"criterion": "Medical necessity", "status": "met"},
                        {"criterion": "Conservative therapy", "status": "not_met"},
                        {"criterion": "Duration requirement", "status": "indeterminate"},
                    ],
                    "challenger_findings": {
                        "verification_result": "FAIL",
                        "tamper_alerts": [],
                        "violated_rules": ["POL-RAD-501.rule3"],
                    },
                    "trace_summary": "7 trace entries, L1-L7 processed in 12.5s",
                    "escalated_at": "2024-06-15T14:30:00.000Z",
                },
            },
        )

        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        assert len(data["cases"]) == 1

        case = data["cases"][0]
        assert case["case_id"] == "case-001"
        assert case["briefing_summary"] == "PA for MRI lumbar spine, member has chronic back pain"
        assert len(case["criteria_assessment"]) == 3
        assert case["criteria_assessment"][0]["criterion"] == "Medical necessity"
        assert case["criteria_assessment"][0]["status"] == "met"
        assert case["criteria_assessment"][1]["status"] == "not_met"
        assert case["criteria_assessment"][2]["status"] == "indeterminate"
        assert "Verification: FAIL" in case["challenger_findings"]
        assert "Violated rules: 1" in case["challenger_findings"]
        assert case["trace_summary"] == "7 trace entries, L1-L7 processed in 12.5s"
        assert case["escalated_at"] == "2024-06-15T14:30:00.000Z"

    @pytest.mark.asyncio
    async def test_returns_multiple_escalated_cases(
        self, audit_trail_service, client
    ):
        """Multiple escalation entries produce multiple cases."""
        await _record_entry(
            audit_trail_service,
            "req-esc-002",
            TraceCategory.DECISION_STEP,
            details={
                "escalation": {
                    "case_id": "case-A",
                    "briefing_summary": "Case A summary",
                    "criteria_assessment": [
                        {"criterion": "Criterion 1", "status": "not_met"},
                    ],
                    "challenger_findings": "Signature verification failed",
                    "trace_summary": "5 entries",
                    "escalated_at": "2024-06-16T10:00:00.000Z",
                },
            },
        )
        await _record_entry(
            audit_trail_service,
            "req-esc-003",
            TraceCategory.DECISION_STEP,
            details={
                "escalation": {
                    "case_id": "case-B",
                    "briefing_summary": "Case B summary",
                    "criteria_assessment": [
                        {"criterion": "Criterion 1", "status": "met"},
                        {"criterion": "Criterion 2", "status": "not_met"},
                    ],
                    "challenger_findings": "OPA rule violation detected",
                    "trace_summary": "8 entries",
                    "escalated_at": "2024-06-16T11:00:00.000Z",
                },
            },
        )

        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        assert len(data["cases"]) == 2
        case_ids = {c["case_id"] for c in data["cases"]}
        assert case_ids == {"case-A", "case-B"}

    @pytest.mark.asyncio
    async def test_case_from_md_queue_cases_list(
        self, audit_trail_service, client
    ):
        """Cases from md_queue_cases list in trace details are returned."""
        await _record_entry(
            audit_trail_service,
            "req-esc-004",
            TraceCategory.DECISION_STEP,
            details={
                "md_queue_cases": [
                    {
                        "case_id": "case-list-1",
                        "briefing_summary": "Listed case summary",
                        "criteria_assessment": [
                            {"criterion": "C1", "status": "met"},
                        ],
                        "challenger_findings": "No issues",
                        "trace_summary": "3 entries",
                        "escalated_at": "2024-06-17T09:00:00.000Z",
                    }
                ],
            },
        )

        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        assert len(data["cases"]) == 1
        assert data["cases"][0]["case_id"] == "case-list-1"
        assert data["cases"][0]["briefing_summary"] == "Listed case summary"

    @pytest.mark.asyncio
    async def test_case_uses_request_id_when_no_case_id(
        self, audit_trail_service, client
    ):
        """When case_id is not provided, request_id is used as fallback."""
        await _record_entry(
            audit_trail_service,
            "req-fallback-id",
            TraceCategory.DECISION_STEP,
            details={
                "escalation": {
                    "briefing_summary": "Case without explicit ID",
                    "criteria_assessment": [],
                    "challenger_findings": "",
                    "trace_summary": "",
                    "escalated_at": "2024-06-18T12:00:00.000Z",
                },
            },
        )

        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        assert len(data["cases"]) == 1
        assert data["cases"][0]["case_id"] == "req-fallback-id"

    @pytest.mark.asyncio
    async def test_challenger_findings_dict_formatted(
        self, audit_trail_service, client
    ):
        """Challenger findings as dict are formatted into summary string."""
        await _record_entry(
            audit_trail_service,
            "req-esc-dict",
            TraceCategory.DECISION_STEP,
            details={
                "escalation": {
                    "case_id": "case-dict",
                    "briefing_summary": "Test",
                    "criteria_assessment": [],
                    "challenger_findings": {
                        "verification_result": "PASS",
                        "tamper_alerts": ["alert-1", "alert-2"],
                        "violated_rules": ["rule-A"],
                    },
                    "trace_summary": "test",
                    "escalated_at": "2024-06-19T08:00:00.000Z",
                },
            },
        )

        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        findings = data["cases"][0]["challenger_findings"]
        assert "Tamper alerts: 2" in findings
        assert "Violated rules: 1" in findings
        assert "Verification: PASS" in findings

    @pytest.mark.asyncio
    async def test_ignores_non_escalation_entries(
        self, audit_trail_service, client
    ):
        """Non-escalation trace entries don't appear in MD queue."""
        # Regular agent action - should not appear
        await _record_entry(
            audit_trail_service,
            "req-normal",
            TraceCategory.AGENT_ACTION,
            details={"identity": "user-1", "authentication": "success"},
        )
        # Context retrieval - should not appear
        await _record_entry(
            audit_trail_service,
            "req-normal",
            TraceCategory.CONTEXT_RETRIEVAL,
            details={"context_planner": "assembled", "chunks": []},
        )

        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        assert data["cases"] == []

    @pytest.mark.asyncio
    async def test_response_model_validation(self, client):
        """Response conforms to MDQueueResponse schema."""
        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = MDQueueResponse(**response.json())
        assert isinstance(data.cases, list)

    @pytest.mark.asyncio
    async def test_criteria_status_normalized_to_lowercase(
        self, audit_trail_service, client
    ):
        """Criteria status values are normalized to lowercase with underscores."""
        await _record_entry(
            audit_trail_service,
            "req-esc-norm",
            TraceCategory.DECISION_STEP,
            details={
                "escalation": {
                    "case_id": "case-norm",
                    "briefing_summary": "Test normalization",
                    "criteria_assessment": [
                        {"criterion": "C1", "status": "MET"},
                        {"criterion": "C2", "status": "NOT MET"},
                        {"criterion": "C3", "status": "INDETERMINATE"},
                    ],
                    "challenger_findings": "None",
                    "trace_summary": "test",
                    "escalated_at": "2024-06-20T10:00:00.000Z",
                },
            },
        )

        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        data = response.json()
        statuses = [c["status"] for c in data["cases"][0]["criteria_assessment"]]
        assert statuses == ["met", "not_met", "indeterminate"]
