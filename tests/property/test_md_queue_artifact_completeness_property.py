"""Property-based tests for Medical Director Queue Artifact Completeness.

**Validates: Requirements 15.7**

Property 35: Medical Director Queue Artifact Completeness
- For every escalated case in the MD queue API response, all 4 artifact sections
  are present and non-null: briefing_summary, criteria_assessment,
  challenger_findings, and trace_summary.
- No required artifact is missing from any escalated case.
"""

from __future__ import annotations

import asyncio

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
    create_frontend_router,
)
from src.clinical_reasoning_fabric.models.core import TraceCategory


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Non-empty identifier strings
identifier_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=30,
)

# Non-empty text for summaries and findings
summary_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.,;:()",
    min_size=5,
    max_size=200,
)

# Valid criterion statuses
criterion_status_strategy = st.sampled_from([
    "met", "not_met", "indeterminate", "not_evaluated",
])

# ISO-8601 timestamp strings
timestamp_strategy = st.datetimes(
    min_value=__import__("datetime").datetime(2020, 1, 1),
    max_value=__import__("datetime").datetime(2030, 12, 31),
).map(lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")

# Criterion names
criterion_name_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_",
    min_size=3,
    max_size=60,
)


@st.composite
def criterion_assessment_strategy(draw):
    """Generate a single criterion assessment entry."""
    return {
        "criterion": draw(criterion_name_strategy),
        "status": draw(criterion_status_strategy),
    }


@st.composite
def escalation_case_strategy(draw):
    """Generate a single escalation case with all 4 required artifacts."""
    num_criteria = draw(st.integers(min_value=1, max_value=5))
    criteria = [draw(criterion_assessment_strategy()) for _ in range(num_criteria)]

    return {
        "case_id": draw(identifier_strategy),
        "briefing_summary": draw(summary_strategy),
        "criteria_assessment": criteria,
        "challenger_findings": draw(summary_strategy),
        "trace_summary": draw(summary_strategy),
        "escalated_at": draw(timestamp_strategy),
    }


@st.composite
def escalation_cases_list_strategy(draw):
    """Generate a list of 1-5 escalation cases with unique case_ids."""
    num_cases = draw(st.integers(min_value=1, max_value=5))
    cases = []
    used_ids: set[str] = set()
    for _ in range(num_cases):
        case = draw(escalation_case_strategy())
        while case["case_id"] in used_ids:
            case["case_id"] = draw(identifier_strategy)
        used_ids.add(case["case_id"])
        cases.append(case)
    return cases


# =============================================================================
# Helpers
# =============================================================================


def _create_app_and_service():
    """Create a fresh FastAPI app and AuditTrailService for each test."""
    storage = InMemoryAppendOnlyStorage()
    audit_service = AuditTrailService(storage_backend=storage)
    app = FastAPI()
    router = create_frontend_router(audit_service)
    app.include_router(router)
    return app, audit_service


async def _store_escalations_and_query(cases: list[dict]) -> dict:
    """Store escalation entries via audit trail and query the MD queue endpoint.

    Each case is stored as a separate escalation trace entry.
    Returns the API response as a parsed dictionary.
    """
    app, audit_service = _create_app_and_service()

    # Store each case as an escalation trace entry
    for case in cases:
        await audit_service.record_entry(
            request_id=f"req-{case['case_id']}",
            identity_id="test-user",
            category=TraceCategory.DECISION_STEP,
            details={
                "human_gate": "escalated",
                "escalation": {
                    "case_id": case["case_id"],
                    "briefing_summary": case["briefing_summary"],
                    "criteria_assessment": case["criteria_assessment"],
                    "challenger_findings": case["challenger_findings"],
                    "trace_summary": case["trace_summary"],
                    "escalated_at": case["escalated_at"],
                },
            },
        )

    # Query the API endpoint
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/md-queue")
        assert response.status_code == 200
        return response.json()


# =============================================================================
# Property 35: Medical Director Queue Artifact Completeness
# =============================================================================


@pytest.mark.property
class TestMDQueueArtifactCompleteness:
    """Property 35: Medical Director Queue Artifact Completeness.

    **Validates: Requirements 15.7**

    For every escalated case in the MD queue API, all 4 artifact sections
    are present and non-null: briefing_summary, criteria_assessment,
    challenger_findings, and trace_summary.
    """

    @given(cases=escalation_cases_list_strategy())
    @settings(max_examples=100)
    def test_all_escalated_cases_present_in_response(self, cases: list[dict]):
        """All stored escalation cases appear in the API response.

        **Validates: Requirements 15.7**

        For N escalation cases stored, the API response must contain
        exactly N cases.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_escalations_and_query(cases)
        )

        expected_count = len(cases)
        actual_count = len(result["cases"])
        assert actual_count == expected_count, (
            f"Expected {expected_count} cases, got {actual_count}"
        )

    @given(cases=escalation_cases_list_strategy())
    @settings(max_examples=100)
    def test_briefing_summary_present_in_all_cases(self, cases: list[dict]):
        """Every case in the response has a non-empty briefing_summary.

        **Validates: Requirements 15.7**

        The briefing_summary artifact provides the Briefing Packet summary
        and must be present and non-empty for every escalated case.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_escalations_and_query(cases)
        )

        for i, case in enumerate(result["cases"]):
            assert "briefing_summary" in case, (
                f"cases[{i}] is missing 'briefing_summary' field"
            )
            assert case["briefing_summary"] is not None, (
                f"cases[{i}] has null briefing_summary"
            )
            assert len(case["briefing_summary"]) > 0, (
                f"cases[{i}] has empty briefing_summary"
            )

    @given(cases=escalation_cases_list_strategy())
    @settings(max_examples=100)
    def test_criteria_assessment_present_in_all_cases(self, cases: list[dict]):
        """Every case in the response has a non-empty criteria_assessment.

        **Validates: Requirements 15.7**

        The criteria_assessment artifact lists per-criterion status and
        must be present with at least one criterion for every escalated case.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_escalations_and_query(cases)
        )

        for i, case in enumerate(result["cases"]):
            assert "criteria_assessment" in case, (
                f"cases[{i}] is missing 'criteria_assessment' field"
            )
            assert case["criteria_assessment"] is not None, (
                f"cases[{i}] has null criteria_assessment"
            )
            assert len(case["criteria_assessment"]) >= 1, (
                f"cases[{i}] has empty criteria_assessment"
            )

    @given(cases=escalation_cases_list_strategy())
    @settings(max_examples=100)
    def test_challenger_findings_present_in_all_cases(self, cases: list[dict]):
        """Every case in the response has a non-empty challenger_findings.

        **Validates: Requirements 15.7**

        The challenger_findings artifact provides OPA verification results
        and must be present and non-empty for every escalated case.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_escalations_and_query(cases)
        )

        for i, case in enumerate(result["cases"]):
            assert "challenger_findings" in case, (
                f"cases[{i}] is missing 'challenger_findings' field"
            )
            assert case["challenger_findings"] is not None, (
                f"cases[{i}] has null challenger_findings"
            )
            assert len(case["challenger_findings"]) > 0, (
                f"cases[{i}] has empty challenger_findings"
            )

    @given(cases=escalation_cases_list_strategy())
    @settings(max_examples=100)
    def test_trace_summary_present_in_all_cases(self, cases: list[dict]):
        """Every case in the response has a non-empty trace_summary.

        **Validates: Requirements 15.7**

        The trace_summary artifact provides the execution trace summary
        and must be present and non-empty for every escalated case.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_escalations_and_query(cases)
        )

        for i, case in enumerate(result["cases"]):
            assert "trace_summary" in case, (
                f"cases[{i}] is missing 'trace_summary' field"
            )
            assert case["trace_summary"] is not None, (
                f"cases[{i}] has null trace_summary"
            )
            assert len(case["trace_summary"]) > 0, (
                f"cases[{i}] has empty trace_summary"
            )

    @given(cases=escalation_cases_list_strategy())
    @settings(max_examples=100)
    def test_all_four_artifacts_non_null_in_every_case(self, cases: list[dict]):
        """Every case has all 4 artifacts present and non-null simultaneously.

        **Validates: Requirements 15.7**

        This composite check validates that no single case is missing any
        of the 4 required artifacts: briefing_summary, criteria_assessment,
        challenger_findings, trace_summary.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_escalations_and_query(cases)
        )

        required_artifacts = [
            "briefing_summary",
            "criteria_assessment",
            "challenger_findings",
            "trace_summary",
        ]

        for i, case in enumerate(result["cases"]):
            for artifact in required_artifacts:
                assert artifact in case, (
                    f"cases[{i}] is missing required artifact '{artifact}'"
                )
                assert case[artifact] is not None, (
                    f"cases[{i}] has null '{artifact}'"
                )

    @given(cases=escalation_cases_list_strategy())
    @settings(max_examples=100)
    def test_criteria_assessment_has_per_criterion_status(self, cases: list[dict]):
        """Each criterion in criteria_assessment has both criterion name and status.

        **Validates: Requirements 15.7**

        Per-criterion status must include the criterion name and a valid
        status value (met, not_met, indeterminate, not_evaluated).
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_escalations_and_query(cases)
        )

        valid_statuses = {"met", "not_met", "indeterminate", "not_evaluated"}
        for i, case in enumerate(result["cases"]):
            for j, criterion in enumerate(case["criteria_assessment"]):
                assert "criterion" in criterion, (
                    f"cases[{i}].criteria_assessment[{j}] missing 'criterion'"
                )
                assert "status" in criterion, (
                    f"cases[{i}].criteria_assessment[{j}] missing 'status'"
                )
                assert criterion["status"] in valid_statuses, (
                    f"cases[{i}].criteria_assessment[{j}] has invalid status "
                    f"'{criterion['status']}'"
                )
