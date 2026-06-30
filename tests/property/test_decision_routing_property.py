"""Property-based tests for Decision Routing and Escalation.

**Validates: Requirements 9.1, 9.2, 9.3, 9.5**

Property 14: Decision Routing Correctness
- All MET + PASS → approved
- Any NOT_MET or INDETERMINATE → escalated
- Verification FAIL → escalated
- NEVER denied

Property 15: Escalation Artifact Completeness
- Every escalation includes all four non-null artifacts:
  Briefing_Packet, criteria assessment, OPA findings, execution trace
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.human_gate_service import (
    DecisionInput,
    EscalationPackage,
    HumanGateService,
    MedicalDirectorQueue,
    RoutingResult,
)
from clinical_reasoning_fabric.beacon.opa_challenger_service import (
    ChallengerResult,
    PolicyViolation,
)
from clinical_reasoning_fabric.models.core import (
    BriefingPacket,
    CriterionAssessment,
    CriterionStatus,
    Disposition,
    KMSSignature,
    MemberActiveState,
    ScoredChunk,
    TraceCategory,
    TraceEntry,
    VerificationResult,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for CriterionStatus
criterion_status_strategy = st.sampled_from(list(CriterionStatus))

# Strategy for VerificationResult
verification_result_strategy = st.sampled_from(list(VerificationResult))


@st.composite
def criterion_assessment_strategy(draw):
    """Generate a single CriterionAssessment with random status."""
    status = draw(criterion_status_strategy)
    criterion_id = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=3,
        max_size=10,
    ))
    criterion_name = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz ",
        min_size=5,
        max_size=30,
    ))
    return CriterionAssessment(
        criterion_id=f"crit-{criterion_id}",
        criterion_name=criterion_name.strip() or "criterion",
        status=status,
    )


@st.composite
def all_met_assessments_strategy(draw):
    """Generate 1-10 criteria assessments where ALL are MET."""
    num = draw(st.integers(min_value=1, max_value=10))
    assessments = []
    for i in range(num):
        assessments.append(CriterionAssessment(
            criterion_id=f"crit-{i}",
            criterion_name=f"criterion {i}",
            status=CriterionStatus.MET,
        ))
    return assessments


@st.composite
def at_least_one_not_met_strategy(draw):
    """Generate assessments where at least one is NOT_MET or INDETERMINATE."""
    num = draw(st.integers(min_value=1, max_value=10))
    assessments = []
    for i in range(num):
        status = draw(criterion_status_strategy)
        assessments.append(CriterionAssessment(
            criterion_id=f"crit-{i}",
            criterion_name=f"criterion {i}",
            status=status,
        ))
    # Ensure at least one is NOT_MET or INDETERMINATE
    has_non_met = any(
        a.status in (CriterionStatus.NOT_MET, CriterionStatus.INDETERMINATE)
        for a in assessments
    )
    if not has_non_met:
        # Force one to be NOT_MET or INDETERMINATE
        bad_status = draw(st.sampled_from([CriterionStatus.NOT_MET, CriterionStatus.INDETERMINATE]))
        idx = draw(st.integers(min_value=0, max_value=len(assessments) - 1))
        assessments[idx] = CriterionAssessment(
            criterion_id=assessments[idx].criterion_id,
            criterion_name=assessments[idx].criterion_name,
            status=bad_status,
        )
    return assessments


@st.composite
def mixed_assessments_strategy(draw):
    """Generate any combination of criteria assessments (1-10)."""
    num = draw(st.integers(min_value=1, max_value=10))
    assessments = []
    for i in range(num):
        status = draw(criterion_status_strategy)
        assessments.append(CriterionAssessment(
            criterion_id=f"crit-{i}",
            criterion_name=f"criterion {i}",
            status=status,
        ))
    return assessments


def _make_briefing_packet() -> BriefingPacket:
    """Create a minimal valid BriefingPacket for testing."""
    return BriefingPacket(
        request_id="req-test-001",
        member_id="member-001",
        cpt_code="99213",
        active_clinical_state=MemberActiveState(
            member_id="member-001",
            active_diagnoses=[],
            active_prescriptions=[],
            sdoh_factors=[],
            governing_policies=[],
        ),
        verified_evidence_snippets=[],
        no_evidence_found=True,
    )


def _make_execution_trace() -> list[TraceEntry]:
    """Create a minimal valid execution trace for testing."""
    return [
        TraceEntry(
            sequence_number=0,
            timestamp="2024-01-15T10:30:45.123Z",
            request_id="req-test-001",
            identity_id="agent-001",
            category=TraceCategory.DECISION_STEP,
        ),
        TraceEntry(
            sequence_number=1,
            timestamp="2024-01-15T10:30:46.456Z",
            request_id="req-test-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        ),
    ]


def _make_challenger_findings(
    verification_result: VerificationResult = VerificationResult.PASS,
) -> ChallengerResult:
    """Create a ChallengerResult for testing."""
    return ChallengerResult(
        verification_result=verification_result,
        signature_result=None,
        policy_result=None,
        tamper_alerts=[],
        violated_rules=[],
        escalation_reason=None if verification_result == VerificationResult.PASS else "Verification failed",
    )


def _make_service() -> HumanGateService:
    """Create a HumanGateService with a mock MD queue that always succeeds."""
    mock_queue = AsyncMock(spec=MedicalDirectorQueue)
    mock_queue.enqueue = AsyncMock(return_value=True)

    mock_audit = AsyncMock()
    mock_audit.record_entry = AsyncMock(return_value=None)

    return HumanGateService(md_queue=mock_queue, audit=mock_audit)


# =============================================================================
# Property 14: Decision Routing Correctness
# =============================================================================


@pytest.mark.property
class TestDecisionRoutingCorrectness:
    """Property 14: Decision Routing Correctness.

    **Validates: Requirements 9.1, 9.2, 9.3, 9.5**

    For any combination of criteria statuses and verification results:
    - All MET + PASS → APPROVED
    - Any NOT_MET or INDETERMINATE (regardless of verification) → ESCALATED
    - Any FAIL (regardless of criteria) → ESCALATED
    - Disposition is NEVER DENIED under any combination
    """

    @given(assessments=all_met_assessments_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_all_met_and_pass_yields_approved(self, assessments):
        """All criteria MET + verification PASS → Auto-APPROVED.

        **Validates: Requirements 9.1**

        When every criterion evaluates to MET and the OPA Challenger
        verification passes, the system auto-approves without MD review.
        """
        service = _make_service()
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=_make_briefing_packet(),
            execution_trace=_make_execution_trace(),
            challenger_findings=_make_challenger_findings(VerificationResult.PASS),
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.APPROVED, (
            f"Expected APPROVED when all criteria MET and verification PASS, "
            f"got {result.disposition}"
        )
        assert result.escalation_package is None, (
            "Approved decisions should not have an escalation package"
        )

    @given(assessments=at_least_one_not_met_strategy(),
           verification=verification_result_strategy)
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_any_not_met_or_indeterminate_yields_escalated(
        self, assessments, verification
    ):
        """Any NOT_MET or INDETERMINATE criterion → ESCALATED.

        **Validates: Requirements 9.3**

        Regardless of verification result, if any criterion is NOT_MET
        or INDETERMINATE, the case is escalated to Medical Director.
        """
        service = _make_service()
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=verification,
            briefing_packet=_make_briefing_packet(),
            execution_trace=_make_execution_trace(),
            challenger_findings=_make_challenger_findings(verification),
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED, (
            f"Expected ESCALATED when criteria include NOT_MET/INDETERMINATE, "
            f"got {result.disposition}. "
            f"Criteria statuses: {[a.status.value for a in assessments]}"
        )

    @given(assessments=mixed_assessments_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_verification_fail_yields_escalated(self, assessments):
        """Verification FAIL → ESCALATED regardless of criteria.

        **Validates: Requirements 9.3**

        When the OPA Challenger verification fails, the case is always
        escalated to Medical Director, regardless of criteria statuses.
        """
        service = _make_service()
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=VerificationResult.FAIL,
            briefing_packet=_make_briefing_packet(),
            execution_trace=_make_execution_trace(),
            challenger_findings=_make_challenger_findings(VerificationResult.FAIL),
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED, (
            f"Expected ESCALATED when verification is FAIL, "
            f"got {result.disposition}. "
            f"Criteria statuses: {[a.status.value for a in assessments]}"
        )

    @given(
        assessments=mixed_assessments_strategy(),
        verification=verification_result_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_disposition_never_denied(self, assessments, verification):
        """Disposition is NEVER DENIED under any combination.

        **Validates: Requirements 9.2**

        The system must never produce an automated denial. The only
        valid dispositions are APPROVED and ESCALATED.
        """
        service = _make_service()
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=verification,
            briefing_packet=_make_briefing_packet(),
            execution_trace=_make_execution_trace(),
            challenger_findings=_make_challenger_findings(verification),
        )

        result = await service.route_decision(decision_input)

        # Verify disposition is never DENIED (only APPROVED or ESCALATED)
        assert result.disposition in (Disposition.APPROVED, Disposition.ESCALATED), (
            f"Disposition must be APPROVED or ESCALATED, got {result.disposition}"
        )
        # Explicit check — Disposition enum should not even have DENIED,
        # but verify the value is not a denial string either
        assert result.disposition != "DENIED", (
            "System must NEVER produce automated denial"
        )
        assert "DENIED" not in str(result.disposition).upper() or result.disposition in (
            Disposition.APPROVED, Disposition.ESCALATED
        ), "No denial disposition allowed"


# =============================================================================
# Property 15: Escalation Artifact Completeness
# =============================================================================


@pytest.mark.property
class TestEscalationArtifactCompleteness:
    """Property 15: Escalation Artifact Completeness.

    **Validates: Requirements 9.5**

    Every escalation includes all four non-null artifacts:
    - briefing_packet is not None
    - criteria_assessments is not None and non-empty
    - execution_trace is not None and non-empty
    - challenger_findings may be None (only present when provided)
    """

    @given(assessments=at_least_one_not_met_strategy(),
           verification=verification_result_strategy)
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_escalation_has_briefing_packet(self, assessments, verification):
        """Escalation includes non-null Briefing_Packet.

        **Validates: Requirements 9.5**

        When a case is escalated, the escalation package must contain
        the full Briefing_Packet.
        """
        service = _make_service()
        briefing_packet = _make_briefing_packet()
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=verification,
            briefing_packet=briefing_packet,
            execution_trace=_make_execution_trace(),
            challenger_findings=_make_challenger_findings(verification),
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.escalation_package is not None, (
            "Escalated disposition must have an escalation package"
        )
        assert result.escalation_package.briefing_packet is not None, (
            "Escalation package must include a non-null Briefing_Packet"
        )
        assert result.escalation_package.briefing_packet == briefing_packet, (
            "Escalation package Briefing_Packet must match the input"
        )

    @given(assessments=at_least_one_not_met_strategy(),
           verification=verification_result_strategy)
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_escalation_has_criteria_assessments(self, assessments, verification):
        """Escalation includes non-null and non-empty criteria assessments.

        **Validates: Requirements 9.5**

        When a case is escalated, the escalation package must contain
        the per-criterion MET/NOT_MET/INDETERMINATE status assessments.
        """
        service = _make_service()
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=verification,
            briefing_packet=_make_briefing_packet(),
            execution_trace=_make_execution_trace(),
            challenger_findings=_make_challenger_findings(verification),
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.escalation_package is not None
        assert result.escalation_package.criteria_assessments is not None, (
            "Escalation package must include non-null criteria assessments"
        )
        assert len(result.escalation_package.criteria_assessments) > 0, (
            "Escalation package criteria assessments must be non-empty"
        )

    @given(assessments=at_least_one_not_met_strategy(),
           verification=verification_result_strategy)
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_escalation_has_execution_trace(self, assessments, verification):
        """Escalation includes non-null and non-empty execution trace.

        **Validates: Requirements 9.5**

        When a case is escalated, the escalation package must contain
        the complete execution trace.
        """
        service = _make_service()
        execution_trace = _make_execution_trace()
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=verification,
            briefing_packet=_make_briefing_packet(),
            execution_trace=execution_trace,
            challenger_findings=_make_challenger_findings(verification),
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.escalation_package is not None
        assert result.escalation_package.execution_trace is not None, (
            "Escalation package must include non-null execution trace"
        )
        assert len(result.escalation_package.execution_trace) > 0, (
            "Escalation package execution trace must be non-empty"
        )

    @given(assessments=at_least_one_not_met_strategy(),
           verification=verification_result_strategy)
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_escalation_has_challenger_findings_when_provided(
        self, assessments, verification
    ):
        """Escalation includes challenger findings when provided.

        **Validates: Requirements 9.5**

        When challenger findings are provided in the input, the
        escalation package must include them.
        """
        service = _make_service()
        challenger_findings = _make_challenger_findings(verification)
        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=verification,
            briefing_packet=_make_briefing_packet(),
            execution_trace=_make_execution_trace(),
            challenger_findings=challenger_findings,
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.escalation_package is not None
        # challenger_findings should be passed through when provided
        assert result.escalation_package.challenger_findings is not None, (
            "Escalation package must include challenger findings when provided"
        )
        assert result.escalation_package.challenger_findings == challenger_findings, (
            "Escalation package challenger findings must match the input"
        )

    @given(
        assessments=mixed_assessments_strategy(),
        verification=verification_result_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_verification_fail_escalation_has_all_artifacts(
        self, assessments, verification
    ):
        """For verification FAIL, escalation includes all required artifacts.

        **Validates: Requirements 9.5**

        When verification fails, the escalation package must include
        all four artifact types: briefing packet, criteria assessments,
        challenger findings, and execution trace.
        """
        # Force verification to FAIL to guarantee escalation
        service = _make_service()
        briefing_packet = _make_briefing_packet()
        execution_trace = _make_execution_trace()
        challenger_findings = _make_challenger_findings(VerificationResult.FAIL)

        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=VerificationResult.FAIL,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
            challenger_findings=challenger_findings,
        )

        result = await service.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED, (
            "Verification FAIL must always produce ESCALATED"
        )
        assert result.escalation_package is not None

        pkg = result.escalation_package

        # Verify all four artifacts are present
        assert pkg.briefing_packet is not None, (
            "Escalation must include Briefing_Packet"
        )
        assert pkg.criteria_assessments is not None and len(pkg.criteria_assessments) > 0, (
            "Escalation must include non-empty criteria assessments"
        )
        assert pkg.challenger_findings is not None, (
            "Escalation must include OPA challenger findings"
        )
        assert pkg.execution_trace is not None and len(pkg.execution_trace) > 0, (
            "Escalation must include non-empty execution trace"
        )

    @given(assessments=at_least_one_not_met_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_all_escalation_artifacts_complete_combined(self, assessments):
        """All escalation artifact invariants hold simultaneously.

        **Validates: Requirements 9.5**

        Combined property: for any escalated decision, ALL of the following
        hold at once:
        - briefing_packet is not None
        - criteria_assessments is not None and non-empty
        - execution_trace is not None and non-empty
        - challenger_findings is present when provided in input
        """
        service = _make_service()
        briefing_packet = _make_briefing_packet()
        execution_trace = _make_execution_trace()
        challenger_findings = _make_challenger_findings(VerificationResult.PASS)

        decision_input = DecisionInput(
            criteria_assessments=assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
            challenger_findings=challenger_findings,
        )

        result = await service.route_decision(decision_input)

        # This should be escalated because at_least_one_not_met_strategy
        # guarantees NOT_MET or INDETERMINATE
        assert result.disposition == Disposition.ESCALATED

        pkg = result.escalation_package
        assert pkg is not None

        # All four artifact checks
        assert pkg.briefing_packet is not None, "Missing Briefing_Packet"
        assert pkg.criteria_assessments is not None, "Missing criteria assessments"
        assert len(pkg.criteria_assessments) > 0, "criteria assessments is empty"
        assert pkg.execution_trace is not None, "Missing execution trace"
        assert len(pkg.execution_trace) > 0, "execution trace is empty"
        assert pkg.challenger_findings is not None, (
            "Missing challenger findings (was provided in input)"
        )
