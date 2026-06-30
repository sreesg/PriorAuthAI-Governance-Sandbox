"""Unit tests for HumanGateService (BEACON Layer 7).

Tests decision routing logic, no-automated-denial invariant,
retry behavior on queue failure, and escalation artifact completeness.

Validates:
    - All criteria MET + verification PASS → Auto-Approve
    - Any NOT_MET or INDETERMINATE → Escalate to MD
    - Verification FAIL → Escalate to MD with findings
    - NEVER produces automated denial regardless of input
    - Escalation includes all four artifacts
    - Queue unavailable: retry at 30s intervals up to 10 attempts

Requirements referenced: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_reasoning_fabric.beacon.human_gate_service import (
    AuditTrailServiceProtocol,
    DecisionInput,
    EscalationPackage,
    HumanGateService,
    MAX_RETRY_ATTEMPTS,
    MedicalDirectorQueue,
    RETRY_INTERVAL_SECONDS,
    RoutingResult,
)
from clinical_reasoning_fabric.models.core import (
    BriefingPacket,
    CriterionAssessment,
    CriterionStatus,
    Disposition,
    MemberActiveState,
    TraceCategory,
    TraceEntry,
    VerificationResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def md_queue():
    """Mock MD queue that successfully enqueues by default."""
    queue = AsyncMock()
    queue.enqueue = AsyncMock(return_value=True)
    return queue


@pytest.fixture
def audit_service():
    """Mock audit trail service."""
    audit = AsyncMock()
    audit.record_entry = AsyncMock(return_value=None)
    return audit


@pytest.fixture
def human_gate(md_queue, audit_service):
    """HumanGateService with mock dependencies."""
    return HumanGateService(md_queue=md_queue, audit=audit_service)


@pytest.fixture
def member_state():
    """A minimal MemberActiveState for testing."""
    return MemberActiveState(
        member_id="member-001",
        active_diagnoses=[{"code": "J45.0", "description": "Asthma"}],
        active_prescriptions=[{"name": "dupilumab", "status": "active"}],
        sdoh_factors=[],
        governing_policies=[],
    )


@pytest.fixture
def briefing_packet(member_state):
    """A valid BriefingPacket for testing."""
    return BriefingPacket(
        request_id="req-001",
        member_id="member-001",
        cpt_code="99213",
        active_clinical_state=member_state,
        verified_evidence_snippets=[],
        no_evidence_found=False,
    )


@pytest.fixture
def execution_trace():
    """A minimal execution trace for testing."""
    return [
        TraceEntry(
            sequence_number=1,
            timestamp="2024-01-15T10:30:45.123Z",
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.DECISION_STEP,
            details={"event": "criteria_evaluation_complete"},
        )
    ]


@pytest.fixture
def all_met_assessments():
    """Criteria assessments where ALL criteria are MET."""
    return [
        CriterionAssessment(
            criterion_id="c1",
            criterion_name="Medical necessity",
            status=CriterionStatus.MET,
            evidence_references=["ev-001"],
            reasoning="Documented medical necessity",
        ),
        CriterionAssessment(
            criterion_id="c2",
            criterion_name="Prior treatments",
            status=CriterionStatus.MET,
            evidence_references=["ev-002"],
            reasoning="Conservative treatments attempted",
        ),
    ]


@pytest.fixture
def mixed_assessments():
    """Criteria assessments with one NOT_MET criterion."""
    return [
        CriterionAssessment(
            criterion_id="c1",
            criterion_name="Medical necessity",
            status=CriterionStatus.MET,
            evidence_references=["ev-001"],
            reasoning="Documented",
        ),
        CriterionAssessment(
            criterion_id="c2",
            criterion_name="Prior treatments",
            status=CriterionStatus.NOT_MET,
            evidence_references=[],
            reasoning="No documentation of prior treatments",
        ),
    ]


@pytest.fixture
def indeterminate_assessments():
    """Criteria assessments with one INDETERMINATE criterion."""
    return [
        CriterionAssessment(
            criterion_id="c1",
            criterion_name="Medical necessity",
            status=CriterionStatus.INDETERMINATE,
            evidence_references=["ev-001"],
            reasoning="Insufficient evidence to determine",
        ),
    ]


# =============================================================================
# Test: Auto-Approve scenario (Requirement 9.1)
# =============================================================================


class TestAutoApproveScenario:
    """All criteria MET + verification PASS → Auto-Approve."""

    async def test_all_met_and_pass_returns_approved(
        self, human_gate, all_met_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.1: Auto-approve when all criteria MET and verification PASS."""
        decision_input = DecisionInput(
            criteria_assessments=all_met_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
            challenger_findings=None,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.disposition == Disposition.APPROVED
        assert result.escalation_package is None
        assert result.delivery_attempts == 0
        assert result.pending is False

    async def test_auto_approve_does_not_call_md_queue(
        self, human_gate, md_queue, all_met_assessments, briefing_packet, execution_trace
    ):
        """Auto-approve should not touch the MD queue."""
        decision_input = DecisionInput(
            criteria_assessments=all_met_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        await human_gate.route_decision(decision_input)

        md_queue.enqueue.assert_not_called()


# =============================================================================
# Test: Escalation on NOT_MET criteria (Requirement 9.3)
# =============================================================================


class TestEscalationOnNotMet:
    """Any criterion NOT_MET → Escalate to MD."""

    async def test_not_met_criterion_escalates(
        self, human_gate, mixed_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.3: NOT_MET criterion triggers escalation."""
        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.escalation_package is not None

    async def test_indeterminate_criterion_escalates(
        self, human_gate, indeterminate_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.3: INDETERMINATE criterion triggers escalation."""
        decision_input = DecisionInput(
            criteria_assessments=indeterminate_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.escalation_package is not None


# =============================================================================
# Test: Escalation on Verification FAIL (Requirement 9.4)
# =============================================================================


class TestEscalationOnVerificationFail:
    """Verification FAIL → Route to MD with challenge findings."""

    async def test_verification_fail_escalates_even_if_all_met(
        self, human_gate, all_met_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.4: Verification FAIL escalates even with all criteria MET."""
        decision_input = DecisionInput(
            criteria_assessments=all_met_assessments,
            verification_result=VerificationResult.FAIL,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.escalation_package is not None

    async def test_verification_fail_includes_reason(
        self, human_gate, all_met_assessments, briefing_packet, execution_trace
    ):
        """Escalation reason should mention OPA verification failure."""
        decision_input = DecisionInput(
            criteria_assessments=all_met_assessments,
            verification_result=VerificationResult.FAIL,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert "verification FAILED" in result.escalation_package.escalation_reason


# =============================================================================
# Test: No Automated Denial Invariant (Requirement 9.2)
# =============================================================================


class TestNoAutomatedDenial:
    """The system NEVER produces an automated denial."""

    async def test_all_not_met_and_fail_does_not_deny(
        self, human_gate, briefing_packet, execution_trace
    ):
        """Even worst case (all NOT_MET + FAIL) → ESCALATED, never DENIED."""
        all_not_met = [
            CriterionAssessment(
                criterion_id=f"c{i}",
                criterion_name=f"Criterion {i}",
                status=CriterionStatus.NOT_MET,
                evidence_references=[],
                reasoning="Not met",
            )
            for i in range(5)
        ]
        decision_input = DecisionInput(
            criteria_assessments=all_not_met,
            verification_result=VerificationResult.FAIL,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.disposition != "DENIED"
        assert result.disposition in (Disposition.APPROVED, Disposition.ESCALATED)

    async def test_empty_criteria_does_not_deny(
        self, human_gate, briefing_packet, execution_trace
    ):
        """Empty criteria list → ESCALATED (not approved, not denied)."""
        decision_input = DecisionInput(
            criteria_assessments=[],
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED


    async def test_disposition_enum_has_no_denied_value(self):
        """Verify the Disposition enum itself does not have a DENIED value."""
        disposition_values = [d.value for d in Disposition]
        assert "DENIED" not in disposition_values
        assert set(disposition_values) == {"APPROVED", "ESCALATED"}


# =============================================================================
# Test: Escalation Artifact Completeness (Requirement 9.5)
# =============================================================================


class TestEscalationArtifactCompleteness:
    """Escalation includes all four artifacts."""

    async def test_escalation_includes_briefing_packet(
        self, human_gate, mixed_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.5: Briefing_Packet artifact present."""
        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.escalation_package.briefing_packet is not None
        assert result.escalation_package.briefing_packet.request_id == "req-001"

    async def test_escalation_includes_criteria_assessments(
        self, human_gate, mixed_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.5: Criteria assessment artifact present."""
        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.escalation_package.criteria_assessments is not None
        assert len(result.escalation_package.criteria_assessments) == 2


    async def test_escalation_includes_execution_trace(
        self, human_gate, mixed_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.5: Execution trace artifact present."""
        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.escalation_package.execution_trace is not None
        assert len(result.escalation_package.execution_trace) >= 1

    async def test_escalation_includes_challenger_findings(
        self, human_gate, mixed_assessments, briefing_packet, execution_trace
    ):
        """Requirement 9.5: OPA findings artifact present when provided."""
        from clinical_reasoning_fabric.beacon.opa_challenger_service import (
            ChallengerResult,
            PolicyViolation,
        )

        findings = ChallengerResult(
            verification_result=VerificationResult.FAIL,
            violated_rules=[
                PolicyViolation(rule_id="R001", description="Rule violation")
            ],
            escalation_reason="Policy violation detected",
        )

        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.FAIL,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
            challenger_findings=findings,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.escalation_package.challenger_findings is not None
        assert len(result.escalation_package.challenger_findings.violated_rules) == 1


# =============================================================================
# Test: MD Queue Retry Logic (Requirement 9.6)
# =============================================================================


class TestMDQueueRetryLogic:
    """Queue unavailable: retry at 30s intervals up to 10 attempts."""

    async def test_successful_delivery_on_first_attempt(
        self, human_gate, md_queue, mixed_assessments, briefing_packet, execution_trace
    ):
        """Successful first delivery returns immediately."""
        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.delivery_attempts == 1
        assert result.pending is False

    @patch("clinical_reasoning_fabric.beacon.human_gate_service.asyncio.sleep")
    async def test_retries_on_queue_failure(
        self, mock_sleep, md_queue, audit_service, mixed_assessments,
        briefing_packet, execution_trace
    ):
        """Retry on queue failure, succeed on 3rd attempt."""
        md_queue.enqueue = AsyncMock(
            side_effect=[False, False, True]
        )
        gate = HumanGateService(md_queue=md_queue, audit=audit_service)

        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.delivery_attempts == 3
        assert result.pending is False
        # Should have logged 2 failures
        assert audit_service.record_entry.call_count == 2


    @patch("clinical_reasoning_fabric.beacon.human_gate_service.asyncio.sleep")
    async def test_max_retries_exhausted_stays_pending(
        self, mock_sleep, md_queue, audit_service, mixed_assessments,
        briefing_packet, execution_trace
    ):
        """All 10 attempts fail → pending state."""
        md_queue.enqueue = AsyncMock(return_value=False)
        gate = HumanGateService(md_queue=md_queue, audit=audit_service)

        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.delivery_attempts == MAX_RETRY_ATTEMPTS
        assert result.pending is True
        # All 10 failures logged
        assert audit_service.record_entry.call_count == MAX_RETRY_ATTEMPTS

    @patch("clinical_reasoning_fabric.beacon.human_gate_service.asyncio.sleep")
    async def test_retry_interval_is_30_seconds(
        self, mock_sleep, md_queue, audit_service, mixed_assessments,
        briefing_packet, execution_trace
    ):
        """Verify sleep is called with 30-second intervals between retries."""
        md_queue.enqueue = AsyncMock(
            side_effect=[False, False, True]
        )
        gate = HumanGateService(md_queue=md_queue, audit=audit_service)

        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        await gate.route_decision(decision_input)

        # Sleep called twice (between attempt 1-2 and 2-3, not after success)
        assert mock_sleep.call_count == 2
        for call in mock_sleep.call_args_list:
            assert call.args[0] == RETRY_INTERVAL_SECONDS


    @patch("clinical_reasoning_fabric.beacon.human_gate_service.asyncio.sleep")
    async def test_queue_exception_triggers_retry(
        self, mock_sleep, md_queue, audit_service, mixed_assessments,
        briefing_packet, execution_trace
    ):
        """Queue raising exception is treated as failure and retried."""
        md_queue.enqueue = AsyncMock(
            side_effect=[Exception("Connection refused"), True]
        )
        gate = HumanGateService(md_queue=md_queue, audit=audit_service)

        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await gate.route_decision(decision_input)

        assert result.disposition == Disposition.ESCALATED
        assert result.delivery_attempts == 2
        assert result.pending is False

    @patch("clinical_reasoning_fabric.beacon.human_gate_service.asyncio.sleep")
    async def test_never_denies_even_when_queue_exhausted(
        self, mock_sleep, md_queue, audit_service, mixed_assessments,
        briefing_packet, execution_trace
    ):
        """Even when all retries fail, disposition is ESCALATED not DENIED."""
        md_queue.enqueue = AsyncMock(return_value=False)
        gate = HumanGateService(md_queue=md_queue, audit=audit_service)

        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.FAIL,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await gate.route_decision(decision_input)

        # Critical safety check: NEVER denied
        assert result.disposition == Disposition.ESCALATED
        assert result.disposition != Disposition.APPROVED


# =============================================================================
# Test: Escalation Reason Construction
# =============================================================================


class TestEscalationReasonConstruction:
    """Verify escalation reasons contain useful diagnostic info."""

    async def test_reason_includes_not_met_criteria_names(
        self, human_gate, mixed_assessments, briefing_packet, execution_trace
    ):
        """Escalation reason lists names of NOT_MET criteria."""
        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert "Prior treatments" in result.escalation_package.escalation_reason
        assert "NOT_MET" in result.escalation_package.escalation_reason

    async def test_reason_includes_indeterminate_criteria_names(
        self, human_gate, indeterminate_assessments, briefing_packet, execution_trace
    ):
        """Escalation reason lists names of INDETERMINATE criteria."""
        decision_input = DecisionInput(
            criteria_assessments=indeterminate_assessments,
            verification_result=VerificationResult.PASS,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        assert "Medical necessity" in result.escalation_package.escalation_reason
        assert "INDETERMINATE" in result.escalation_package.escalation_reason

    async def test_combined_not_met_and_fail_reason(
        self, human_gate, mixed_assessments, briefing_packet, execution_trace
    ):
        """Both criteria failure and verification failure in reason."""
        decision_input = DecisionInput(
            criteria_assessments=mixed_assessments,
            verification_result=VerificationResult.FAIL,
            briefing_packet=briefing_packet,
            execution_trace=execution_trace,
        )

        result = await human_gate.route_decision(decision_input)

        reason = result.escalation_package.escalation_reason
        assert "NOT_MET" in reason
        assert "verification FAILED" in reason
