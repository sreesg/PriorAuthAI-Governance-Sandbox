"""BEACON Layer 7 — Human Gates Service.

Enforces the no-automated-denial policy and routes PA decisions to either
auto-approval or Medical Director escalation. The system NEVER produces
an automated denial.

Requirements referenced: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

from clinical_reasoning_fabric.beacon.opa_challenger_service import (
    ChallengerResult,
    PolicyViolation,
)
from clinical_reasoning_fabric.models.core import (
    BriefingPacket,
    CriterionAssessment,
    CriterionStatus,
    Disposition,
    TraceCategory,
    TraceEntry,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class MedicalDirectorQueue(Protocol):
    """Protocol for the Medical Director review queue.

    Implementations must provide an enqueue method that delivers
    escalation packages to the MD review queue.
    """

    async def enqueue(self, escalation_package: "EscalationPackage") -> bool:
        """Enqueue an escalation package for MD review.

        Returns True if successfully enqueued, False if queue is unavailable.
        """
        ...


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DecisionInput:
    """Input data for the human gate routing decision.

    Contains all the information needed to determine whether to
    auto-approve or escalate to a Medical Director.
    """

    criteria_assessments: list[CriterionAssessment]
    """Per-criterion MET/NOT_MET/INDETERMINATE assessments."""

    verification_result: VerificationResult
    """PASS or FAIL from OPA Challenger verification."""

    briefing_packet: BriefingPacket
    """The assembled Briefing Packet with clinical context."""

    execution_trace: list[TraceEntry]
    """Complete execution trace for the PA request."""

    challenger_findings: Optional[ChallengerResult] = None
    """Detailed findings from the OPA Challenger Agent."""


@dataclass
class EscalationPackage:
    """Package of artifacts delivered to the Medical Director queue.

    Requirement 9.5: Includes all four artifacts — Briefing_Packet,
    criteria assessment, OPA findings, and complete execution trace.
    """

    briefing_packet: BriefingPacket
    """The full Briefing Packet with clinical context."""

    criteria_assessments: list[CriterionAssessment]
    """Per-criterion MET/NOT_MET/INDETERMINATE status."""

    challenger_findings: Optional[ChallengerResult]
    """OPA Challenger Agent findings (tamper alerts, violations)."""

    execution_trace: list[TraceEntry]
    """Complete execution trace from request initiation to decision."""

    escalation_reason: str = ""
    """Human-readable reason for escalation."""

    escalated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class RoutingResult:
    """Result of the human gate routing decision."""

    disposition: Disposition
    """APPROVED or ESCALATED — never DENIED."""

    escalation_package: Optional[EscalationPackage] = None
    """Present only when disposition is ESCALATED."""

    delivery_attempts: int = 0
    """Number of MD queue delivery attempts made."""

    pending: bool = False
    """True if still trying to deliver to MD queue."""


# =============================================================================
# Human Gate Service
# =============================================================================

# Constants for retry behavior
MAX_RETRY_ATTEMPTS = 10
RETRY_INTERVAL_SECONDS = 30


class HumanGateService:
    """Enforces no-automated-denial policy and Medical Director routing.

    Requirement 9.1: All criteria MET + verification PASS → Auto-Approve.
    Requirement 9.2: NEVER produces automated denial.
    Requirement 9.3: Any NOT_MET/INDETERMINATE → Escalate to MD.
    Requirement 9.4: Verification FAIL → Route to MD with findings.
    Requirement 9.5: Escalation includes all 4 artifacts.
    Requirement 9.6: Queue unavailable → hold pending, retry at 30s intervals.
    """

    def __init__(
        self,
        md_queue: MedicalDirectorQueue,
        audit: "AuditTrailServiceProtocol",
    ) -> None:
        """Initialize HumanGateService.

        Args:
            md_queue: Medical Director review queue implementation.
            audit: Audit trail service for logging retry failures.
        """
        self._md_queue = md_queue
        self._audit = audit

    @property
    def md_queue(self) -> MedicalDirectorQueue:
        """Access the Medical Director queue."""
        return self._md_queue

    @property
    def audit(self) -> "AuditTrailServiceProtocol":
        """Access the audit trail service."""
        return self._audit

    async def route_decision(
        self,
        decision: DecisionInput,
    ) -> RoutingResult:
        """Route a PA decision based on criteria evaluation and verification.

        Decision logic:
            - All criteria MET AND verification PASS → APPROVED (auto-approve)
            - Any criterion NOT_MET or INDETERMINATE → ESCALATED to MD
            - Verification FAIL → ESCALATED to MD with challenge findings
            - NEVER produces a denial regardless of any input combination

        On MD queue unavailable: holds in pending state, retries delivery
        at 30-second intervals up to 10 attempts, logging each failure.

        Args:
            decision: DecisionInput containing criteria assessments,
                verification result, briefing packet, and execution trace.

        Returns:
            RoutingResult with disposition (APPROVED or ESCALATED) and
            optional escalation package.
        """
        # Determine if all criteria are MET
        all_criteria_met = self._all_criteria_met(decision.criteria_assessments)

        # Determine if verification passed
        verification_passed = decision.verification_result == VerificationResult.PASS

        # CORE SAFETY INVARIANT: Only auto-approve if ALL criteria MET AND verification PASS
        if all_criteria_met and verification_passed:
            return RoutingResult(
                disposition=Disposition.APPROVED,
                escalation_package=None,
                delivery_attempts=0,
                pending=False,
            )

        # Otherwise, escalate to Medical Director
        escalation_reason = self._build_escalation_reason(
            decision.criteria_assessments, decision.verification_result
        )

        escalation_package = EscalationPackage(
            briefing_packet=decision.briefing_packet,
            criteria_assessments=decision.criteria_assessments,
            challenger_findings=decision.challenger_findings,
            execution_trace=decision.execution_trace,
            escalation_reason=escalation_reason,
        )

        # Attempt to deliver to MD queue with retry logic
        delivery_result = await self._deliver_to_md_queue(
            escalation_package, decision.briefing_packet.request_id
        )

        return RoutingResult(
            disposition=Disposition.ESCALATED,
            escalation_package=escalation_package,
            delivery_attempts=delivery_result.attempts,
            pending=delivery_result.still_pending,
        )

    def _all_criteria_met(
        self, assessments: list[CriterionAssessment]
    ) -> bool:
        """Check if all criteria evaluations are MET.

        Returns True only if every criterion has status MET.
        Returns False if any criterion is NOT_MET or INDETERMINATE,
        or if the assessments list is empty.
        """
        if not assessments:
            return False
        return all(
            assessment.status == CriterionStatus.MET
            for assessment in assessments
        )

    def _build_escalation_reason(
        self,
        assessments: list[CriterionAssessment],
        verification_result: VerificationResult,
    ) -> str:
        """Build a human-readable reason for escalation."""
        reasons = []

        # Check for failed criteria
        not_met = [
            a for a in assessments if a.status == CriterionStatus.NOT_MET
        ]
        indeterminate = [
            a for a in assessments if a.status == CriterionStatus.INDETERMINATE
        ]

        if not_met:
            criteria_names = ", ".join(a.criterion_name for a in not_met)
            reasons.append(f"Criteria NOT_MET: {criteria_names}")

        if indeterminate:
            criteria_names = ", ".join(a.criterion_name for a in indeterminate)
            reasons.append(f"Criteria INDETERMINATE: {criteria_names}")

        if verification_result == VerificationResult.FAIL:
            reasons.append("OPA Challenger verification FAILED")

        if not reasons:
            reasons.append("Escalated for Medical Director review")

        return "; ".join(reasons)

    async def _deliver_to_md_queue(
        self, package: EscalationPackage, request_id: str
    ) -> "_DeliveryResult":
        """Deliver escalation package to MD queue with retry logic.

        Requirement 9.6: On queue unavailable, hold in pending state,
        retry at 30-second intervals up to 10 attempts, logging each
        failed attempt to the audit trail.

        Args:
            package: The escalation package to deliver.
            request_id: The PA request ID for audit logging.

        Returns:
            _DeliveryResult with attempt count and pending status.
        """
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                success = await self._md_queue.enqueue(package)
                if success:
                    return _DeliveryResult(
                        attempts=attempt, still_pending=False
                    )
            except Exception as e:
                logger.error(
                    f"MD queue delivery attempt {attempt}/{MAX_RETRY_ATTEMPTS} "
                    f"failed for request {request_id}: {e}"
                )

            # Log the failed attempt to audit trail
            await self._log_delivery_failure(
                request_id=request_id,
                attempt=attempt,
                reason=f"MD queue unavailable or delivery failed on attempt {attempt}",
            )

            # Wait before retrying (except on last attempt)
            if attempt < MAX_RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_INTERVAL_SECONDS)

        # All attempts exhausted — remain in pending state
        logger.warning(
            f"MD queue delivery exhausted all {MAX_RETRY_ATTEMPTS} attempts "
            f"for request {request_id}. Holding in pending state."
        )
        return _DeliveryResult(
            attempts=MAX_RETRY_ATTEMPTS, still_pending=True
        )

    async def _log_delivery_failure(
        self, request_id: str, attempt: int, reason: str
    ) -> None:
        """Log a failed delivery attempt to the audit trail.

        Args:
            request_id: The PA request ID.
            attempt: The attempt number (1-based).
            reason: Description of the failure.
        """
        try:
            await self._audit.record_entry(
                request_id=request_id,
                identity_id="system:human_gate_service",
                category=TraceCategory.DECISION_STEP,
                details={
                    "event": "md_queue_delivery_failure",
                    "attempt": attempt,
                    "max_attempts": MAX_RETRY_ATTEMPTS,
                    "reason": reason,
                    "retry_interval_seconds": RETRY_INTERVAL_SECONDS,
                },
            )
        except Exception as e:
            # If audit logging itself fails, log to standard logger
            # but do not halt retry logic
            logger.error(
                f"Failed to log delivery failure to audit trail: {e}"
            )


# =============================================================================
# Internal helpers
# =============================================================================


@dataclass
class _DeliveryResult:
    """Internal result of MD queue delivery attempts."""

    attempts: int
    still_pending: bool


# =============================================================================
# Protocol for Audit Trail (to avoid circular imports)
# =============================================================================


class AuditTrailServiceProtocol(Protocol):
    """Protocol for the audit trail service interface."""

    async def record_entry(
        self,
        request_id: str,
        identity_id: str,
        category: TraceCategory,
        details: Optional[dict] = None,
    ) -> None:
        """Record an entry to the immutable audit trail."""
        ...
