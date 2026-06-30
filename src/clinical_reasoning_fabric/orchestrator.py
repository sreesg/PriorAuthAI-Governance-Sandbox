"""CRF Orchestrator — Wires all services together for PA request processing.

The CRFOrchestrator class implements the full BEACON pipeline:
  Identity → Context Planner → MCP Gateway → OPA Challenger → Human Gates → Evidence Bundle

It accepts a PA request and runs through the complete flow, returning the final
Disposition and Evidence Bundle. The Audit Trail Service records trace entries
at every step. Error propagation follows these rules:
  - Trace recording failure → halt processing
  - KMS/signature failure → escalate to Medical Director
  - MD queue unavailable → retry (handled by HumanGateService)

Requirements referenced: 1.1-14.9 (full system integration)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from clinical_reasoning_fabric.beacon.audit_trail_service import AuditTrailService
from clinical_reasoning_fabric.beacon.context_planner_service import (
    ContextPlannerService,
    PARequest,
)
from clinical_reasoning_fabric.beacon.evidence_bundle_service import (
    EvidenceBundleService,
)
from clinical_reasoning_fabric.beacon.human_gate_service import (
    DecisionInput,
    HumanGateService,
    RoutingResult,
)
from clinical_reasoning_fabric.beacon.identity_service import (
    Credentials,
    IdentityService,
)
from clinical_reasoning_fabric.beacon.mcp_gateway_service import MCPGatewayService
from clinical_reasoning_fabric.beacon.opa_challenger_service import (
    ChallengerResult,
    OPAChallengerService,
)
from clinical_reasoning_fabric.models.core import (
    AuthResult,
    BriefingPacket,
    CriterionAssessment,
    CriterionStatus,
    Disposition,
    EvidenceBundle,
    KMSSignature,
    LineageEntry,
    TraceCategory,
    TraceEntry,
    VerificationResult,
)
from clinical_reasoning_fabric.models.exceptions import (
    KMSUnavailableError,
    TraceRecordingError,
    UnauthorizedError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Orchestrator Result
# =============================================================================


@dataclass
class OrchestratorResult:
    """Result of a full PA request processing through the CRF pipeline."""

    disposition: Disposition
    """Final disposition: APPROVED or ESCALATED (never DENIED)."""

    evidence_bundle: Optional[EvidenceBundle] = None
    """The produced Evidence Bundle (present for both approve and escalate)."""

    routing_result: Optional[RoutingResult] = None
    """Routing result from the Human Gate."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique execution ID for this PA processing run."""

    error: Optional[str] = None
    """Error message if processing was halted."""


# =============================================================================
# CRF Orchestrator
# =============================================================================


class CRFOrchestrator:
    """Orchestrates the full CRF pipeline for Prior Authorization requests.

    Wires all services together in sequence:
    1. IdentityService — authenticate the request
    2. ContextPlannerService — assemble the Briefing Packet
    3. MCPGatewayService — available for tool invocations during reasoning
    4. OPAChallengerService — verify the decision independently
    5. HumanGateService — route to auto-approve or escalate
    6. EvidenceBundleService — produce the final Evidence Bundle
    7. AuditTrailService — records trace entries at every step

    The orchestrator does NOT duplicate logic from individual services.
    It simply calls them in sequence, passing results forward.
    """

    def __init__(
        self,
        identity_service: IdentityService,
        context_planner: ContextPlannerService,
        mcp_gateway: MCPGatewayService,
        opa_challenger: OPAChallengerService,
        human_gate: HumanGateService,
        evidence_bundle_service: EvidenceBundleService,
        audit_trail: AuditTrailService,
    ) -> None:
        """Initialize the orchestrator with all required services.

        Args:
            identity_service: BEACON L1 — authentication and RBAC.
            context_planner: BEACON L2 — Briefing Packet assembly.
            mcp_gateway: BEACON L3 — controlled tool invocation.
            opa_challenger: BEACON L5 — independent verification.
            human_gate: BEACON L7 — no-auto-denial routing.
            evidence_bundle_service: Evidence Bundle production.
            audit_trail: BEACON L6 — immutable trace recording.
        """
        self._identity_service = identity_service
        self._context_planner = context_planner
        self._mcp_gateway = mcp_gateway
        self._opa_challenger = opa_challenger
        self._human_gate = human_gate
        self._evidence_bundle_service = evidence_bundle_service
        self._audit_trail = audit_trail

    @property
    def identity_service(self) -> IdentityService:
        return self._identity_service

    @property
    def context_planner(self) -> ContextPlannerService:
        return self._context_planner

    @property
    def mcp_gateway(self) -> MCPGatewayService:
        return self._mcp_gateway

    @property
    def opa_challenger(self) -> OPAChallengerService:
        return self._opa_challenger

    @property
    def human_gate(self) -> HumanGateService:
        return self._human_gate

    @property
    def evidence_bundle_service(self) -> EvidenceBundleService:
        return self._evidence_bundle_service

    @property
    def audit_trail(self) -> AuditTrailService:
        return self._audit_trail

    async def process_pa_request(
        self,
        credentials: Credentials,
        pa_request: PARequest,
    ) -> OrchestratorResult:
        """Process a Prior Authorization request through the full BEACON pipeline.

        Steps:
        1. Authenticate — verify identity and permissions (L1)
        2. Assemble Briefing Packet — gather clinical context (L2)
        3. Reason — produce clinical criteria assessments (agent reasoning)
        4. Verify — OPA Challenger checks signatures and policy (L5)
        5. Route — Human Gate decides approve/escalate (L7)
        6. Bundle — produce Evidence Bundle with lineage

        The AuditTrailService records a trace entry at each step.

        Args:
            credentials: Authentication credentials for the request.
            pa_request: The PA request with member_id, cpt_code, etc.

        Returns:
            OrchestratorResult with final disposition and evidence bundle.

        Raises:
            TraceRecordingError: If audit trail recording fails (halts processing).
            UnauthorizedError: If authentication/authorization fails.
        """
        execution_id = str(uuid.uuid4())
        request_id = pa_request.request_id
        identity_id = "unknown"

        try:
            # ==================================================================
            # Step 1: Authenticate (BEACON L1)
            # ==================================================================
            auth_result = await self._authenticate(credentials, request_id)
            identity_id = auth_result.identity_id

            await self._record_trace(
                request_id=request_id,
                identity_id=identity_id,
                category=TraceCategory.AGENT_ACTION,
                details={"step": "authenticate", "status": "success"},
            )

            # ==================================================================
            # Step 2: Assemble Briefing Packet (BEACON L2)
            # ==================================================================
            briefing_packet = await self._assemble_briefing(pa_request)

            await self._record_trace(
                request_id=request_id,
                identity_id=identity_id,
                category=TraceCategory.CONTEXT_RETRIEVAL,
                details={
                    "step": "assemble_briefing",
                    "status": "success",
                    "snippet_count": len(briefing_packet.verified_evidence_snippets),
                    "inferred_fact_count": len(briefing_packet.inferred_facts),
                },
            )

            # ==================================================================
            # Step 3: Clinical Reasoning (produces criteria assessments)
            # ==================================================================
            criteria_assessments = self._assess_criteria(briefing_packet)

            await self._record_trace(
                request_id=request_id,
                identity_id=identity_id,
                category=TraceCategory.DECISION_STEP,
                details={
                    "step": "clinical_reasoning",
                    "status": "success",
                    "criteria_count": len(criteria_assessments),
                },
            )

            # ==================================================================
            # Step 4: Produce preliminary Evidence Bundle for verification
            # ==================================================================
            lineage_trail = self._build_lineage_trail(briefing_packet)
            signatures = self._collect_signatures(briefing_packet)

            preliminary_bundle = self._evidence_bundle_service.produce_bundle(
                execution_id=execution_id,
                decision="pending_verification",
                reason="Awaiting OPA Challenger verification",
                lineage_trail=lineage_trail,
                document_signatures=signatures,
            )

            await self._record_trace(
                request_id=request_id,
                identity_id=identity_id,
                category=TraceCategory.DECISION_STEP,
                details={
                    "step": "evidence_bundle_preliminary",
                    "status": "success",
                },
            )

            # ==================================================================
            # Step 5: OPA Challenger Verification (BEACON L5)
            # ==================================================================
            challenger_result = await self._verify_decision(preliminary_bundle)

            await self._record_trace(
                request_id=request_id,
                identity_id=identity_id,
                category=TraceCategory.DECISION_STEP,
                details={
                    "step": "opa_verification",
                    "status": "success",
                    "verification_result": challenger_result.verification_result.value,
                },
            )

            # ==================================================================
            # Step 6: Human Gate Routing (BEACON L7)
            # ==================================================================
            execution_trace = await self._audit_trail.get_trace(request_id)

            decision_input = DecisionInput(
                criteria_assessments=criteria_assessments,
                verification_result=challenger_result.verification_result,
                briefing_packet=briefing_packet,
                execution_trace=execution_trace,
                challenger_findings=challenger_result,
            )

            routing_result = await self._human_gate.route_decision(decision_input)

            await self._record_trace(
                request_id=request_id,
                identity_id=identity_id,
                category=TraceCategory.DECISION_STEP,
                details={
                    "step": "human_gate_routing",
                    "status": "success",
                    "disposition": routing_result.disposition.value,
                },
            )

            # ==================================================================
            # Step 7: Produce final Evidence Bundle
            # ==================================================================
            final_decision = (
                "approve" if routing_result.disposition == Disposition.APPROVED
                else "escalate"
            )
            final_reason = self._build_final_reason(
                routing_result, criteria_assessments, challenger_result
            )

            final_bundle = self._evidence_bundle_service.produce_bundle(
                execution_id=execution_id,
                decision=final_decision,
                reason=final_reason,
                lineage_trail=lineage_trail,
                document_signatures=signatures,
            )

            # Attach execution trace to the bundle
            final_trace = await self._audit_trail.get_trace(request_id)
            final_bundle.execution_trace = final_trace

            await self._record_trace(
                request_id=request_id,
                identity_id=identity_id,
                category=TraceCategory.DECISION_STEP,
                details={
                    "step": "final_bundle",
                    "status": "success",
                    "execution_id": execution_id,
                },
            )

            return OrchestratorResult(
                disposition=routing_result.disposition,
                evidence_bundle=final_bundle,
                routing_result=routing_result,
                execution_id=execution_id,
            )

        except TraceRecordingError as e:
            # Trace failure halts processing — no unaudited decisions
            logger.error(f"Trace recording failure, halting PA processing: {e}")
            raise

        except UnauthorizedError as e:
            # Authentication failure — log and re-raise
            logger.warning(f"Unauthorized PA request: {e}")
            raise

        except KMSUnavailableError as e:
            # KMS failure → escalate to Medical Director
            logger.error(f"KMS unavailable during PA processing: {e}")
            try:
                await self._record_trace(
                    request_id=request_id,
                    identity_id=identity_id,
                    category=TraceCategory.DECISION_STEP,
                    details={
                        "step": "kms_failure",
                        "status": "escalated",
                        "error": str(e),
                    },
                )
            except TraceRecordingError:
                raise

            return OrchestratorResult(
                disposition=Disposition.ESCALATED,
                execution_id=execution_id,
                error=f"KMS unavailable: {e}",
            )

        except Exception as e:
            # Any other unexpected error → escalate
            logger.error(f"Unexpected error during PA processing: {e}")
            try:
                await self._record_trace(
                    request_id=request_id,
                    identity_id=identity_id,
                    category=TraceCategory.DECISION_STEP,
                    details={
                        "step": "unexpected_error",
                        "status": "escalated",
                        "error": str(e),
                    },
                )
            except TraceRecordingError:
                raise

            return OrchestratorResult(
                disposition=Disposition.ESCALATED,
                execution_id=execution_id,
                error=f"Processing error: {e}",
            )

    # =========================================================================
    # Private helper methods
    # =========================================================================

    async def _authenticate(
        self, credentials: Credentials, request_id: str
    ) -> AuthResult:
        """Authenticate and authorize the PA request via IdentityService."""
        return await self._identity_service.authenticate_and_authorize(
            credentials=credentials,
            operation="process_pa_request",
        )

    async def _assemble_briefing(self, pa_request: PARequest) -> BriefingPacket:
        """Assemble the Briefing Packet via ContextPlannerService."""
        return await self._context_planner.assemble_briefing_packet(pa_request)

    def _assess_criteria(
        self, briefing_packet: BriefingPacket
    ) -> list[CriterionAssessment]:
        """Produce criteria assessments based on the Briefing Packet.

        In the full system, this would invoke the clinical reasoning agent
        via the MCP Gateway. For the orchestrator, we produce assessments
        based on available evidence.
        """
        assessments = []

        # If no evidence found, criteria are INDETERMINATE
        if briefing_packet.no_evidence_found:
            assessments.append(
                CriterionAssessment(
                    criterion_id="criterion-evidence",
                    criterion_name="Supporting Clinical Evidence",
                    status=CriterionStatus.INDETERMINATE,
                    evidence_references=[],
                    reasoning="No supporting evidence found in retrieval",
                )
            )
            return assessments

        # Assess based on available evidence
        evidence_refs = [
            s.chunk_id for s in briefing_packet.verified_evidence_snippets
        ]

        assessments.append(
            CriterionAssessment(
                criterion_id="criterion-evidence",
                criterion_name="Supporting Clinical Evidence",
                status=CriterionStatus.MET if evidence_refs else CriterionStatus.INDETERMINATE,
                evidence_references=evidence_refs,
                reasoning=(
                    f"Found {len(evidence_refs)} supporting evidence snippets"
                    if evidence_refs
                    else "No evidence references available"
                ),
            )
        )

        # Assess clinical necessity based on active diagnoses
        has_diagnoses = bool(
            briefing_packet.active_clinical_state.active_diagnoses
        )
        assessments.append(
            CriterionAssessment(
                criterion_id="criterion-medical-necessity",
                criterion_name="Medical Necessity",
                status=CriterionStatus.MET if has_diagnoses else CriterionStatus.INDETERMINATE,
                evidence_references=evidence_refs[:5],
                reasoning=(
                    "Active diagnosis supports medical necessity"
                    if has_diagnoses
                    else "No active diagnosis found to support necessity"
                ),
            )
        )

        return assessments

    def _build_lineage_trail(
        self, briefing_packet: BriefingPacket
    ) -> list[LineageEntry]:
        """Build lineage trail from the Briefing Packet evidence."""
        trail = []
        for snippet in briefing_packet.verified_evidence_snippets:
            trail.append(
                LineageEntry(
                    conclusion=f"Evidence from document {snippet.provenance.document_id}",
                    evidence_id=snippet.chunk_id,
                    retrieval_timestamp=snippet.provenance.ingestion_timestamp,
                )
            )

        # Ensure at least one entry
        if not trail:
            trail.append(
                LineageEntry(
                    conclusion="PA request processed with no direct evidence",
                    evidence_id=f"request-{briefing_packet.request_id}",
                    retrieval_timestamp=datetime.now(timezone.utc),
                )
            )

        return trail

    def _collect_signatures(
        self, briefing_packet: BriefingPacket
    ) -> list[KMSSignature]:
        """Collect KMS signatures from evidence snippets."""
        signatures = []
        seen_keys = set()
        for snippet in briefing_packet.verified_evidence_snippets:
            sig = snippet.provenance.kms_signature
            sig_key = f"{sig.key_id}:{sig.signature}"
            if sig_key not in seen_keys:
                signatures.append(sig)
                seen_keys.add(sig_key)

        # Ensure at least one signature
        if not signatures:
            signatures.append(
                KMSSignature(
                    key_id="default-key",
                    signature="no-evidence-placeholder-signature",
                    algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                    signed_at=datetime.now(timezone.utc),
                )
            )

        return signatures

    async def _verify_decision(
        self, evidence_bundle: EvidenceBundle
    ) -> ChallengerResult:
        """Run OPA Challenger verification on the evidence bundle."""
        return await self._opa_challenger.verify_decision(evidence_bundle)

    def _build_final_reason(
        self,
        routing_result: RoutingResult,
        criteria: list[CriterionAssessment],
        challenger: ChallengerResult,
    ) -> str:
        """Build a human-readable reason for the final decision."""
        if routing_result.disposition == Disposition.APPROVED:
            return (
                "All clinical criteria met and OPA verification passed. "
                "PA request auto-approved."
            )

        # Build escalation reason
        reasons = []
        not_met = [c for c in criteria if c.status == CriterionStatus.NOT_MET]
        indeterminate = [
            c for c in criteria if c.status == CriterionStatus.INDETERMINATE
        ]

        if not_met:
            names = ", ".join(c.criterion_name for c in not_met)
            reasons.append(f"Criteria NOT_MET: {names}")

        if indeterminate:
            names = ", ".join(c.criterion_name for c in indeterminate)
            reasons.append(f"Criteria INDETERMINATE: {names}")

        if challenger.verification_result == VerificationResult.FAIL:
            reasons.append("OPA Challenger verification FAILED")
            if challenger.violated_rules:
                rule_ids = ", ".join(
                    r.rule_id for r in challenger.violated_rules
                )
                reasons.append(f"Violated rules: {rule_ids}")

        return "; ".join(reasons) if reasons else "Escalated for Medical Director review"

    async def _record_trace(
        self,
        request_id: str,
        identity_id: str,
        category: TraceCategory,
        details: dict,
    ) -> None:
        """Record a trace entry via the AuditTrailService.

        Raises TraceRecordingError on failure (halts PA processing).
        """
        await self._audit_trail.record_entry(
            request_id=request_id,
            identity_id=identity_id,
            category=category,
            details=details,
        )
