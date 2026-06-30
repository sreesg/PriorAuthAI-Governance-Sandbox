"""Integration tests for end-to-end PA flow through CRFOrchestrator.

Tests the full BEACON pipeline with mocked external dependencies (KMS, Qdrant,
Neo4j, LLM). Verifies auto-approve path, escalation path, KMS failure path,
full audit trail recording, Evidence Bundle production, and the no-denial invariant.

Validates: Requirements 9.1, 9.2, 9.3, 7.1, 8.5, 13.5, 14.1
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from clinical_reasoning_fabric.beacon.context_planner_service import (
    ContextPlannerService,
    PARequest,
)
from clinical_reasoning_fabric.beacon.evidence_bundle_service import (
    EvidenceBundleService,
)
from clinical_reasoning_fabric.beacon.human_gate_service import (
    HumanGateService,
    MedicalDirectorQueue,
    RoutingResult,
)
from clinical_reasoning_fabric.beacon.identity_service import (
    Credentials,
    IdentityService,
    MaskingService,
)
from clinical_reasoning_fabric.beacon.mcp_gateway_service import (
    MCPGatewayService,
)
from clinical_reasoning_fabric.beacon.opa_challenger_service import (
    ChallengerResult,
    OPAChallengerService,
    PolicyEvaluationResult,
    PolicyViolation,
    SignatureVerificationResult,
)
from clinical_reasoning_fabric.models.core import (
    AuthResult,
    BriefingPacket,
    ChunkProvenance,
    CriterionAssessment,
    CriterionStatus,
    Disposition,
    EvidenceBundle,
    KMSSignature,
    LineageEntry,
    MemberActiveState,
    RBACPolicy,
    ScoredChunk,
    TraceCategory,
    TraceEntry,
    VerificationResult,
)
from clinical_reasoning_fabric.models.exceptions import (
    KMSUnavailableError,
    TraceRecordingError,
)
from clinical_reasoning_fabric.orchestrator import (
    CRFOrchestrator,
    OrchestratorResult,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


def _make_kms_signature(key_id: str = "test-key-001") -> KMSSignature:
    """Create a valid KMSSignature for testing."""
    return KMSSignature(
        key_id=key_id,
        signature="dGVzdC1zaWduYXR1cmUtYmFzZTY0",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_scored_chunk(
    chunk_id: str = "chunk-001",
    text: str = "Patient has documented history of knee osteoarthritis",
    score: float = 0.85,
    doc_id: str = "doc-001",
) -> ScoredChunk:
    """Create a ScoredChunk with valid provenance."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        provenance=ChunkProvenance(
            document_id=doc_id,
            content_hash="a" * 64,
            kms_signature=_make_kms_signature(),
            chunk_index=0,
            ingestion_timestamp=datetime.now(timezone.utc),
        ),
    )


def _make_briefing_packet(
    request_id: str = "req-001",
    member_id: str = "member-001",
    cpt_code: str = "27447",
    with_evidence: bool = True,
    with_diagnoses: bool = True,
) -> BriefingPacket:
    """Create a BriefingPacket for testing."""
    snippets = []
    if with_evidence:
        snippets = [
            _make_scored_chunk("chunk-001", "Patient meets knee arthroplasty criteria", 0.92),
            _make_scored_chunk("chunk-002", "Conservative therapy for 6 months documented", 0.87),
        ]

    diagnoses = []
    if with_diagnoses:
        diagnoses = [{"code": "M17.11", "description": "Primary OA right knee", "status": "active"}]

    return BriefingPacket(
        request_id=request_id,
        member_id=member_id,
        cpt_code=cpt_code,
        active_clinical_state=MemberActiveState(
            member_id=member_id,
            active_diagnoses=diagnoses,
            active_prescriptions=[{"drug": "Naproxen", "dose": "500mg"}],
            sdoh_factors=[],
            governing_policies=[],
        ),
        verified_evidence_snippets=snippets,
        inferred_facts=[],
        no_evidence_found=not with_evidence,
    )


def _make_credentials() -> Credentials:
    """Create valid test credentials."""
    return Credentials(identity_id="clinician-001", api_key="test-api-key-valid")


def _make_pa_request(request_id: str = "req-001") -> PARequest:
    """Create a PA request."""
    return PARequest(
        request_id=request_id,
        member_id="member-001",
        cpt_code="27447",
        clinical_context="Knee replacement evaluation",
    )


@pytest.fixture
def audit_trail():
    """Create a real AuditTrailService with in-memory storage."""
    storage = InMemoryAppendOnlyStorage()
    return AuditTrailService(storage_backend=storage)


@pytest.fixture
def identity_service():
    """Create an IdentityService that always authenticates successfully."""
    rbac_policy = RBACPolicy(
        policy_id="test-policy",
        roles={"clinician": ["process_pa_request", "read_patient_data"]},
        identity_role_assignments={"clinician-001": "clinician"},
    )
    masking_service = MaskingService()
    return IdentityService(rbac_policy=rbac_policy, masking_service=masking_service)


@pytest.fixture
def context_planner():
    """Create a mocked ContextPlannerService."""
    mock = AsyncMock(spec=ContextPlannerService)
    mock.assemble_briefing_packet = AsyncMock(
        return_value=_make_briefing_packet()
    )
    return mock


@pytest.fixture
def mcp_gateway():
    """Create a mocked MCPGatewayService."""
    return MagicMock(spec=MCPGatewayService)


@pytest.fixture
def opa_challenger_pass():
    """Create an OPAChallengerService mock that returns PASS."""
    mock = AsyncMock(spec=OPAChallengerService)
    mock.verify_decision = AsyncMock(
        return_value=ChallengerResult(
            verification_result=VerificationResult.PASS,
            signature_result=SignatureVerificationResult(
                valid=["chunk-001", "chunk-002"],
                invalid=[],
                missing=[],
                all_valid=True,
            ),
            policy_result=PolicyEvaluationResult(passed=True),
        )
    )
    return mock


@pytest.fixture
def opa_challenger_fail():
    """Create an OPAChallengerService mock that returns FAIL."""
    mock = AsyncMock(spec=OPAChallengerService)
    mock.verify_decision = AsyncMock(
        return_value=ChallengerResult(
            verification_result=VerificationResult.FAIL,
            signature_result=SignatureVerificationResult(
                valid=["chunk-001"],
                invalid=["chunk-002"],
                missing=[],
                all_valid=False,
            ),
            policy_result=PolicyEvaluationResult(
                passed=False,
                violated_rules=[
                    PolicyViolation(
                        rule_id="rule-001",
                        description="Missing conservative therapy documentation",
                    )
                ],
            ),
            violated_rules=[
                PolicyViolation(
                    rule_id="rule-001",
                    description="Missing conservative therapy documentation",
                )
            ],
            escalation_reason="Policy violation: rule-001",
        )
    )
    return mock


@pytest.fixture
def md_queue():
    """Create a mocked MedicalDirectorQueue that always succeeds."""
    mock = AsyncMock(spec=MedicalDirectorQueue)
    mock.enqueue = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def human_gate(md_queue, audit_trail):
    """Create a real HumanGateService with mocked queue."""
    return HumanGateService(md_queue=md_queue, audit=audit_trail)


@pytest.fixture
def evidence_bundle_service():
    """Create a real EvidenceBundleService."""
    return EvidenceBundleService()


def _build_orchestrator(
    identity_service,
    context_planner,
    mcp_gateway,
    opa_challenger,
    human_gate,
    evidence_bundle_service,
    audit_trail,
) -> CRFOrchestrator:
    """Build a CRFOrchestrator with given services."""
    return CRFOrchestrator(
        identity_service=identity_service,
        context_planner=context_planner,
        mcp_gateway=mcp_gateway,
        opa_challenger=opa_challenger,
        human_gate=human_gate,
        evidence_bundle_service=evidence_bundle_service,
        audit_trail=audit_trail,
    )


# ---------------------------------------------------------------------------
# Tests: Auto-Approve Path
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAutoApprovePath:
    """Tests for the auto-approve path: all criteria MET + verification PASS."""

    @pytest.mark.asyncio
    async def test_auto_approve_produces_approved_disposition(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_pass,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """All criteria MET + verification PASS results in APPROVED disposition."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_pass,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.disposition == Disposition.APPROVED
        assert result.error is None

    @pytest.mark.asyncio
    async def test_auto_approve_produces_evidence_bundle(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_pass,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """Auto-approve path produces a valid Evidence Bundle."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_pass,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.evidence_bundle is not None
        bundle = result.evidence_bundle
        assert bundle.decision == "approve"
        assert bundle.execution_id == result.execution_id
        assert len(bundle.lineage_trail) >= 1
        assert len(bundle.original_document_signatures) >= 1

    @pytest.mark.asyncio
    async def test_auto_approve_records_full_audit_trail(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_pass,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """Auto-approve path records trace entries at each pipeline step."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_pass,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(request_id="req-audit-001"),
        )

        trace = await audit_trail.get_trace("req-audit-001")
        assert len(trace) >= 5  # auth, briefing, reasoning, verification, routing, bundle

        # Verify trace entries are properly ordered
        seq_numbers = [e.sequence_number for e in trace]
        assert seq_numbers == sorted(seq_numbers)

        # Verify steps are covered
        step_names = [
            e.details.get("step") for e in trace if e.details and "step" in e.details
        ]
        assert "authenticate" in step_names
        assert "assemble_briefing" in step_names
        assert "clinical_reasoning" in step_names
        assert "opa_verification" in step_names
        assert "human_gate_routing" in step_names

    @pytest.mark.asyncio
    async def test_auto_approve_never_produces_denial(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_pass,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """No-denial invariant: result is never DENIED."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_pass,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        # The Disposition enum doesn't even have DENIED, but let's be explicit
        assert result.disposition in (Disposition.APPROVED, Disposition.ESCALATED)
        assert result.disposition != "DENIED"


# ---------------------------------------------------------------------------
# Tests: Escalation Path
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEscalationPath:
    """Tests for escalation: criterion NOT_MET or verification FAIL."""

    @pytest.mark.asyncio
    async def test_not_met_criterion_escalates(
        self,
        identity_service,
        mcp_gateway,
        opa_challenger_pass,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """A NOT_MET criterion leads to ESCALATED disposition."""
        # Context planner returns briefing with no evidence (INDETERMINATE)
        context_planner = AsyncMock(spec=ContextPlannerService)
        context_planner.assemble_briefing_packet = AsyncMock(
            return_value=_make_briefing_packet(with_evidence=False, with_diagnoses=False)
        )

        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_pass,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.disposition == Disposition.ESCALATED

    @pytest.mark.asyncio
    async def test_verification_fail_escalates(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_fail,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """OPA verification FAIL leads to ESCALATED disposition."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_fail,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.disposition == Disposition.ESCALATED

    @pytest.mark.asyncio
    async def test_escalation_produces_evidence_bundle(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_fail,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """Escalation path still produces a valid Evidence Bundle."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_fail,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.evidence_bundle is not None
        bundle = result.evidence_bundle
        assert bundle.decision == "escalate"
        assert len(bundle.lineage_trail) >= 1
        assert len(bundle.original_document_signatures) >= 1

    @pytest.mark.asyncio
    async def test_escalation_never_produces_denial(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_fail,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """No-denial invariant holds on escalation path."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_fail,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.disposition in (Disposition.APPROVED, Disposition.ESCALATED)

    @pytest.mark.asyncio
    async def test_escalation_records_full_audit_trail(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_fail,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """Escalation path records complete audit trail."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_fail,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(request_id="req-esc-001"),
        )

        trace = await audit_trail.get_trace("req-esc-001")
        assert len(trace) >= 5

        # Verify verification result is recorded as FAIL
        verification_entries = [
            e for e in trace
            if e.details and e.details.get("step") == "opa_verification"
        ]
        assert len(verification_entries) == 1
        assert verification_entries[0].details["verification_result"] == "FAIL"


# ---------------------------------------------------------------------------
# Tests: KMS Failure Path
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestKMSFailurePath:
    """Tests for KMS unavailable path — should escalate, not deny."""

    @pytest.mark.asyncio
    async def test_kms_failure_escalates(
        self,
        identity_service,
        mcp_gateway,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """KMS unavailability results in escalation, not denial."""
        # Context planner raises KMSUnavailableError during briefing assembly
        context_planner = AsyncMock(spec=ContextPlannerService)
        context_planner.assemble_briefing_packet = AsyncMock(
            side_effect=KMSUnavailableError(
                reason="KMS service timeout",
            )
        )

        opa_challenger = AsyncMock(spec=OPAChallengerService)

        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.disposition == Disposition.ESCALATED
        assert result.error is not None
        assert "KMS" in result.error

    @pytest.mark.asyncio
    async def test_kms_failure_never_denies(
        self,
        identity_service,
        mcp_gateway,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """KMS failure path never produces a denial."""
        context_planner = AsyncMock(spec=ContextPlannerService)
        context_planner.assemble_briefing_packet = AsyncMock(
            side_effect=KMSUnavailableError(
                reason="Connection refused",
            )
        )
        opa_challenger = AsyncMock(spec=OPAChallengerService)

        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )

        assert result.disposition in (Disposition.APPROVED, Disposition.ESCALATED)

    @pytest.mark.asyncio
    async def test_kms_failure_records_trace(
        self,
        identity_service,
        mcp_gateway,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """KMS failure is recorded in the audit trail."""
        context_planner = AsyncMock(spec=ContextPlannerService)
        context_planner.assemble_briefing_packet = AsyncMock(
            side_effect=KMSUnavailableError(
                reason="Service unavailable",
            )
        )
        opa_challenger = AsyncMock(spec=OPAChallengerService)

        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(request_id="req-kms-001"),
        )

        trace = await audit_trail.get_trace("req-kms-001")
        # Should have at least auth trace + kms failure trace
        assert len(trace) >= 2

        kms_entries = [
            e for e in trace
            if e.details and e.details.get("step") == "kms_failure"
        ]
        assert len(kms_entries) == 1
        assert kms_entries[0].details["status"] == "escalated"


# ---------------------------------------------------------------------------
# Tests: Trace Recording Failure (Halt)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTraceRecordingFailure:
    """Tests for trace recording failure — should halt PA processing."""

    @pytest.mark.asyncio
    async def test_trace_failure_halts_processing(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_pass,
        human_gate,
        evidence_bundle_service,
    ):
        """TraceRecordingError halts processing and raises."""
        # Create an audit trail that fails on record
        failing_storage = AsyncMock(spec=InMemoryAppendOnlyStorage)
        failing_storage.append = AsyncMock(
            side_effect=Exception("Storage backend unavailable")
        )
        audit_trail = AuditTrailService(storage_backend=failing_storage)

        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_pass,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        with pytest.raises(TraceRecordingError):
            await orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )


# ---------------------------------------------------------------------------
# Tests: No-Denial Invariant (all scenarios)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNoDenialInvariant:
    """The system never produces an automated denial regardless of input."""

    @pytest.mark.asyncio
    async def test_no_denial_with_all_criteria_met(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_pass,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """All MET criteria results in APPROVED, not denied."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_pass,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )
        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )
        assert result.disposition != "DENIED"
        assert result.disposition in (Disposition.APPROVED, Disposition.ESCALATED)

    @pytest.mark.asyncio
    async def test_no_denial_with_verification_fail(
        self,
        identity_service,
        context_planner,
        mcp_gateway,
        opa_challenger_fail,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """Verification FAIL results in escalation, not denial."""
        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger_fail,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )
        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )
        assert result.disposition != "DENIED"

    @pytest.mark.asyncio
    async def test_no_denial_with_unexpected_error(
        self,
        identity_service,
        mcp_gateway,
        human_gate,
        evidence_bundle_service,
        audit_trail,
    ):
        """Unexpected errors result in escalation, not denial."""
        context_planner = AsyncMock(spec=ContextPlannerService)
        context_planner.assemble_briefing_packet = AsyncMock(
            side_effect=RuntimeError("Unexpected failure")
        )
        opa_challenger = AsyncMock(spec=OPAChallengerService)

        orchestrator = _build_orchestrator(
            identity_service,
            context_planner,
            mcp_gateway,
            opa_challenger,
            human_gate,
            evidence_bundle_service,
            audit_trail,
        )

        result = await orchestrator.process_pa_request(
            credentials=_make_credentials(),
            pa_request=_make_pa_request(),
        )
        assert result.disposition == Disposition.ESCALATED
        assert result.disposition != "DENIED"
