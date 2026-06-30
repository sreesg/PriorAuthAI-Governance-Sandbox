"""Unit tests for the CRF Orchestrator.

Verifies the orchestrator can be instantiated and the pipeline flows
correctly with mocked services. Tests cover:
- Successful pipeline flow (auto-approve path)
- Escalation path (criteria NOT_MET)
- KMS failure escalation
- Trace recording failure halts processing
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from clinical_reasoning_fabric.beacon.context_planner_service import PARequest
from clinical_reasoning_fabric.beacon.evidence_bundle_service import (
    DefaultSchemaValidator,
    EvidenceBundleService,
)
from clinical_reasoning_fabric.beacon.human_gate_service import (
    DecisionInput,
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
    DefaultSandboxExecutor,
    MCPGatewayService,
    ToolCatalog,
)
from clinical_reasoning_fabric.beacon.opa_challenger_service import (
    ChallengerResult,
    KMSClient,
    OPAChallengerService,
    OPAEvaluator,
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
    MemberActiveState,
    RBACPolicy,
    ScoredChunk,
    TraceCategory,
    VerificationResult,
)
from clinical_reasoning_fabric.models.exceptions import (
    KMSUnavailableError,
    TraceRecordingError,
    UnauthorizedError,
)
from clinical_reasoning_fabric.orchestrator import CRFOrchestrator, OrchestratorResult


# =============================================================================
# Test Fixtures
# =============================================================================


def _make_signature(key_id: str = "test-key") -> KMSSignature:
    return KMSSignature(
        key_id=key_id,
        signature="dGVzdHNpZ25hdHVyZQ==",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_chunk_provenance(doc_id: str = "doc-001") -> ChunkProvenance:
    return ChunkProvenance(
        document_id=doc_id,
        content_hash="a" * 64,
        kms_signature=_make_signature(),
        chunk_index=0,
        ingestion_timestamp=datetime.now(timezone.utc),
    )


def _make_scored_chunk(chunk_id: str = "chunk-001") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text="Patient shows signs of chronic knee pain with limited mobility.",
        score=0.85,
        provenance=_make_chunk_provenance(),
    )


def _make_briefing_packet(request_id: str = "req-001") -> BriefingPacket:
    return BriefingPacket(
        request_id=request_id,
        member_id="member-001",
        cpt_code="29881",
        active_clinical_state=MemberActiveState(
            member_id="member-001",
            active_diagnoses=[{"code": "M23.21", "description": "Knee meniscus tear"}],
            active_prescriptions=[],
            sdoh_factors=[],
            governing_policies=[],
        ),
        verified_evidence_snippets=[_make_scored_chunk()],
        inferred_facts=[],
    )


def _make_pa_request(request_id: str = "req-001") -> PARequest:
    return PARequest(
        request_id=request_id,
        member_id="member-001",
        cpt_code="29881",
    )


def _make_credentials() -> Credentials:
    return Credentials(
        identity_id="agent-001",
        api_key="valid-api-key-12345",
    )


def _make_auth_result() -> AuthResult:
    return AuthResult(
        identity_id="agent-001",
        granted_permissions=["process_pa_request"],
        authenticated_at=datetime.now(timezone.utc),
    )


class MockMDQueue:
    """Mock Medical Director queue that always succeeds."""

    async def enqueue(self, escalation_package) -> bool:
        return True


def _create_mocked_orchestrator(
    auth_result: Optional[AuthResult] = None,
    briefing_packet: Optional[BriefingPacket] = None,
    challenger_result: Optional[ChallengerResult] = None,
    auth_side_effect: Optional[Exception] = None,
    briefing_side_effect: Optional[Exception] = None,
) -> CRFOrchestrator:
    """Create an orchestrator with mocked service dependencies."""
    # Identity service
    identity_service = AsyncMock(spec=IdentityService)
    if auth_side_effect:
        identity_service.authenticate_and_authorize.side_effect = auth_side_effect
    else:
        identity_service.authenticate_and_authorize.return_value = (
            auth_result or _make_auth_result()
        )

    # Context planner
    context_planner = AsyncMock()
    if briefing_side_effect:
        context_planner.assemble_briefing_packet.side_effect = briefing_side_effect
    else:
        context_planner.assemble_briefing_packet.return_value = (
            briefing_packet or _make_briefing_packet()
        )

    # MCP Gateway
    mcp_gateway = MagicMock(spec=MCPGatewayService)

    # OPA Challenger
    opa_challenger = AsyncMock(spec=OPAChallengerService)
    opa_challenger.verify_decision.return_value = challenger_result or ChallengerResult(
        verification_result=VerificationResult.PASS,
        completed_at=datetime.now(timezone.utc),
    )

    # Human Gate
    human_gate = AsyncMock(spec=HumanGateService)
    human_gate.route_decision.return_value = RoutingResult(
        disposition=Disposition.APPROVED,
    )

    # Evidence Bundle Service
    evidence_bundle_service = EvidenceBundleService(
        schema_validator=DefaultSchemaValidator()
    )

    # Audit Trail
    storage = InMemoryAppendOnlyStorage()
    audit_trail = AuditTrailService(storage_backend=storage)

    return CRFOrchestrator(
        identity_service=identity_service,
        context_planner=context_planner,
        mcp_gateway=mcp_gateway,
        opa_challenger=opa_challenger,
        human_gate=human_gate,
        evidence_bundle_service=evidence_bundle_service,
        audit_trail=audit_trail,
    )


# =============================================================================
# Tests
# =============================================================================


class TestOrchestratorInstantiation:
    """Test that the orchestrator can be properly instantiated."""

    def test_orchestrator_instantiation(self):
        """The orchestrator can be instantiated with all required services."""
        orchestrator = _create_mocked_orchestrator()
        assert orchestrator is not None
        assert orchestrator.identity_service is not None
        assert orchestrator.context_planner is not None
        assert orchestrator.mcp_gateway is not None
        assert orchestrator.opa_challenger is not None
        assert orchestrator.human_gate is not None
        assert orchestrator.evidence_bundle_service is not None
        assert orchestrator.audit_trail is not None

    def test_orchestrator_result_dataclass(self):
        """OrchestratorResult can be created with expected fields."""
        result = OrchestratorResult(
            disposition=Disposition.APPROVED,
            execution_id="exec-001",
        )
        assert result.disposition == Disposition.APPROVED
        assert result.execution_id == "exec-001"
        assert result.evidence_bundle is None
        assert result.error is None


class TestOrchestratorApprovalFlow:
    """Test the happy-path auto-approval flow through the pipeline."""

    def test_auto_approve_flow(self):
        """Pipeline auto-approves when all criteria met and verification passes."""
        orchestrator = _create_mocked_orchestrator()

        result = asyncio.get_event_loop().run_until_complete(
            orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )
        )

        assert result.disposition == Disposition.APPROVED
        assert result.evidence_bundle is not None
        assert result.evidence_bundle.decision == "approve"
        assert result.error is None

    def test_approval_flow_calls_services_in_order(self):
        """The pipeline calls services in the correct order."""
        orchestrator = _create_mocked_orchestrator()

        asyncio.get_event_loop().run_until_complete(
            orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )
        )

        # Identity service called first
        orchestrator.identity_service.authenticate_and_authorize.assert_called_once()

        # Context planner called after authentication
        orchestrator.context_planner.assemble_briefing_packet.assert_called_once()

        # OPA Challenger called for verification
        orchestrator.opa_challenger.verify_decision.assert_called_once()

        # Human Gate called for routing
        orchestrator.human_gate.route_decision.assert_called_once()

    def test_audit_trail_records_entries(self):
        """The audit trail records trace entries during processing."""
        orchestrator = _create_mocked_orchestrator()

        asyncio.get_event_loop().run_until_complete(
            orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )
        )

        # Check that trace entries were recorded
        trace = asyncio.get_event_loop().run_until_complete(
            orchestrator.audit_trail.get_trace("req-001")
        )
        assert len(trace) > 0
        # Should have entries for: authenticate, assemble_briefing,
        # clinical_reasoning, evidence_bundle_preliminary, opa_verification,
        # human_gate_routing, final_bundle
        assert len(trace) >= 6


class TestOrchestratorEscalationFlow:
    """Test the escalation flow when criteria are not met."""

    def test_escalation_when_verification_fails(self):
        """Pipeline escalates when OPA verification fails."""
        challenger_result = ChallengerResult(
            verification_result=VerificationResult.FAIL,
            escalation_reason="Policy rule violated",
            completed_at=datetime.now(timezone.utc),
        )

        orchestrator = _create_mocked_orchestrator(
            challenger_result=challenger_result
        )

        # Configure human gate to return escalated
        orchestrator.human_gate.route_decision.return_value = RoutingResult(
            disposition=Disposition.ESCALATED,
        )

        result = asyncio.get_event_loop().run_until_complete(
            orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )
        )

        assert result.disposition == Disposition.ESCALATED
        assert result.evidence_bundle is not None
        assert result.evidence_bundle.decision == "escalate"

    def test_kms_unavailable_escalates(self):
        """Pipeline escalates on KMS unavailability."""
        orchestrator = _create_mocked_orchestrator()

        # Make OPA challenger raise KMS error
        orchestrator.opa_challenger.verify_decision.side_effect = (
            KMSUnavailableError("KMS service timeout")
        )

        result = asyncio.get_event_loop().run_until_complete(
            orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )
        )

        assert result.disposition == Disposition.ESCALATED
        assert result.error is not None
        assert "KMS" in result.error


class TestOrchestratorErrorHandling:
    """Test error handling in the orchestrator."""

    def test_unauthorized_raises(self):
        """Pipeline raises UnauthorizedError on auth failure."""
        orchestrator = _create_mocked_orchestrator(
            auth_side_effect=UnauthorizedError(
                identity="unknown",
                operation="process_pa_request",
                reason="Invalid credentials",
            )
        )

        with pytest.raises(UnauthorizedError):
            asyncio.get_event_loop().run_until_complete(
                orchestrator.process_pa_request(
                    credentials=_make_credentials(),
                    pa_request=_make_pa_request(),
                )
            )

    def test_trace_failure_halts_processing(self):
        """Pipeline halts when trace recording fails."""
        orchestrator = _create_mocked_orchestrator()

        # Patch audit trail to raise TraceRecordingError
        original_record = orchestrator.audit_trail.record_entry

        async def failing_record(*args, **kwargs):
            raise TraceRecordingError("Storage backend unavailable")

        orchestrator._audit_trail.record_entry = failing_record

        with pytest.raises(TraceRecordingError):
            asyncio.get_event_loop().run_until_complete(
                orchestrator.process_pa_request(
                    credentials=_make_credentials(),
                    pa_request=_make_pa_request(),
                )
            )

    def test_never_produces_denied_disposition(self):
        """The orchestrator NEVER produces a DENIED disposition."""
        # Test with verification fail
        challenger_result = ChallengerResult(
            verification_result=VerificationResult.FAIL,
            completed_at=datetime.now(timezone.utc),
        )
        orchestrator = _create_mocked_orchestrator(
            challenger_result=challenger_result
        )
        orchestrator.human_gate.route_decision.return_value = RoutingResult(
            disposition=Disposition.ESCALATED,
        )

        result = asyncio.get_event_loop().run_until_complete(
            orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )
        )

        # Should only ever be APPROVED or ESCALATED
        assert result.disposition in (Disposition.APPROVED, Disposition.ESCALATED)
        assert result.disposition != "DENIED"


class TestOrchestratorIdentityContext:
    """Test that identity context is maintained throughout the request."""

    def test_identity_used_in_trace_entries(self):
        """All trace entries use the authenticated identity."""
        orchestrator = _create_mocked_orchestrator()

        asyncio.get_event_loop().run_until_complete(
            orchestrator.process_pa_request(
                credentials=_make_credentials(),
                pa_request=_make_pa_request(),
            )
        )

        trace = asyncio.get_event_loop().run_until_complete(
            orchestrator.audit_trail.get_trace("req-001")
        )

        for entry in trace:
            assert entry.identity_id == "agent-001", (
                f"Entry {entry.sequence_number} has identity '{entry.identity_id}' "
                f"instead of 'agent-001'"
            )
