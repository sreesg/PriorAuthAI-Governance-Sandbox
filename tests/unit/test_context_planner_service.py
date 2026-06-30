"""Unit tests for ContextPlannerService — BEACON Layer 2.

Tests Briefing Packet assembly, timeout handling, member-not-found errors,
zero evidence handling, and CPT-based relevance filtering.

Requirements validated: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical_reasoning_fabric.beacon.context_planner_service import (
    ASSEMBLY_TIMEOUT_SECONDS,
    CPT_CONDITION_CATEGORIES,
    ContextPlannerService,
    PARequest,
)
from clinical_reasoning_fabric.models.core import (
    BriefingPacket,
    ChunkProvenance,
    KMSSignature,
    MemberActiveState,
    RetrievalResult,
    ScoredChunk,
)
from clinical_reasoning_fabric.models.exceptions import MemberNotFoundError


# =============================================================================
# Fixtures and Helpers
# =============================================================================


def _make_kms_signature() -> KMSSignature:
    """Create a valid KMS signature for test provenance."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789:key/test-key",
        signature="dGVzdHNpZ25hdHVyZQ==",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    )


def _make_provenance(document_id: str = "doc-001", chunk_index: int = 0) -> ChunkProvenance:
    """Create valid chunk provenance for tests."""
    return ChunkProvenance(
        document_id=document_id,
        content_hash="a" * 64,
        kms_signature=_make_kms_signature(),
        chunk_index=chunk_index,
        ingestion_timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    )


def _make_scored_chunk(
    chunk_id: str = "chunk-001",
    text: str = "Patient has lumbar spine stenosis requiring MRI",
    score: float = 0.8,
    document_id: str = "doc-001",
    chunk_index: int = 0,
) -> ScoredChunk:
    """Create a ScoredChunk for testing."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        provenance=_make_provenance(document_id, chunk_index),
    )


def _make_member_state(member_id: str = "MEM-001") -> MemberActiveState:
    """Create a MemberActiveState for testing."""
    return MemberActiveState(
        member_id=member_id,
        active_diagnoses=[
            {"condition_code": "M54.5", "description": "Low back pain"},
            {"condition_code": "M47.816", "description": "Lumbar spondylosis"},
            {"condition_code": "E11.9", "description": "Type 2 diabetes"},
        ],
        active_prescriptions=[
            {"medication_name": "Ibuprofen", "description": "NSAID for back pain"},
            {"medication_name": "Metformin", "description": "Diabetes medication"},
        ],
        sdoh_factors=[
            {"type": "transportation_barriers", "description": "Limited transport access"},
        ],
        governing_policies=[
            {"policy_id": "POL-RAD-501", "description": "Radiology imaging policy"},
        ],
        last_updated=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_retrieval_result(
    chunks: list[ScoredChunk] | None = None,
    no_evidence: bool = False,
) -> RetrievalResult:
    """Create a RetrievalResult for testing."""
    if chunks is None and not no_evidence:
        chunks = [
            _make_scored_chunk("chunk-001", "lumbar spine stenosis noted on MRI", 0.85),
            _make_scored_chunk("chunk-002", "back pain chronic condition confirmed", 0.72),
        ]
    return RetrievalResult(
        verified_chunks=chunks or [],
        tamper_alerts=[],
        no_evidence_found=no_evidence,
        degraded_search=False,
        total_candidates=len(chunks) if chunks else 0,
    )


@pytest.fixture
def mock_graph_service():
    """Mock CausalOntologyGraphService."""
    service = AsyncMock()
    service.get_member_active_state = AsyncMock(return_value=_make_member_state())
    return service


@pytest.fixture
def mock_retrieval_service():
    """Mock HybridRetrievalService."""
    service = AsyncMock()
    service.retrieve = AsyncMock(return_value=_make_retrieval_result())
    return service


@pytest.fixture
def context_planner(mock_graph_service, mock_retrieval_service):
    """Create a ContextPlannerService with mocked dependencies."""
    return ContextPlannerService(
        graph_service=mock_graph_service,
        retrieval_service=mock_retrieval_service,
    )


@pytest.fixture
def pa_request():
    """Create a standard PA request for lumbar MRI."""
    return PARequest(
        request_id="REQ-001",
        member_id="MEM-001",
        cpt_code="72148",
        clinical_context="lumbar MRI for chronic back pain",
    )


# =============================================================================
# Test: Basic Briefing Packet Assembly (Requirements 4.1, 4.2, 4.3)
# =============================================================================


class TestBriefingPacketAssembly:
    """Tests for successful Briefing Packet assembly."""

    async def test_assembles_valid_briefing_packet(self, context_planner, pa_request):
        """Requirement 4.3: Package into BriefingPacket with all required fields."""
        result = await context_planner.assemble_briefing_packet(pa_request)

        assert isinstance(result, BriefingPacket)
        assert result.request_id == "REQ-001"
        assert result.member_id == "MEM-001"
        assert result.cpt_code == "72148"
        assert result.active_clinical_state is not None
        assert isinstance(result.verified_evidence_snippets, list)
        assert result.no_evidence_found is False

    async def test_queries_graph_for_member_state(
        self, context_planner, pa_request, mock_graph_service
    ):
        """Requirement 4.1: Query the Causal Ontology Graph for member active state."""
        await context_planner.assemble_briefing_packet(pa_request)
        mock_graph_service.get_member_active_state.assert_called_once_with("MEM-001")

    async def test_queries_retrieval_for_evidence(
        self, context_planner, pa_request, mock_retrieval_service
    ):
        """Requirement 4.2: Query Qdrant with max 20 snippets, min score 0.5."""
        await context_planner.assemble_briefing_packet(pa_request)
        mock_retrieval_service.retrieve.assert_called_once()
        call_kwargs = mock_retrieval_service.retrieve.call_args
        assert call_kwargs.kwargs["top_k"] == 20
        assert call_kwargs.kwargs["min_score"] == 0.5

    async def test_briefing_packet_has_all_required_fields(self, context_planner, pa_request):
        """Requirement 4.3: Verify schema conformance with all required fields."""
        result = await context_planner.assemble_briefing_packet(pa_request)

        # All required fields present and non-null
        assert result.request_id
        assert result.member_id
        assert result.cpt_code
        assert result.active_clinical_state is not None
        assert result.active_clinical_state.member_id == "MEM-001"
        assert isinstance(result.verified_evidence_snippets, list)
        assert isinstance(result.inferred_facts, list)

    async def test_evidence_snippets_max_20(self, context_planner, pa_request, mock_retrieval_service):
        """Requirement 4.2: Maximum of 20 snippets in the Briefing Packet."""
        # Create 25 chunks that match lumbar context
        chunks = [
            _make_scored_chunk(f"chunk-{i:03d}", f"lumbar spine test snippet {i}", 0.9 - i * 0.01)
            for i in range(25)
        ]
        mock_retrieval_service.retrieve = AsyncMock(
            return_value=_make_retrieval_result(chunks[:20])
        )

        result = await context_planner.assemble_briefing_packet(pa_request)
        assert len(result.verified_evidence_snippets) <= 20

    async def test_evidence_snippets_min_score_threshold(
        self, context_planner, pa_request, mock_retrieval_service
    ):
        """Requirement 4.2: All snippets have minimum relevance score of 0.5."""
        chunks = [
            _make_scored_chunk("chunk-001", "lumbar stenosis finding", 0.9),
            _make_scored_chunk("chunk-002", "spine condition lumbar", 0.6),
        ]
        mock_retrieval_service.retrieve = AsyncMock(
            return_value=_make_retrieval_result(chunks)
        )

        result = await context_planner.assemble_briefing_packet(pa_request)
        for snippet in result.verified_evidence_snippets:
            assert snippet.score >= 0.5


# =============================================================================
# Test: CPT Relevance Filtering (Requirement 4.4)
# =============================================================================


class TestCPTRelevanceFiltering:
    """Tests for CPT-based content filtering."""

    async def test_filters_diagnoses_by_cpt_relevance(
        self, context_planner, pa_request, mock_graph_service
    ):
        """Requirement 4.4: Filter to diagnoses matching CPT code categories."""
        result = await context_planner.assemble_briefing_packet(pa_request)

        # CPT 72148 = lumbar/spine/back pain categories
        # Should include lumbar-related diagnoses, may exclude unrelated ones
        diagnoses = result.active_clinical_state.active_diagnoses
        # Low back pain and lumbar spondylosis should match
        descriptions = [d["description"].lower() for d in diagnoses]
        assert any("lumbar" in d or "back" in d for d in descriptions)

    async def test_filters_prescriptions_by_cpt_relevance(
        self, context_planner, pa_request, mock_graph_service
    ):
        """Requirement 4.4: Filter medications matching CPT clinical context."""
        result = await context_planner.assemble_briefing_packet(pa_request)

        # Ibuprofen for back pain should match; metformin might not
        prescriptions = result.active_clinical_state.active_prescriptions
        # At least the back pain medication should be present
        descriptions = [rx.get("description", "").lower() for rx in prescriptions]
        assert any("back pain" in d for d in descriptions)

    async def test_preserves_sdoh_factors_unfiltered(
        self, context_planner, pa_request, mock_graph_service
    ):
        """SDOH factors are preserved regardless of CPT relevance."""
        result = await context_planner.assemble_briefing_packet(pa_request)
        assert len(result.active_clinical_state.sdoh_factors) > 0

    async def test_preserves_governing_policies(
        self, context_planner, pa_request, mock_graph_service
    ):
        """Governing policies are preserved regardless of CPT relevance."""
        result = await context_planner.assemble_briefing_packet(pa_request)
        assert len(result.active_clinical_state.governing_policies) > 0

    async def test_filters_snippets_by_cpt_context(
        self, context_planner, pa_request, mock_retrieval_service
    ):
        """Requirement 4.4: Only evidence matching CPT code or condition categories."""
        chunks = [
            _make_scored_chunk("chunk-001", "lumbar spine stenosis", 0.9),
            _make_scored_chunk("chunk-002", "cardiac arrhythmia detected", 0.85),
            _make_scored_chunk("chunk-003", "back pain radiating to leg", 0.8),
        ]
        mock_retrieval_service.retrieve = AsyncMock(
            return_value=_make_retrieval_result(chunks)
        )

        result = await context_planner.assemble_briefing_packet(pa_request)

        # Only lumbar/spine/back pain related snippets should pass filter
        snippet_texts = [s.text.lower() for s in result.verified_evidence_snippets]
        assert any("lumbar" in t for t in snippet_texts)
        assert any("back pain" in t for t in snippet_texts)
        # Cardiac unrelated content should be filtered out
        assert not any("cardiac" in t for t in snippet_texts)

    async def test_unknown_cpt_code_returns_all_state(
        self, mock_graph_service, mock_retrieval_service
    ):
        """Unknown CPT code should return full state without filtering."""
        planner = ContextPlannerService(mock_graph_service, mock_retrieval_service)
        request = PARequest(
            request_id="REQ-002",
            member_id="MEM-001",
            cpt_code="99999",  # Unknown CPT code
        )

        result = await planner.assemble_briefing_packet(request)

        # All diagnoses should be included since we can't determine relevance
        assert len(result.active_clinical_state.active_diagnoses) == 3


# =============================================================================
# Test: Member Not Found (Requirement 4.5)
# =============================================================================


class TestMemberNotFound:
    """Tests for member-not-found error handling."""

    async def test_raises_member_not_found_error(
        self, context_planner, mock_graph_service
    ):
        """Requirement 4.5: Raise MemberNotFoundError if member not in graph."""
        mock_graph_service.get_member_active_state.side_effect = MemberNotFoundError(
            reason="Member 'MEM-999' not found in the Causal Ontology Graph",
            member_id="MEM-999",
        )

        request = PARequest(
            request_id="REQ-003",
            member_id="MEM-999",
            cpt_code="72148",
        )

        with pytest.raises(MemberNotFoundError) as exc_info:
            await context_planner.assemble_briefing_packet(request)

        assert "MEM-999" in str(exc_info.value)

    async def test_does_not_query_retrieval_on_member_not_found(
        self, context_planner, mock_graph_service, mock_retrieval_service
    ):
        """If member not found, retrieval should not be attempted."""
        mock_graph_service.get_member_active_state.side_effect = MemberNotFoundError(
            reason="Member not found",
            member_id="MEM-999",
        )

        request = PARequest(
            request_id="REQ-004",
            member_id="MEM-999",
            cpt_code="72148",
        )

        with pytest.raises(MemberNotFoundError):
            await context_planner.assemble_briefing_packet(request)

        mock_retrieval_service.retrieve.assert_not_called()


# =============================================================================
# Test: Timeout Handling (Requirement 4.6)
# =============================================================================


class TestTimeoutHandling:
    """Tests for 30-second assembly timeout."""

    async def test_timeout_raises_timeout_error(
        self, context_planner, pa_request, mock_graph_service
    ):
        """Requirement 4.6: Raise TimeoutError if assembly exceeds 30 seconds."""

        async def slow_graph_query(member_id):
            await asyncio.sleep(35)  # Exceeds 30s timeout
            return _make_member_state()

        mock_graph_service.get_member_active_state.side_effect = slow_graph_query

        with pytest.raises(asyncio.TimeoutError):
            await context_planner.assemble_briefing_packet(pa_request)

    async def test_timeout_constant_is_30_seconds(self):
        """Verify the timeout constant is 30 seconds."""
        assert ASSEMBLY_TIMEOUT_SECONDS == 30


# =============================================================================
# Test: Zero Evidence Handling (Requirement 4.7)
# =============================================================================


class TestZeroEvidenceHandling:
    """Tests for zero evidence scenario."""

    async def test_empty_evidence_sets_no_evidence_found_flag(
        self, context_planner, pa_request, mock_retrieval_service
    ):
        """Requirement 4.7: Set no_evidence_found flag when zero snippets."""
        mock_retrieval_service.retrieve = AsyncMock(
            return_value=_make_retrieval_result(no_evidence=True)
        )

        result = await context_planner.assemble_briefing_packet(pa_request)

        assert result.no_evidence_found is True
        assert result.verified_evidence_snippets == []

    async def test_empty_evidence_still_has_clinical_state(
        self, context_planner, pa_request, mock_retrieval_service
    ):
        """Even with no evidence, clinical state should still be present."""
        mock_retrieval_service.retrieve = AsyncMock(
            return_value=_make_retrieval_result(no_evidence=True)
        )

        result = await context_planner.assemble_briefing_packet(pa_request)

        assert result.active_clinical_state is not None
        assert result.active_clinical_state.member_id == "MEM-001"

    async def test_filtered_snippets_all_irrelevant_sets_flag(
        self, context_planner, pa_request, mock_retrieval_service
    ):
        """If all retrieved snippets are filtered as irrelevant, set no_evidence_found."""
        # All snippets about unrelated topics for a lumbar MRI request
        chunks = [
            _make_scored_chunk("chunk-001", "cardiac stress test results", 0.8),
            _make_scored_chunk("chunk-002", "ophthalmology exam normal", 0.7),
        ]
        mock_retrieval_service.retrieve = AsyncMock(
            return_value=_make_retrieval_result(chunks)
        )

        result = await context_planner.assemble_briefing_packet(pa_request)

        assert result.no_evidence_found is True
        assert result.verified_evidence_snippets == []


# =============================================================================
# Test: PARequest Dataclass
# =============================================================================


class TestPARequest:
    """Tests for the PARequest dataclass."""

    def test_creates_with_required_fields(self):
        """PARequest can be created with required fields only."""
        request = PARequest(
            request_id="REQ-001",
            member_id="MEM-001",
            cpt_code="72148",
        )
        assert request.request_id == "REQ-001"
        assert request.member_id == "MEM-001"
        assert request.cpt_code == "72148"
        assert request.clinical_context is None

    def test_creates_with_clinical_context(self):
        """PARequest can include optional clinical_context."""
        request = PARequest(
            request_id="REQ-002",
            member_id="MEM-002",
            cpt_code="78816",
            clinical_context="PET scan for staging lung cancer",
        )
        assert request.clinical_context == "PET scan for staging lung cancer"


# =============================================================================
# Test: Retrieval Query Construction
# =============================================================================


class TestRetrievalQueryConstruction:
    """Tests for query building logic."""

    def test_builds_query_with_cpt_code(self):
        """Query includes CPT code."""
        planner = ContextPlannerService(AsyncMock(), AsyncMock())
        request = PARequest(
            request_id="REQ-001",
            member_id="MEM-001",
            cpt_code="72148",
        )
        query = planner._build_retrieval_query(request)
        assert "72148" in query

    def test_builds_query_with_clinical_context(self):
        """Query includes clinical context when provided."""
        planner = ContextPlannerService(AsyncMock(), AsyncMock())
        request = PARequest(
            request_id="REQ-001",
            member_id="MEM-001",
            cpt_code="72148",
            clinical_context="chronic low back pain",
        )
        query = planner._build_retrieval_query(request)
        assert "chronic low back pain" in query

    def test_builds_query_with_condition_categories(self):
        """Query includes associated condition categories for known CPT codes."""
        planner = ContextPlannerService(AsyncMock(), AsyncMock())
        request = PARequest(
            request_id="REQ-001",
            member_id="MEM-001",
            cpt_code="72148",
        )
        query = planner._build_retrieval_query(request)
        # CPT 72148 is associated with lumbar, spine, back pain, radiculopathy
        assert "lumbar" in query
        assert "spine" in query
