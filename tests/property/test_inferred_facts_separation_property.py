"""Property-based tests for Inferred Facts Separation in Briefing Packet.

**Validates: Requirements 14.8**

Property 30: Inferred Facts Separation in Briefing Packet
- Inferred facts in the BriefingPacket are structurally distinct from explicit
  evidence snippets (dicts, not ScoredChunk objects).
- Each inferred fact includes inference_type, confidence score, and source
  inference chain.
- Inferred facts are in a separate section from verified_evidence_snippets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.context_planner_service import (
    ContextPlannerService,
    PARequest,
)
from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    ClinicalInferenceEngine,
    InferenceResult,
    InferredFact,
    InferenceChain,
    InferenceHop,
)
from clinical_reasoning_fabric.models.core import (
    BriefingPacket,
    ChunkProvenance,
    KMSSignature,
    MemberActiveState,
    RetrievalResult,
    ScoredChunk,
)


# =============================================================================
# Constants
# =============================================================================

VALID_INFERENCE_TYPES = {"sdoh_factor", "medication_adherence_risk", "care_access_barrier"}
VALID_SDOH_CATEGORIES = {
    "housing_instability",
    "transportation_barriers",
    "medication_storage_limitations",
    "food_insecurity",
    "caregiver_availability",
}


# =============================================================================
# Mock Inference Engine
# =============================================================================


class MockInferenceEngine:
    """Mock Clinical Inference Engine that produces configurable inferred facts.

    Generates valid InferenceResult objects with the specified number of facts
    for each snippet analyzed.
    """

    def __init__(self, num_facts_per_snippet: int = 3, seed: int = 42):
        self.num_facts_per_snippet = num_facts_per_snippet
        self.rng = random.Random(seed)

    async def analyze_snippet(
        self, snippet: ScoredChunk, member_id: str
    ) -> InferenceResult:
        """Generate a valid InferenceResult for the given snippet."""
        facts = []
        num_facts = min(self.num_facts_per_snippet, 10)

        for i in range(num_facts):
            inference_type = self.rng.choice(list(VALID_INFERENCE_TYPES))
            sdoh_category = None
            if inference_type == "sdoh_factor":
                sdoh_category = self.rng.choice(list(VALID_SDOH_CATEGORIES))

            confidence = round(self.rng.uniform(0.3, 1.0), 3)

            hop = InferenceHop(
                hop_number=1,
                source_text=f"Source text from snippet {snippet.chunk_id}",
                intermediate_conclusion=f"Conclusion {i}",
                confidence=confidence,
            )

            chain = InferenceChain(
                chain_id=f"chain-{uuid.uuid4().hex[:8]}",
                hops=[hop],
                cumulative_confidence=confidence,
                source_snippet_id=snippet.chunk_id,
                final_conclusion=f"Final conclusion {i} from {snippet.chunk_id}",
            )

            fact = InferredFact(
                fact_id=f"fact-{uuid.uuid4().hex[:8]}",
                inference_type=inference_type,
                sdoh_category=sdoh_category,
                conclusion=f"Inferred conclusion {i}",
                confidence=confidence,
                inference_chain=chain,
                source_text_excerpt=f"Excerpt from snippet {snippet.chunk_id}",
                inferred_at=datetime.now(timezone.utc),
            )
            facts.append(fact)

        return InferenceResult(
            snippet_id=snippet.chunk_id,
            inferred_facts=facts,
            processing_time_ms=50,
            depth_used="shallow",
            total_hops_executed=num_facts,
        )


# =============================================================================
# Mock Services
# =============================================================================


class MockGraphService:
    """Mock graph service that returns configurable member state."""

    def __init__(self, member_state: MemberActiveState):
        self.member_state = member_state

    async def get_member_active_state(self, member_id: str) -> MemberActiveState:
        return self.member_state


class MockRetrievalService:
    """Mock retrieval service that returns configurable evidence snippets."""

    def __init__(self, snippets: list[ScoredChunk]):
        self.snippets = snippets

    async def retrieve(
        self, query: str, top_k: int = 20, min_score: float = 0.5
    ) -> RetrievalResult:
        return RetrievalResult(
            verified_chunks=self.snippets,
            tamper_alerts=[],
            no_evidence_found=len(self.snippets) == 0,
            degraded_search=False,
            total_candidates=len(self.snippets),
        )


# =============================================================================
# Helpers
# =============================================================================


def _make_kms_signature() -> KMSSignature:
    """Create a valid KMSSignature for testing."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789012:key/test-key-id",
        signature="dGVzdC1zaWduYXR1cmUtYmFzZTY0",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_scored_chunk(chunk_id: str, text: str, score: float = 0.75) -> ScoredChunk:
    """Create a valid ScoredChunk for testing."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        provenance=ChunkProvenance(
            document_id=f"doc-{chunk_id}",
            content_hash=content_hash,
            kms_signature=_make_kms_signature(),
            chunk_index=0,
            ingestion_timestamp=datetime.now(timezone.utc),
        ),
        dense_rank=None,
        sparse_rank=None,
    )


def _make_member_state(member_id: str) -> MemberActiveState:
    """Create a valid MemberActiveState for testing."""
    return MemberActiveState(
        member_id=member_id,
        active_diagnoses=[
            {"condition_code": "M54.5", "description": "low back pain lumbar"}
        ],
        active_prescriptions=[
            {"medication_name": "Ibuprofen", "description": "pain management lumbar"}
        ],
        sdoh_factors=[],
        governing_policies=[],
        last_updated=datetime.now(timezone.utc),
    )


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Number of evidence snippets to include in the Briefing Packet (0 to 5 for speed)
num_snippets_strategy = st.integers(min_value=0, max_value=5)

# Number of inferred facts per snippet from mock engine (0 to 10)
num_facts_strategy = st.integers(min_value=0, max_value=10)

# Random seed for mock engine
seed_strategy = st.integers(min_value=0, max_value=10000)

# Member ID strategy
member_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")

# Request ID strategy
request_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")


@st.composite
def briefing_packet_scenario_strategy(draw):
    """Generate a complete Briefing Packet assembly scenario.

    Returns a tuple of (num_snippets, num_facts_per_snippet, seed, member_id, request_id).
    """
    num_snippets = draw(num_snippets_strategy)
    num_facts = draw(num_facts_strategy)
    seed = draw(seed_strategy)
    member_id = draw(member_id_strategy)
    request_id = draw(request_id_strategy)
    return num_snippets, num_facts, seed, member_id, request_id


# =============================================================================
# Property 30: Inferred Facts Separation in Briefing Packet
# =============================================================================


@pytest.mark.property
class TestInferredFactsSeparation:
    """Property 30: Inferred Facts Separation in Briefing Packet.

    **Validates: Requirements 14.8**

    Tests that inferred facts in the BriefingPacket are structurally distinct
    from explicit evidence snippets, each includes inference_type, confidence
    score, and source inference chain, and they are in a separate section.
    """

    @given(scenario=briefing_packet_scenario_strategy())
    @settings(max_examples=200, deadline=30000)
    @pytest.mark.asyncio
    async def test_inferred_facts_are_dicts_not_scored_chunks(self, scenario):
        """Inferred facts are structurally distinct from explicit evidence snippets.

        **Validates: Requirements 14.8**

        Inferred facts in the BriefingPacket are dict objects, never ScoredChunk
        instances. This ensures structural separation between explicit evidence
        and inferred conclusions.
        """
        num_snippets, num_facts, seed, member_id, request_id = scenario

        # Create snippets with text relevant to the CPT code categories
        snippets = [
            _make_scored_chunk(
                f"chunk-{i}",
                f"Patient lumbar spine examination findings note {i}",
                score=0.8,
            )
            for i in range(num_snippets)
        ]

        member_state = _make_member_state(member_id)
        mock_graph = MockGraphService(member_state)
        mock_retrieval = MockRetrievalService(snippets)
        mock_inference = MockInferenceEngine(num_facts_per_snippet=num_facts, seed=seed)

        service = ContextPlannerService(
            graph_service=mock_graph,
            retrieval_service=mock_retrieval,
            inference_engine=mock_inference,
            confidence_threshold=0.3,
        )

        pa_request = PARequest(
            request_id=request_id,
            member_id=member_id,
            cpt_code="72148",  # lumbar MRI - matches our snippet text
        )

        packet = await service.assemble_briefing_packet(pa_request)

        # Every inferred fact must be a dict, NOT a ScoredChunk
        for fact in packet.inferred_facts:
            assert isinstance(fact, dict), (
                f"Inferred fact must be a dict, got {type(fact).__name__}. "
                f"Inferred facts must be structurally distinct from ScoredChunk evidence."
            )
            assert not isinstance(fact, ScoredChunk), (
                "Inferred fact must NOT be a ScoredChunk instance."
            )

    @given(scenario=briefing_packet_scenario_strategy())
    @settings(max_examples=200, deadline=30000)
    @pytest.mark.asyncio
    async def test_each_inferred_fact_has_inference_type(self, scenario):
        """Each inferred fact includes inference_type field.

        **Validates: Requirements 14.8**

        Every inferred fact dict in the BriefingPacket must contain an
        'inference_type' key with a valid value from the allowed set.
        """
        num_snippets, num_facts, seed, member_id, request_id = scenario
        assume(num_snippets > 0 and num_facts > 0)

        snippets = [
            _make_scored_chunk(
                f"chunk-{i}",
                f"Patient lumbar spine examination findings note {i}",
                score=0.8,
            )
            for i in range(num_snippets)
        ]

        member_state = _make_member_state(member_id)
        mock_graph = MockGraphService(member_state)
        mock_retrieval = MockRetrievalService(snippets)
        mock_inference = MockInferenceEngine(num_facts_per_snippet=num_facts, seed=seed)

        service = ContextPlannerService(
            graph_service=mock_graph,
            retrieval_service=mock_retrieval,
            inference_engine=mock_inference,
            confidence_threshold=0.3,
        )

        pa_request = PARequest(
            request_id=request_id,
            member_id=member_id,
            cpt_code="72148",
        )

        packet = await service.assemble_briefing_packet(pa_request)

        for fact in packet.inferred_facts:
            assert "inference_type" in fact, (
                f"Inferred fact missing 'inference_type' key. Keys: {list(fact.keys())}"
            )
            assert fact["inference_type"] in VALID_INFERENCE_TYPES, (
                f"inference_type '{fact['inference_type']}' not in valid set "
                f"{VALID_INFERENCE_TYPES}"
            )

    @given(scenario=briefing_packet_scenario_strategy())
    @settings(max_examples=200, deadline=30000)
    @pytest.mark.asyncio
    async def test_each_inferred_fact_has_confidence_score(self, scenario):
        """Each inferred fact includes a confidence score.

        **Validates: Requirements 14.8**

        Every inferred fact dict must contain a 'confidence' key with a
        numeric value between 0.0 and 1.0 inclusive.
        """
        num_snippets, num_facts, seed, member_id, request_id = scenario
        assume(num_snippets > 0 and num_facts > 0)

        snippets = [
            _make_scored_chunk(
                f"chunk-{i}",
                f"Patient lumbar spine examination findings note {i}",
                score=0.8,
            )
            for i in range(num_snippets)
        ]

        member_state = _make_member_state(member_id)
        mock_graph = MockGraphService(member_state)
        mock_retrieval = MockRetrievalService(snippets)
        mock_inference = MockInferenceEngine(num_facts_per_snippet=num_facts, seed=seed)

        service = ContextPlannerService(
            graph_service=mock_graph,
            retrieval_service=mock_retrieval,
            inference_engine=mock_inference,
            confidence_threshold=0.3,
        )

        pa_request = PARequest(
            request_id=request_id,
            member_id=member_id,
            cpt_code="72148",
        )

        packet = await service.assemble_briefing_packet(pa_request)

        for fact in packet.inferred_facts:
            assert "confidence" in fact, (
                f"Inferred fact missing 'confidence' key. Keys: {list(fact.keys())}"
            )
            assert isinstance(fact["confidence"], (int, float)), (
                f"confidence must be numeric, got {type(fact['confidence']).__name__}"
            )
            assert 0.0 <= fact["confidence"] <= 1.0, (
                f"confidence {fact['confidence']} outside valid range [0.0, 1.0]"
            )

    @given(scenario=briefing_packet_scenario_strategy())
    @settings(max_examples=200, deadline=30000)
    @pytest.mark.asyncio
    async def test_each_inferred_fact_has_inference_chain(self, scenario):
        """Each inferred fact includes a source inference chain.

        **Validates: Requirements 14.8**

        Every inferred fact dict must contain an 'inference_chain' key with
        a dict value containing chain_id, hops (non-empty list), and
        cumulative_confidence.
        """
        num_snippets, num_facts, seed, member_id, request_id = scenario
        assume(num_snippets > 0 and num_facts > 0)

        snippets = [
            _make_scored_chunk(
                f"chunk-{i}",
                f"Patient lumbar spine examination findings note {i}",
                score=0.8,
            )
            for i in range(num_snippets)
        ]

        member_state = _make_member_state(member_id)
        mock_graph = MockGraphService(member_state)
        mock_retrieval = MockRetrievalService(snippets)
        mock_inference = MockInferenceEngine(num_facts_per_snippet=num_facts, seed=seed)

        service = ContextPlannerService(
            graph_service=mock_graph,
            retrieval_service=mock_retrieval,
            inference_engine=mock_inference,
            confidence_threshold=0.3,
        )

        pa_request = PARequest(
            request_id=request_id,
            member_id=member_id,
            cpt_code="72148",
        )

        packet = await service.assemble_briefing_packet(pa_request)

        for fact in packet.inferred_facts:
            assert "inference_chain" in fact, (
                f"Inferred fact missing 'inference_chain' key. Keys: {list(fact.keys())}"
            )
            chain = fact["inference_chain"]
            assert isinstance(chain, dict), (
                f"inference_chain must be a dict, got {type(chain).__name__}"
            )
            assert "chain_id" in chain, (
                f"inference_chain missing 'chain_id'. Keys: {list(chain.keys())}"
            )
            assert "hops" in chain, (
                f"inference_chain missing 'hops'. Keys: {list(chain.keys())}"
            )
            assert isinstance(chain["hops"], list), (
                f"inference_chain.hops must be a list, got {type(chain['hops']).__name__}"
            )
            assert len(chain["hops"]) >= 1, (
                "inference_chain.hops must have at least 1 hop"
            )
            assert "cumulative_confidence" in chain, (
                f"inference_chain missing 'cumulative_confidence'. Keys: {list(chain.keys())}"
            )

    @given(scenario=briefing_packet_scenario_strategy())
    @settings(max_examples=200, deadline=30000)
    @pytest.mark.asyncio
    async def test_inferred_facts_separate_from_evidence_snippets(self, scenario):
        """Inferred facts and verified evidence snippets are in separate sections.

        **Validates: Requirements 14.8**

        No inferred fact dict appears in the verified_evidence_snippets list,
        and no ScoredChunk appears in the inferred_facts list. The two sections
        are structurally distinct with no overlap.
        """
        num_snippets, num_facts, seed, member_id, request_id = scenario

        snippets = [
            _make_scored_chunk(
                f"chunk-{i}",
                f"Patient lumbar spine examination findings note {i}",
                score=0.8,
            )
            for i in range(num_snippets)
        ]

        member_state = _make_member_state(member_id)
        mock_graph = MockGraphService(member_state)
        mock_retrieval = MockRetrievalService(snippets)
        mock_inference = MockInferenceEngine(num_facts_per_snippet=num_facts, seed=seed)

        service = ContextPlannerService(
            graph_service=mock_graph,
            retrieval_service=mock_retrieval,
            inference_engine=mock_inference,
            confidence_threshold=0.3,
        )

        pa_request = PARequest(
            request_id=request_id,
            member_id=member_id,
            cpt_code="72148",
        )

        packet = await service.assemble_briefing_packet(pa_request)

        # Verify verified_evidence_snippets contains only ScoredChunk instances
        for snippet in packet.verified_evidence_snippets:
            assert isinstance(snippet, ScoredChunk), (
                f"verified_evidence_snippets must contain ScoredChunk instances, "
                f"got {type(snippet).__name__}"
            )

        # Verify inferred_facts contains only dicts (never ScoredChunk)
        for fact in packet.inferred_facts:
            assert isinstance(fact, dict), (
                f"inferred_facts must contain dicts, got {type(fact).__name__}"
            )
            assert not isinstance(fact, ScoredChunk), (
                "inferred_facts must not contain ScoredChunk instances"
            )

        # Verify no overlap: collect identifiers from both sections
        snippet_ids = {s.chunk_id for s in packet.verified_evidence_snippets}
        fact_ids = {f.get("fact_id") for f in packet.inferred_facts if "fact_id" in f}

        # fact_ids should not overlap with snippet_ids
        overlap = snippet_ids & fact_ids
        assert len(overlap) == 0, (
            f"Overlap detected between snippet IDs and fact IDs: {overlap}. "
            f"Inferred facts must be in a separate section from evidence snippets."
        )

    @given(scenario=briefing_packet_scenario_strategy())
    @settings(max_examples=200, deadline=30000)
    @pytest.mark.asyncio
    async def test_inferred_facts_empty_when_no_snippets(self, scenario):
        """When no evidence snippets are available, inferred_facts is empty.

        **Validates: Requirements 14.8**

        If verified_evidence_snippets is empty (no evidence to analyze),
        inferred_facts should also be empty since there are no snippets
        to derive inferences from.
        """
        _, num_facts, seed, member_id, request_id = scenario

        # Force zero snippets
        member_state = _make_member_state(member_id)
        mock_graph = MockGraphService(member_state)
        mock_retrieval = MockRetrievalService([])  # No snippets
        mock_inference = MockInferenceEngine(num_facts_per_snippet=num_facts, seed=seed)

        service = ContextPlannerService(
            graph_service=mock_graph,
            retrieval_service=mock_retrieval,
            inference_engine=mock_inference,
            confidence_threshold=0.3,
        )

        pa_request = PARequest(
            request_id=request_id,
            member_id=member_id,
            cpt_code="72148",
        )

        packet = await service.assemble_briefing_packet(pa_request)

        # With no snippets to analyze, inferred_facts should be empty
        assert packet.inferred_facts == [], (
            f"Expected empty inferred_facts when no snippets available, "
            f"got {len(packet.inferred_facts)} facts"
        )

    @given(scenario=briefing_packet_scenario_strategy())
    @settings(max_examples=200, deadline=30000)
    @pytest.mark.asyncio
    async def test_inference_chain_hops_have_required_fields(self, scenario):
        """Each hop in an inference chain has required fields.

        **Validates: Requirements 14.8**

        Each hop within an inferred fact's inference_chain must have:
        hop_number, source_text, intermediate_conclusion, and confidence.
        """
        num_snippets, num_facts, seed, member_id, request_id = scenario
        assume(num_snippets > 0 and num_facts > 0)

        snippets = [
            _make_scored_chunk(
                f"chunk-{i}",
                f"Patient lumbar spine examination findings note {i}",
                score=0.8,
            )
            for i in range(num_snippets)
        ]

        member_state = _make_member_state(member_id)
        mock_graph = MockGraphService(member_state)
        mock_retrieval = MockRetrievalService(snippets)
        mock_inference = MockInferenceEngine(num_facts_per_snippet=num_facts, seed=seed)

        service = ContextPlannerService(
            graph_service=mock_graph,
            retrieval_service=mock_retrieval,
            inference_engine=mock_inference,
            confidence_threshold=0.3,
        )

        pa_request = PARequest(
            request_id=request_id,
            member_id=member_id,
            cpt_code="72148",
        )

        packet = await service.assemble_briefing_packet(pa_request)

        for fact in packet.inferred_facts:
            chain = fact.get("inference_chain", {})
            hops = chain.get("hops", [])
            for hop in hops:
                assert isinstance(hop, dict), (
                    f"Each hop must be a dict, got {type(hop).__name__}"
                )
                assert "hop_number" in hop, (
                    f"Hop missing 'hop_number'. Keys: {list(hop.keys())}"
                )
                assert "source_text" in hop, (
                    f"Hop missing 'source_text'. Keys: {list(hop.keys())}"
                )
                assert "intermediate_conclusion" in hop, (
                    f"Hop missing 'intermediate_conclusion'. Keys: {list(hop.keys())}"
                )
                assert "confidence" in hop, (
                    f"Hop missing 'confidence'. Keys: {list(hop.keys())}"
                )
                assert isinstance(hop["confidence"], (int, float)), (
                    f"Hop confidence must be numeric, got {type(hop['confidence']).__name__}"
                )
                assert 0.0 <= hop["confidence"] <= 1.0, (
                    f"Hop confidence {hop['confidence']} outside [0.0, 1.0]"
                )
