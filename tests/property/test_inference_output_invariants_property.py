"""Property-based tests for Inference Output Invariants.

**Validates: Requirements 14.2, 14.6**

Property 27: Inference Output Invariants
- For any inferred fact: inference_type is one of {sdoh_factor, medication_adherence_risk,
  care_access_barrier}, confidence is in [0.0, 1.0], if sdoh_factor then sdoh_category
  is from valid set, max 10 facts per snippet.
"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    ClinicalInferenceEngine,
    InferenceResult,
    InferredFact,
    InferenceChain,
    InferenceHop,
)
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    ScoredChunk,
)


# =============================================================================
# Constants from the engine
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
# Mock LLM Client
# =============================================================================


class MockLLMClient:
    """Mock LLM client that generates random but valid-structured responses.

    Produces JSON arrays of inferred facts with valid inference types,
    SDOH categories, confidence scores, and hop structures.
    """

    def __init__(self, num_facts: int = 3, seed: int = 42):
        """Initialize the mock with configurable number of facts to generate.

        Args:
            num_facts: Number of facts to generate per call (capped at 12
                to test the max-10 filtering behavior).
            seed: Random seed for reproducibility.
        """
        self.num_facts = num_facts
        self.rng = random.Random(seed)

    async def infer(self, prompt: str) -> str:
        """Generate a random but valid-structured LLM response."""
        facts = []
        for _ in range(self.num_facts):
            inference_type = self.rng.choice(list(VALID_INFERENCE_TYPES))
            fact = {
                "inference_type": inference_type,
                "conclusion": f"Inferred conclusion: {uuid.uuid4().hex[:8]}",
                "confidence": round(self.rng.uniform(0.0, 1.0), 3),
                "hops": self._generate_hops(),
            }
            if inference_type == "sdoh_factor":
                fact["sdoh_category"] = self.rng.choice(list(VALID_SDOH_CATEGORIES))
            facts.append(fact)
        return json.dumps(facts)

    def _generate_hops(self) -> list[dict]:
        """Generate 1-3 reasoning hops."""
        num_hops = self.rng.randint(1, 3)
        hops = []
        for i in range(num_hops):
            hops.append({
                "source_text": f"Source text for hop {i + 1}: {uuid.uuid4().hex[:12]}",
                "intermediate_conclusion": f"Intermediate conclusion {i + 1}",
                "confidence": round(self.rng.uniform(0.1, 1.0), 3),
            })
        return hops


class MockGraphService:
    """Mock graph service that does nothing."""

    async def upsert_node(self, node_type, node_id, properties, execution_id=None):
        pass

    async def upsert_relationship(self, source_id, target_id, rel_type, properties):
        pass


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


def _make_scored_chunk(chunk_id: str, text: str) -> ScoredChunk:
    """Create a valid ScoredChunk for testing."""
    import hashlib

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.75,
        provenance=ChunkProvenance(
            document_id="doc-001",
            content_hash=content_hash,
            kms_signature=_make_kms_signature(),
            chunk_index=0,
            ingestion_timestamp=datetime.now(timezone.utc),
        ),
        dense_rank=None,
        sparse_rank=None,
    )


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for number of facts the mock LLM generates (0 to 15 to stress test max-10 cap)
num_facts_strategy = st.integers(min_value=0, max_value=15)

# Strategy for confidence thresholds
confidence_threshold_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for inference depth
depth_strategy = st.sampled_from(["shallow", "deep"])

# Strategy for clinical note text
clinical_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=10,
    max_size=500,
).filter(lambda s: s.strip() != "")

# Strategy for member IDs
member_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")

# Random seed strategy for the mock LLM
seed_strategy = st.integers(min_value=0, max_value=10000)


@st.composite
def inference_scenario_strategy(draw):
    """Generate a complete inference analysis scenario.

    Returns a tuple of (num_facts, confidence_threshold, depth, clinical_text, member_id, seed).
    """
    num_facts = draw(num_facts_strategy)
    confidence_threshold = draw(confidence_threshold_strategy)
    depth = draw(depth_strategy)
    clinical_text = draw(clinical_text_strategy)
    member_id = draw(member_id_strategy)
    seed = draw(seed_strategy)
    return num_facts, confidence_threshold, depth, clinical_text, member_id, seed


# =============================================================================
# Property 27: Inference Output Invariants
# =============================================================================


@pytest.mark.property
class TestInferenceOutputInvariants:
    """Property 27: Inference Output Invariants.

    **Validates: Requirements 14.2, 14.6**

    Tests that for any inferred fact produced by the Clinical Inference Engine:
    - inference_type is one of {sdoh_factor, medication_adherence_risk, care_access_barrier}
    - confidence is in [0.0, 1.0]
    - if inference_type == sdoh_factor then sdoh_category is from valid set
    - max 10 facts per snippet in InferenceResult
    """

    @given(scenario=inference_scenario_strategy())
    @settings(max_examples=200, deadline=15000)
    @pytest.mark.asyncio
    async def test_inference_type_is_valid(self, scenario):
        """Every inferred fact has a valid inference_type.

        **Validates: Requirements 14.2**

        inference_type must be one of {sdoh_factor, medication_adherence_risk,
        care_access_barrier} for every fact in the result.
        """
        num_facts, confidence_threshold, depth, clinical_text, member_id, seed = scenario

        mock_llm = MockLLMClient(num_facts=num_facts, seed=seed)
        mock_graph = MockGraphService()

        engine = ClinicalInferenceEngine(
            llm_client=mock_llm,
            graph_service=mock_graph,
            confidence_threshold=confidence_threshold,
            inference_depth=depth,
        )

        snippet = _make_scored_chunk(f"chunk-{seed}", clinical_text)
        result = await engine.analyze_snippet(snippet, member_id)

        for fact in result.inferred_facts:
            assert fact.inference_type in VALID_INFERENCE_TYPES, (
                f"inference_type '{fact.inference_type}' is not in valid set "
                f"{VALID_INFERENCE_TYPES}"
            )

    @given(scenario=inference_scenario_strategy())
    @settings(max_examples=200, deadline=15000)
    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self, scenario):
        """Every inferred fact has confidence in [0.0, 1.0].

        **Validates: Requirements 14.2**

        Confidence scores must be bounded between 0.0 and 1.0 inclusive
        for every fact produced by the engine.
        """
        num_facts, confidence_threshold, depth, clinical_text, member_id, seed = scenario

        mock_llm = MockLLMClient(num_facts=num_facts, seed=seed)
        mock_graph = MockGraphService()

        engine = ClinicalInferenceEngine(
            llm_client=mock_llm,
            graph_service=mock_graph,
            confidence_threshold=confidence_threshold,
            inference_depth=depth,
        )

        snippet = _make_scored_chunk(f"chunk-{seed}", clinical_text)
        result = await engine.analyze_snippet(snippet, member_id)

        for fact in result.inferred_facts:
            assert 0.0 <= fact.confidence <= 1.0, (
                f"Confidence {fact.confidence} is outside valid range [0.0, 1.0] "
                f"for fact '{fact.fact_id}'"
            )

    @given(scenario=inference_scenario_strategy())
    @settings(max_examples=200, deadline=15000)
    @pytest.mark.asyncio
    async def test_sdoh_category_valid_when_sdoh_factor(self, scenario):
        """If inference_type is sdoh_factor, sdoh_category is from valid set.

        **Validates: Requirements 14.2, 14.6**

        When a fact has inference_type == 'sdoh_factor', its sdoh_category must
        be one of {housing_instability, transportation_barriers,
        medication_storage_limitations, food_insecurity, caregiver_availability}.
        """
        num_facts, confidence_threshold, depth, clinical_text, member_id, seed = scenario

        mock_llm = MockLLMClient(num_facts=num_facts, seed=seed)
        mock_graph = MockGraphService()

        engine = ClinicalInferenceEngine(
            llm_client=mock_llm,
            graph_service=mock_graph,
            confidence_threshold=confidence_threshold,
            inference_depth=depth,
        )

        snippet = _make_scored_chunk(f"chunk-{seed}", clinical_text)
        result = await engine.analyze_snippet(snippet, member_id)

        for fact in result.inferred_facts:
            if fact.inference_type == "sdoh_factor":
                assert fact.sdoh_category is not None, (
                    f"sdoh_category must not be None for sdoh_factor fact '{fact.fact_id}'"
                )
                assert fact.sdoh_category in VALID_SDOH_CATEGORIES, (
                    f"sdoh_category '{fact.sdoh_category}' is not in valid set "
                    f"{VALID_SDOH_CATEGORIES} for fact '{fact.fact_id}'"
                )

    @given(scenario=inference_scenario_strategy())
    @settings(max_examples=200, deadline=15000)
    @pytest.mark.asyncio
    async def test_max_10_facts_per_snippet(self, scenario):
        """InferenceResult contains at most 10 inferred facts.

        **Validates: Requirements 14.6**

        The engine must enforce the maximum of 10 inferred facts per snippet,
        even when the LLM produces more than 10 candidate facts.
        """
        num_facts, confidence_threshold, depth, clinical_text, member_id, seed = scenario

        mock_llm = MockLLMClient(num_facts=num_facts, seed=seed)
        mock_graph = MockGraphService()

        engine = ClinicalInferenceEngine(
            llm_client=mock_llm,
            graph_service=mock_graph,
            confidence_threshold=confidence_threshold,
            inference_depth=depth,
        )

        snippet = _make_scored_chunk(f"chunk-{seed}", clinical_text)
        result = await engine.analyze_snippet(snippet, member_id)

        assert len(result.inferred_facts) <= 10, (
            f"InferenceResult contains {len(result.inferred_facts)} facts, "
            f"exceeding the maximum of 10 per snippet"
        )

    @given(scenario=inference_scenario_strategy())
    @settings(max_examples=200, deadline=15000)
    @pytest.mark.asyncio
    async def test_non_sdoh_facts_have_no_sdoh_category(self, scenario):
        """Non-sdoh_factor facts have sdoh_category set to None.

        **Validates: Requirements 14.2**

        Facts with inference_type of medication_adherence_risk or
        care_access_barrier should not have an sdoh_category.
        """
        num_facts, confidence_threshold, depth, clinical_text, member_id, seed = scenario

        mock_llm = MockLLMClient(num_facts=num_facts, seed=seed)
        mock_graph = MockGraphService()

        engine = ClinicalInferenceEngine(
            llm_client=mock_llm,
            graph_service=mock_graph,
            confidence_threshold=confidence_threshold,
            inference_depth=depth,
        )

        snippet = _make_scored_chunk(f"chunk-{seed}", clinical_text)
        result = await engine.analyze_snippet(snippet, member_id)

        for fact in result.inferred_facts:
            if fact.inference_type != "sdoh_factor":
                assert fact.sdoh_category is None, (
                    f"Non-sdoh_factor fact '{fact.fact_id}' (type={fact.inference_type}) "
                    f"should have sdoh_category=None, got '{fact.sdoh_category}'"
                )

    @given(scenario=inference_scenario_strategy())
    @settings(max_examples=200, deadline=15000)
    @pytest.mark.asyncio
    async def test_result_is_valid_inference_result(self, scenario):
        """analyze_snippet returns a valid InferenceResult with all required fields.

        **Validates: Requirements 14.2, 14.6**

        The InferenceResult must have a non-empty snippet_id, valid depth_used,
        non-negative processing_time_ms, and non-negative total_hops_executed.
        """
        num_facts, confidence_threshold, depth, clinical_text, member_id, seed = scenario

        mock_llm = MockLLMClient(num_facts=num_facts, seed=seed)
        mock_graph = MockGraphService()

        engine = ClinicalInferenceEngine(
            llm_client=mock_llm,
            graph_service=mock_graph,
            confidence_threshold=confidence_threshold,
            inference_depth=depth,
        )

        snippet = _make_scored_chunk(f"chunk-{seed}", clinical_text)
        result = await engine.analyze_snippet(snippet, member_id)

        assert isinstance(result, InferenceResult), (
            f"Expected InferenceResult, got {type(result)}"
        )
        assert result.snippet_id is not None and len(result.snippet_id) > 0, (
            "snippet_id must be non-null and non-empty"
        )
        assert result.depth_used in ("shallow", "deep"), (
            f"depth_used must be 'shallow' or 'deep', got '{result.depth_used}'"
        )
        assert result.processing_time_ms >= 0, (
            f"processing_time_ms must be non-negative, got {result.processing_time_ms}"
        )
        assert result.total_hops_executed >= 0, (
            f"total_hops_executed must be non-negative, got {result.total_hops_executed}"
        )

    @given(scenario=inference_scenario_strategy())
    @settings(max_examples=200, deadline=15000)
    @pytest.mark.asyncio
    async def test_all_facts_meet_confidence_threshold(self, scenario):
        """All returned facts have confidence >= the configured threshold.

        **Validates: Requirements 14.2**

        The engine's threshold filtering ensures no fact below the
        configured confidence_threshold appears in the output.
        """
        num_facts, confidence_threshold, depth, clinical_text, member_id, seed = scenario

        mock_llm = MockLLMClient(num_facts=num_facts, seed=seed)
        mock_graph = MockGraphService()

        engine = ClinicalInferenceEngine(
            llm_client=mock_llm,
            graph_service=mock_graph,
            confidence_threshold=confidence_threshold,
            inference_depth=depth,
        )

        snippet = _make_scored_chunk(f"chunk-{seed}", clinical_text)
        result = await engine.analyze_snippet(snippet, member_id)

        for fact in result.inferred_facts:
            assert fact.confidence >= confidence_threshold, (
                f"Fact '{fact.fact_id}' has confidence {fact.confidence} "
                f"below threshold {confidence_threshold}"
            )
