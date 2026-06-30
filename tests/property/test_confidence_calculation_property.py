"""Property-based tests for Confidence Calculation and Threshold Filtering.

**Validates: Requirements 14.5, 14.7**

Property 26: Inference Confidence Calculation
- For any chain with N hops (1≤N≤3), cumulative confidence equals product of hop scores.

Property 28: Inference Threshold Filtering
- Every fact in downstream output has confidence >= threshold; no fact below threshold appears.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    ClinicalInferenceEngine,
    InferenceChain,
    InferenceHop,
    InferredFact,
)
from clinical_reasoning_fabric.inference.inference_chain_builder import (
    InferenceChainBuilder,
    MAX_CHAIN_HOPS,
)


# =============================================================================
# Mock services for engine instantiation
# =============================================================================


class MockLLMClient:
    """Minimal mock LLM client for testing engine methods that don't call LLM."""

    async def infer(self, prompt: str) -> str:
        return "[]"


class MockGraphService:
    """Minimal mock graph service."""

    async def upsert_node(self, node_type, node_id, properties, execution_id=None):
        pass

    async def upsert_relationship(self, source_id, target_id, rel_type, properties):
        pass


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for individual hop confidence scores (0.0 to 1.0 inclusive)
hop_confidence_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for number of hops (1 to 3)
num_hops_strategy = st.integers(min_value=1, max_value=3)

# Strategy for confidence threshold (0.0 to 1.0)
threshold_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def inference_chain_strategy(draw):
    """Generate a valid InferenceChain with 1-3 hops and random confidences.

    Returns an InferenceChain whose cumulative_confidence is set to the product
    of individual hop confidences.
    """
    n_hops = draw(num_hops_strategy)
    hops = []
    cumulative = 1.0
    for i in range(n_hops):
        conf = draw(hop_confidence_strategy)
        cumulative *= conf
        hops.append(
            InferenceHop(
                hop_number=i + 1,
                source_text=f"Source text for hop {i + 1}",
                intermediate_conclusion=f"Conclusion at hop {i + 1}",
                confidence=conf,
            )
        )

    chain = InferenceChain(
        chain_id=f"chain-{uuid.uuid4().hex[:8]}",
        hops=hops,
        cumulative_confidence=cumulative,
        source_snippet_id=f"snippet-{uuid.uuid4().hex[:8]}",
        final_conclusion=hops[-1].intermediate_conclusion,
    )
    return chain


@st.composite
def inferred_fact_strategy(draw):
    """Generate a valid InferredFact with random confidence and inference chain."""
    confidence = draw(hop_confidence_strategy)
    inference_type = draw(st.sampled_from(list(ClinicalInferenceEngine.INFERENCE_TYPES)))

    sdoh_category = None
    if inference_type == "sdoh_factor":
        sdoh_category = draw(st.sampled_from(list(ClinicalInferenceEngine.SDOH_CATEGORIES)))

    # Build a minimal chain for the fact
    chain = InferenceChain(
        chain_id=f"chain-{uuid.uuid4().hex[:8]}",
        hops=[
            InferenceHop(
                hop_number=1,
                source_text="Source text evidence",
                intermediate_conclusion="Derived conclusion",
                confidence=confidence,
            )
        ],
        cumulative_confidence=confidence,
        source_snippet_id="snippet-001",
        final_conclusion="Derived conclusion",
    )

    fact = InferredFact(
        fact_id=f"fact-{uuid.uuid4().hex[:8]}",
        inference_type=inference_type,
        sdoh_category=sdoh_category,
        conclusion="Test inferred conclusion",
        confidence=confidence,
        inference_chain=chain,
        source_text_excerpt="Clinical note excerpt for testing",
        inferred_at=datetime.now(timezone.utc),
    )
    return fact


@st.composite
def facts_list_strategy(draw):
    """Generate a list of InferredFacts with mixed confidence scores."""
    num_facts = draw(st.integers(min_value=0, max_value=10))
    facts = [draw(inferred_fact_strategy()) for _ in range(num_facts)]
    return facts


# =============================================================================
# Property 26: Inference Confidence Calculation
# =============================================================================


@pytest.mark.property
class TestInferenceConfidenceCalculation:
    """Property 26: Inference Confidence Calculation.

    **Validates: Requirements 14.5**

    For any chain with N hops (1≤N≤3), cumulative confidence equals product
    of hop scores.
    """

    @given(chain=inference_chain_strategy())
    @settings(max_examples=200, deadline=10000)
    def test_cumulative_confidence_equals_product_of_hop_scores(self, chain):
        """Cumulative confidence is the product of all hop confidence scores.

        **Validates: Requirements 14.5**

        For any InferenceChain with 1-3 hops, compute_chain_confidence must
        return a value equal to hop_1.confidence * hop_2.confidence * ... * hop_n.confidence.
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
        )

        computed_confidence = engine.compute_chain_confidence(chain)

        # Compute expected product manually
        expected_confidence = 1.0
        for hop in chain.hops:
            expected_confidence *= hop.confidence

        assert abs(computed_confidence - expected_confidence) < 1e-10, (
            f"compute_chain_confidence returned {computed_confidence}, but expected "
            f"product of hop confidences {[h.confidence for h in chain.hops]} = "
            f"{expected_confidence}"
        )

    @given(chain=inference_chain_strategy())
    @settings(max_examples=200, deadline=10000)
    def test_cumulative_confidence_within_valid_range(self, chain):
        """Cumulative confidence is always within [0.0, 1.0].

        **Validates: Requirements 14.5**

        Since each hop confidence is in [0.0, 1.0], the product of any
        number of such values must also be in [0.0, 1.0].
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
        )

        computed_confidence = engine.compute_chain_confidence(chain)

        assert 0.0 <= computed_confidence <= 1.0, (
            f"Cumulative confidence {computed_confidence} is outside [0.0, 1.0] "
            f"for chain with hop confidences {[h.confidence for h in chain.hops]}"
        )

    @given(
        confidences=st.lists(
            hop_confidence_strategy,
            min_size=1,
            max_size=3,
        )
    )
    @settings(max_examples=200, deadline=10000)
    def test_confidence_is_monotonically_decreasing_with_more_hops(self, confidences):
        """Adding more hops with confidence < 1.0 decreases cumulative confidence.

        **Validates: Requirements 14.5**

        The product of N values in (0.0, 1.0) is less than or equal to
        the product of any subset of those values. More hops (with conf < 1.0)
        means lower cumulative confidence.
        """
        # Only test with strictly less-than-1.0 hops to show decrease
        assume(all(0.0 < c < 1.0 for c in confidences))
        assume(len(confidences) >= 2)

        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
        )

        # Build a chain with all hops
        hops = [
            InferenceHop(
                hop_number=i + 1,
                source_text=f"Source {i + 1}",
                intermediate_conclusion=f"Conclusion {i + 1}",
                confidence=c,
            )
            for i, c in enumerate(confidences)
        ]
        full_chain = InferenceChain(
            chain_id="chain-full",
            hops=hops,
            cumulative_confidence=0.0,  # placeholder
            source_snippet_id="snippet-001",
            final_conclusion="Final",
        )

        # Build a chain with fewer hops (first hop only)
        shorter_chain = InferenceChain(
            chain_id="chain-short",
            hops=[hops[0]],
            cumulative_confidence=0.0,  # placeholder
            source_snippet_id="snippet-001",
            final_conclusion="Final",
        )

        full_confidence = engine.compute_chain_confidence(full_chain)
        shorter_confidence = engine.compute_chain_confidence(shorter_chain)

        assert full_confidence <= shorter_confidence, (
            f"Full chain confidence ({full_confidence}) should be <= shorter chain "
            f"confidence ({shorter_confidence}) when all hop scores are in (0, 1)"
        )

    @given(
        hops_data=st.lists(
            st.fixed_dictionaries({
                "source_text": st.just("Source text"),
                "intermediate_conclusion": st.just("Conclusion"),
                "confidence": st.floats(
                    min_value=0.01, max_value=1.0,
                    allow_nan=False, allow_infinity=False
                ),
            }),
            min_size=1,
            max_size=3,
        ),
        source_snippet_id=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=200, deadline=10000)
    def test_builder_chain_confidence_matches_engine_computation(
        self, hops_data, source_snippet_id
    ):
        """InferenceChainBuilder cumulative_confidence matches engine's compute_chain_confidence.

        **Validates: Requirements 14.5**

        The chain produced by InferenceChainBuilder should have its
        cumulative_confidence equal to what compute_chain_confidence returns.
        """
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_chain(
            hops_data=hops_data,
            source_snippet_id=source_snippet_id,
            final_conclusion="Test conclusion",
        )

        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
        )
        computed = engine.compute_chain_confidence(chain)

        assert abs(chain.cumulative_confidence - computed) < 1e-10, (
            f"Builder cumulative_confidence ({chain.cumulative_confidence}) does not match "
            f"engine compute_chain_confidence ({computed}) for chain with "
            f"{len(chain.hops)} hops"
        )

    @given(chain=inference_chain_strategy())
    @settings(max_examples=200, deadline=10000)
    def test_single_hop_confidence_equals_hop_score(self, chain):
        """For a single-hop chain, cumulative confidence equals the single hop score.

        **Validates: Requirements 14.5**

        When N=1, the product of hop scores is just the single hop confidence.
        """
        assume(len(chain.hops) == 1)

        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
        )

        computed = engine.compute_chain_confidence(chain)
        assert abs(computed - chain.hops[0].confidence) < 1e-10, (
            f"Single-hop chain confidence ({computed}) should equal the hop's "
            f"confidence ({chain.hops[0].confidence})"
        )


# =============================================================================
# Property 28: Inference Threshold Filtering
# =============================================================================


@pytest.mark.property
class TestInferenceThresholdFiltering:
    """Property 28: Inference Threshold Filtering.

    **Validates: Requirements 14.7**

    Every fact in downstream output has confidence >= threshold; no fact below
    threshold appears in the output.
    """

    @given(facts=facts_list_strategy(), threshold=threshold_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_all_output_facts_meet_threshold(self, facts, threshold):
        """Every fact in the filtered output has confidence >= threshold.

        **Validates: Requirements 14.7**

        After apply_threshold_filter, no fact in the returned list has
        confidence below the configured threshold.
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
            confidence_threshold=threshold,
        )

        filtered = engine.apply_threshold_filter(facts)

        for fact in filtered:
            assert fact.confidence >= threshold, (
                f"Fact '{fact.fact_id}' has confidence {fact.confidence} "
                f"which is below threshold {threshold}"
            )

    @given(facts=facts_list_strategy(), threshold=threshold_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_no_below_threshold_fact_in_output(self, facts, threshold):
        """No fact with confidence < threshold appears in the output.

        **Validates: Requirements 14.7**

        Facts that are below the threshold must be discarded and must not
        appear in the filtered output list.
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
            confidence_threshold=threshold,
        )

        filtered = engine.apply_threshold_filter(facts)
        filtered_ids = {f.fact_id for f in filtered}

        # Check that any fact below threshold is NOT in the output
        for fact in facts:
            if fact.confidence < threshold:
                assert fact.fact_id not in filtered_ids, (
                    f"Fact '{fact.fact_id}' with confidence {fact.confidence} "
                    f"(below threshold {threshold}) should not be in filtered output"
                )

    @given(facts=facts_list_strategy(), threshold=threshold_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_all_above_threshold_facts_are_preserved(self, facts, threshold):
        """All facts with confidence >= threshold are preserved in the output.

        **Validates: Requirements 14.7**

        The filter does not discard facts that meet or exceed the threshold.
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
            confidence_threshold=threshold,
        )

        filtered = engine.apply_threshold_filter(facts)
        filtered_ids = {f.fact_id for f in filtered}

        # Every fact at or above threshold must be in the output
        for fact in facts:
            if fact.confidence >= threshold:
                assert fact.fact_id in filtered_ids, (
                    f"Fact '{fact.fact_id}' with confidence {fact.confidence} "
                    f"(>= threshold {threshold}) should be preserved in output"
                )

    @given(facts=facts_list_strategy(), threshold=threshold_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_filter_output_count_equals_above_threshold_count(self, facts, threshold):
        """Filtered output count equals the number of facts at or above threshold.

        **Validates: Requirements 14.7**

        The filter partitions facts into kept (>= threshold) and discarded (< threshold)
        such that len(output) == count(facts with confidence >= threshold).
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
            confidence_threshold=threshold,
        )

        filtered = engine.apply_threshold_filter(facts)
        expected_count = sum(1 for f in facts if f.confidence >= threshold)

        assert len(filtered) == expected_count, (
            f"Filtered output has {len(filtered)} facts, but expected {expected_count} "
            f"facts with confidence >= {threshold}. "
            f"Input confidences: {[f.confidence for f in facts]}"
        )

    @given(facts=facts_list_strategy())
    @settings(max_examples=200, deadline=10000)
    def test_threshold_zero_returns_all_facts(self, facts):
        """With threshold 0.0, all facts are returned (since confidence >= 0.0 always).

        **Validates: Requirements 14.7**

        A threshold of 0.0 means no fact is discarded because all confidences
        are in [0.0, 1.0].
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
            confidence_threshold=0.0,
        )

        filtered = engine.apply_threshold_filter(facts)

        assert len(filtered) == len(facts), (
            f"With threshold 0.0, all {len(facts)} facts should be returned, "
            f"but got {len(filtered)}"
        )

    @given(facts=facts_list_strategy())
    @settings(max_examples=200, deadline=10000)
    def test_threshold_one_returns_only_perfect_confidence(self, facts):
        """With threshold 1.0, only facts with confidence == 1.0 are returned.

        **Validates: Requirements 14.7**

        A threshold of 1.0 is the strictest possible filter, only keeping
        facts with perfect confidence.
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
            confidence_threshold=1.0,
        )

        filtered = engine.apply_threshold_filter(facts)

        for fact in filtered:
            assert fact.confidence >= 1.0, (
                f"With threshold 1.0, fact '{fact.fact_id}' with confidence "
                f"{fact.confidence} should not be in output"
            )
        expected_count = sum(1 for f in facts if f.confidence >= 1.0)
        assert len(filtered) == expected_count, (
            f"With threshold 1.0, expected {expected_count} facts with "
            f"confidence >= 1.0, got {len(filtered)}"
        )

    @given(facts=facts_list_strategy(), threshold=threshold_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_filter_preserves_fact_ordering(self, facts, threshold):
        """Threshold filter preserves the original ordering of facts.

        **Validates: Requirements 14.7**

        The relative order of facts in the output matches their relative
        order in the input (filter is stable / order-preserving).
        """
        engine = ClinicalInferenceEngine(
            llm_client=MockLLMClient(),
            graph_service=MockGraphService(),
            confidence_threshold=threshold,
        )

        filtered = engine.apply_threshold_filter(facts)

        # Verify ordering is preserved relative to input
        filtered_ids = [f.fact_id for f in filtered]
        original_order = [f.fact_id for f in facts if f.confidence >= threshold]

        assert filtered_ids == original_order, (
            f"Filtered output order {filtered_ids} does not match expected "
            f"original order {original_order}"
        )
