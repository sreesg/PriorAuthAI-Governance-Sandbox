"""Unit tests for confidence calculation and threshold filtering.

Tests compute_chain_confidence() and apply_threshold_filter() methods of
the ClinicalInferenceEngine, verifying correct confidence computation
(product of hop confidences) and threshold-based filtering behavior.

Validates:
    - compute_chain_confidence(): single hop = hop confidence
    - compute_chain_confidence(): 2 hops = product of hop confidences
    - compute_chain_confidence(): 3 hops = product of all hop confidences
    - compute_chain_confidence(): 0.0 hop makes cumulative zero
    - apply_threshold_filter(): includes facts at threshold
    - apply_threshold_filter(): excludes facts below threshold
    - apply_threshold_filter(): respects custom thresholds
    - Integration: end-to-end from chain through filter with various thresholds

Requirements referenced: 14.5, 14.7
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    ClinicalInferenceEngine,
    InferenceChain,
    InferenceHop,
    InferredFact,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.infer = AsyncMock(return_value="[]")
    return client


@pytest.fixture
def mock_graph_service():
    """Mock graph service."""
    service = MagicMock()
    service.upsert_node = AsyncMock()
    service.upsert_relationship = AsyncMock()
    return service


@pytest.fixture
def engine(mock_llm_client, mock_graph_service):
    """ClinicalInferenceEngine with default threshold (0.3)."""
    return ClinicalInferenceEngine(
        llm_client=mock_llm_client,
        graph_service=mock_graph_service,
        confidence_threshold=0.3,
        inference_depth="shallow",
    )


@pytest.fixture
def engine_custom_threshold(mock_llm_client, mock_graph_service):
    """ClinicalInferenceEngine with custom threshold (0.6)."""
    return ClinicalInferenceEngine(
        llm_client=mock_llm_client,
        graph_service=mock_graph_service,
        confidence_threshold=0.6,
        inference_depth="deep",
    )


def _make_chain(hops_confidences: list[float]) -> InferenceChain:
    """Helper to build an InferenceChain with given hop confidences."""
    hops = [
        InferenceHop(
            hop_number=i + 1,
            source_text=f"source text for hop {i + 1}",
            intermediate_conclusion=f"conclusion at hop {i + 1}",
            confidence=conf,
        )
        for i, conf in enumerate(hops_confidences)
    ]
    # Compute expected cumulative
    cumulative = 1.0
    for conf in hops_confidences:
        cumulative *= conf
    return InferenceChain(
        chain_id=f"chain-test-{len(hops_confidences)}hop",
        hops=hops,
        cumulative_confidence=cumulative,
        source_snippet_id="snippet-test",
        final_conclusion="final test conclusion",
    )


def _make_fact(confidence: float, inference_type: str = "care_access_barrier") -> InferredFact:
    """Helper to create an InferredFact with a given confidence."""
    chain = _make_chain([confidence])
    return InferredFact(
        fact_id=f"fact-{confidence}",
        inference_type=inference_type,
        sdoh_category=None if inference_type != "sdoh_factor" else "housing_instability",
        conclusion=f"Test conclusion with confidence {confidence}",
        confidence=confidence,
        inference_chain=chain,
        source_text_excerpt="test source text",
        inferred_at=datetime.now(timezone.utc),
    )


# =============================================================================
# compute_chain_confidence Tests
# =============================================================================


class TestComputeChainConfidence:
    """Tests for compute_chain_confidence() method.

    Validates Requirement 14.5: cumulative confidence = product of individual hop
    confidence scores (h1 × h2 × ... × hn).
    """

    def test_single_hop_equals_hop_confidence(self, engine):
        """Single-hop chain: cumulative confidence equals the single hop confidence."""
        chain = _make_chain([0.85])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.85)

    def test_single_hop_high_confidence(self, engine):
        """Single-hop with high confidence (0.99)."""
        chain = _make_chain([0.99])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.99)

    def test_single_hop_low_confidence(self, engine):
        """Single-hop with low confidence (0.1)."""
        chain = _make_chain([0.1])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.1)

    def test_single_hop_exact_one(self, engine):
        """Single-hop with confidence 1.0 yields 1.0."""
        chain = _make_chain([1.0])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(1.0)

    def test_single_hop_exact_zero(self, engine):
        """Single-hop with confidence 0.0 yields 0.0."""
        chain = _make_chain([0.0])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.0)

    def test_two_hops_product(self, engine):
        """Two-hop chain: cumulative = hop1 × hop2."""
        chain = _make_chain([0.9, 0.8])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.9 * 0.8)

    def test_two_hops_equal_confidences(self, engine):
        """Two hops with equal confidences: cumulative = conf^2."""
        chain = _make_chain([0.7, 0.7])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.7 * 0.7)

    def test_two_hops_decreasing(self, engine):
        """Two hops with decreasing confidence."""
        chain = _make_chain([0.95, 0.4])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.95 * 0.4)

    def test_three_hops_product(self, engine):
        """Three-hop chain: cumulative = hop1 × hop2 × hop3."""
        chain = _make_chain([0.9, 0.8, 0.7])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.9 * 0.8 * 0.7)

    def test_three_hops_all_high(self, engine):
        """Three hops with high confidences."""
        chain = _make_chain([0.95, 0.92, 0.88])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.95 * 0.92 * 0.88)

    def test_three_hops_all_low(self, engine):
        """Three hops with low confidences produce a very small cumulative."""
        chain = _make_chain([0.3, 0.3, 0.3])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.3 * 0.3 * 0.3)
        assert result == pytest.approx(0.027)

    def test_zero_hop_confidence_makes_chain_zero(self, engine):
        """A single 0.0 confidence hop makes the entire chain confidence 0.0."""
        chain = _make_chain([0.9, 0.0, 0.8])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.0)

    def test_zero_confidence_first_hop(self, engine):
        """Zero confidence in the first hop makes cumulative 0.0."""
        chain = _make_chain([0.0, 0.9, 0.8])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.0)

    def test_zero_confidence_last_hop(self, engine):
        """Zero confidence in the last hop makes cumulative 0.0."""
        chain = _make_chain([0.9, 0.8, 0.0])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(0.0)

    def test_all_hops_one_yields_one(self, engine):
        """All hops with confidence 1.0 yields cumulative 1.0."""
        chain = _make_chain([1.0, 1.0, 1.0])
        result = engine.compute_chain_confidence(chain)
        assert result == pytest.approx(1.0)

    def test_result_always_between_zero_and_one(self, engine):
        """Result is always in [0.0, 1.0] for valid inputs."""
        test_cases = [
            [0.5],
            [0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 0.1],
            [0.99, 0.99, 0.99],
        ]
        for confidences in test_cases:
            chain = _make_chain(confidences)
            result = engine.compute_chain_confidence(chain)
            assert 0.0 <= result <= 1.0, f"Failed for confidences {confidences}"

    def test_more_hops_means_lower_or_equal_confidence(self, engine):
        """Adding more hops (with confidence < 1.0) reduces cumulative confidence."""
        chain_1 = _make_chain([0.8])
        chain_2 = _make_chain([0.8, 0.8])
        chain_3 = _make_chain([0.8, 0.8, 0.8])

        conf_1 = engine.compute_chain_confidence(chain_1)
        conf_2 = engine.compute_chain_confidence(chain_2)
        conf_3 = engine.compute_chain_confidence(chain_3)

        assert conf_1 > conf_2 > conf_3


# =============================================================================
# apply_threshold_filter Tests
# =============================================================================


class TestApplyThresholdFilter:
    """Tests for apply_threshold_filter() method.

    Validates Requirement 14.7: discard any inferred fact with confidence below
    configurable threshold (default 0.3).
    """

    def test_includes_facts_above_threshold(self, engine):
        """Facts with confidence above threshold (0.3) are included."""
        facts = [
            _make_fact(0.5),
            _make_fact(0.8),
            _make_fact(0.95),
        ]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 3

    def test_excludes_facts_below_threshold(self, engine):
        """Facts with confidence below threshold (0.3) are excluded."""
        facts = [
            _make_fact(0.1),
            _make_fact(0.2),
            _make_fact(0.29),
        ]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 0

    def test_includes_facts_exactly_at_threshold(self, engine):
        """Facts exactly at threshold (0.3) are included (>= comparison)."""
        facts = [_make_fact(0.3)]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.3

    def test_mixed_facts_filtered_correctly(self, engine):
        """Mixed bag of facts: only those >= threshold survive."""
        facts = [
            _make_fact(0.1),   # excluded (below 0.3)
            _make_fact(0.3),   # included (at threshold)
            _make_fact(0.5),   # included
            _make_fact(0.29),  # excluded (just below)
            _make_fact(0.31),  # included (just above)
        ]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 3
        confidences = [f.confidence for f in filtered]
        assert 0.3 in confidences
        assert 0.5 in confidences
        assert 0.31 in confidences

    def test_empty_input_returns_empty(self, engine):
        """Empty input list returns empty output."""
        filtered = engine.apply_threshold_filter([])
        assert filtered == []

    def test_all_above_threshold_returns_all(self, engine):
        """When all facts are above threshold, all are returned."""
        facts = [_make_fact(0.9), _make_fact(0.7), _make_fact(0.4)]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 3

    def test_all_below_threshold_returns_empty(self, engine):
        """When all facts are below threshold, none are returned."""
        facts = [_make_fact(0.1), _make_fact(0.2), _make_fact(0.29)]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 0

    def test_respects_custom_threshold_high(self, engine_custom_threshold):
        """Custom threshold (0.6) correctly filters out lower-confidence facts."""
        facts = [
            _make_fact(0.3),   # excluded (below 0.6)
            _make_fact(0.5),   # excluded (below 0.6)
            _make_fact(0.6),   # included (at 0.6 threshold)
            _make_fact(0.8),   # included
        ]
        filtered = engine_custom_threshold.apply_threshold_filter(facts)
        assert len(filtered) == 2
        assert all(f.confidence >= 0.6 for f in filtered)

    def test_respects_custom_threshold_zero(self, mock_llm_client, mock_graph_service):
        """Threshold of 0.0 includes all facts."""
        engine = ClinicalInferenceEngine(
            llm_client=mock_llm_client,
            graph_service=mock_graph_service,
            confidence_threshold=0.0,
        )
        facts = [_make_fact(0.0), _make_fact(0.01), _make_fact(0.5)]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 3

    def test_respects_custom_threshold_one(self, mock_llm_client, mock_graph_service):
        """Threshold of 1.0 only includes facts with confidence exactly 1.0."""
        engine = ClinicalInferenceEngine(
            llm_client=mock_llm_client,
            graph_service=mock_graph_service,
            confidence_threshold=1.0,
        )
        facts = [_make_fact(0.99), _make_fact(1.0)]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 1
        assert filtered[0].confidence == 1.0

    def test_preserves_fact_order(self, engine):
        """Filtered facts maintain their original order."""
        facts = [
            _make_fact(0.8),
            _make_fact(0.1),  # excluded
            _make_fact(0.5),
            _make_fact(0.2),  # excluded
            _make_fact(0.9),
        ]
        filtered = engine.apply_threshold_filter(facts)
        confidences = [f.confidence for f in filtered]
        assert confidences == [0.8, 0.5, 0.9]

    def test_filter_does_not_mutate_input(self, engine):
        """Filtering does not modify the original list."""
        facts = [_make_fact(0.1), _make_fact(0.5)]
        original_len = len(facts)
        engine.apply_threshold_filter(facts)
        assert len(facts) == original_len


# =============================================================================
# Integration Tests: Chain Confidence → Threshold Filtering
# =============================================================================


class TestChainConfidenceToThresholdIntegration:
    """End-to-end tests verifying confidence flows correctly from chain to filter.

    Tests the workflow: build chain → compute confidence → assign to fact → filter.
    Validates Requirements 14.5 and 14.7 working together.
    """

    def test_high_confidence_chain_passes_default_threshold(self, engine):
        """A chain with high hop confidences passes the default 0.3 threshold."""
        chain = _make_chain([0.9, 0.8])  # cumulative = 0.72
        cumulative = engine.compute_chain_confidence(chain)
        assert cumulative > engine.confidence_threshold

        fact = InferredFact(
            fact_id="fact-pass",
            inference_type="care_access_barrier",
            sdoh_category=None,
            conclusion="High confidence conclusion",
            confidence=cumulative,
            inference_chain=chain,
            source_text_excerpt="source text",
            inferred_at=datetime.now(timezone.utc),
        )
        filtered = engine.apply_threshold_filter([fact])
        assert len(filtered) == 1

    def test_low_confidence_chain_fails_default_threshold(self, engine):
        """A chain with low hop confidences fails the default 0.3 threshold."""
        chain = _make_chain([0.5, 0.5])  # cumulative = 0.25
        cumulative = engine.compute_chain_confidence(chain)
        assert cumulative < engine.confidence_threshold

        fact = InferredFact(
            fact_id="fact-fail",
            inference_type="medication_adherence_risk",
            sdoh_category=None,
            conclusion="Low confidence conclusion",
            confidence=cumulative,
            inference_chain=chain,
            source_text_excerpt="source text",
            inferred_at=datetime.now(timezone.utc),
        )
        filtered = engine.apply_threshold_filter([fact])
        assert len(filtered) == 0

    def test_three_hop_chain_above_threshold(self, engine):
        """Three-hop chain with decent confidences can still pass threshold."""
        chain = _make_chain([0.8, 0.7, 0.6])  # cumulative = 0.336
        cumulative = engine.compute_chain_confidence(chain)
        assert cumulative >= engine.confidence_threshold  # 0.336 >= 0.3

        fact = InferredFact(
            fact_id="fact-3hop-pass",
            inference_type="sdoh_factor",
            sdoh_category="housing_instability",
            conclusion="Multi-hop passes threshold",
            confidence=cumulative,
            inference_chain=chain,
            source_text_excerpt="source",
            inferred_at=datetime.now(timezone.utc),
        )
        filtered = engine.apply_threshold_filter([fact])
        assert len(filtered) == 1

    def test_three_hop_chain_below_threshold(self, engine):
        """Three-hop chain with lower confidences fails threshold."""
        chain = _make_chain([0.6, 0.6, 0.6])  # cumulative = 0.216
        cumulative = engine.compute_chain_confidence(chain)
        assert cumulative < engine.confidence_threshold  # 0.216 < 0.3

        fact = InferredFact(
            fact_id="fact-3hop-fail",
            inference_type="sdoh_factor",
            sdoh_category="food_insecurity",
            conclusion="Multi-hop fails threshold",
            confidence=cumulative,
            inference_chain=chain,
            source_text_excerpt="source",
            inferred_at=datetime.now(timezone.utc),
        )
        filtered = engine.apply_threshold_filter([fact])
        assert len(filtered) == 0

    def test_zero_hop_in_chain_always_fails_threshold(self, engine):
        """Any chain with a 0.0 hop always produces 0.0, always fails threshold."""
        chain = _make_chain([0.9, 0.0, 0.9])  # cumulative = 0.0
        cumulative = engine.compute_chain_confidence(chain)
        assert cumulative == 0.0

        fact = InferredFact(
            fact_id="fact-zero",
            inference_type="care_access_barrier",
            sdoh_category=None,
            conclusion="Zero confidence chain",
            confidence=cumulative,
            inference_chain=chain,
            source_text_excerpt="source",
            inferred_at=datetime.now(timezone.utc),
        )
        # Only passes if threshold is exactly 0.0
        filtered = engine.apply_threshold_filter([fact])
        assert len(filtered) == 0  # 0.0 < 0.3 threshold

    def test_mixed_chains_various_thresholds(self, engine_custom_threshold):
        """Multiple chains with various confidence levels against custom threshold (0.6)."""
        # Chain 1: single hop 0.8 → cumulative 0.8 → passes 0.6
        chain1 = _make_chain([0.8])
        cum1 = engine_custom_threshold.compute_chain_confidence(chain1)

        # Chain 2: two hops 0.9 × 0.8 = 0.72 → passes 0.6
        chain2 = _make_chain([0.9, 0.8])
        cum2 = engine_custom_threshold.compute_chain_confidence(chain2)

        # Chain 3: two hops 0.7 × 0.7 = 0.49 → fails 0.6
        chain3 = _make_chain([0.7, 0.7])
        cum3 = engine_custom_threshold.compute_chain_confidence(chain3)

        # Chain 4: three hops 0.9 × 0.9 × 0.9 = 0.729 → passes 0.6
        chain4 = _make_chain([0.9, 0.9, 0.9])
        cum4 = engine_custom_threshold.compute_chain_confidence(chain4)

        facts = [
            InferredFact(
                fact_id="fact-1",
                inference_type="care_access_barrier",
                sdoh_category=None,
                conclusion="chain 1",
                confidence=cum1,
                inference_chain=chain1,
                source_text_excerpt="s1",
                inferred_at=datetime.now(timezone.utc),
            ),
            InferredFact(
                fact_id="fact-2",
                inference_type="medication_adherence_risk",
                sdoh_category=None,
                conclusion="chain 2",
                confidence=cum2,
                inference_chain=chain2,
                source_text_excerpt="s2",
                inferred_at=datetime.now(timezone.utc),
            ),
            InferredFact(
                fact_id="fact-3",
                inference_type="sdoh_factor",
                sdoh_category="transportation_barriers",
                conclusion="chain 3",
                confidence=cum3,
                inference_chain=chain3,
                source_text_excerpt="s3",
                inferred_at=datetime.now(timezone.utc),
            ),
            InferredFact(
                fact_id="fact-4",
                inference_type="sdoh_factor",
                sdoh_category="caregiver_availability",
                conclusion="chain 4",
                confidence=cum4,
                inference_chain=chain4,
                source_text_excerpt="s4",
                inferred_at=datetime.now(timezone.utc),
            ),
        ]

        filtered = engine_custom_threshold.apply_threshold_filter(facts)
        # Chain 1 (0.8), Chain 2 (0.72), Chain 4 (0.729) pass; Chain 3 (0.49) fails
        assert len(filtered) == 3
        filtered_ids = [f.fact_id for f in filtered]
        assert "fact-1" in filtered_ids
        assert "fact-2" in filtered_ids
        assert "fact-4" in filtered_ids
        assert "fact-3" not in filtered_ids

    def test_exactly_at_threshold_boundary(self, mock_llm_client, mock_graph_service):
        """Chain producing confidence exactly at threshold is included."""
        # Threshold 0.5, chain with hops [0.5, 1.0] = 0.5
        engine = ClinicalInferenceEngine(
            llm_client=mock_llm_client,
            graph_service=mock_graph_service,
            confidence_threshold=0.5,
        )
        chain = _make_chain([0.5, 1.0])  # cumulative = 0.5
        cumulative = engine.compute_chain_confidence(chain)
        assert cumulative == pytest.approx(0.5)

        fact = InferredFact(
            fact_id="fact-boundary",
            inference_type="care_access_barrier",
            sdoh_category=None,
            conclusion="Exactly at boundary",
            confidence=cumulative,
            inference_chain=chain,
            source_text_excerpt="boundary source",
            inferred_at=datetime.now(timezone.utc),
        )
        filtered = engine.apply_threshold_filter([fact])
        assert len(filtered) == 1

    def test_default_threshold_is_0_3(self, engine):
        """Verify the default confidence threshold is 0.3 as specified."""
        assert engine.confidence_threshold == 0.3
        assert ClinicalInferenceEngine.DEFAULT_CONFIDENCE_THRESHOLD == 0.3
