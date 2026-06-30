"""Unit tests for InferenceChainBuilder.

Tests the dedicated builder for constructing inference chains with configurable
depth (shallow vs deep), enforcing the 3-hop maximum, and computing cumulative
confidence correctly.

Validates:
    - Shallow always produces exactly 1 hop
    - Deep can produce 1-3 hops
    - No chain ever has > 3 hops (even with more input)
    - Cumulative confidence computation is correct (product of hop confidences)
    - Default depth is shallow
    - Invalid inputs raise appropriate errors

Requirements referenced: 14.4, 14.5
"""

import pytest

from clinical_reasoning_fabric.inference.inference_chain_builder import (
    DEFAULT_DEPTH,
    MAX_CHAIN_HOPS,
    InferenceChainBuilder,
)
from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    InferenceChain,
    InferenceHop,
)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInferenceChainBuilderInit:
    """Tests for InferenceChainBuilder initialization."""

    def test_default_depth_is_shallow(self):
        """Default depth is 'shallow'."""
        builder = InferenceChainBuilder()
        assert builder.depth == "shallow"
        assert DEFAULT_DEPTH == "shallow"

    def test_explicit_shallow_depth(self):
        """Can initialize with explicit 'shallow' depth."""
        builder = InferenceChainBuilder(depth="shallow")
        assert builder.depth == "shallow"

    def test_explicit_deep_depth(self):
        """Can initialize with explicit 'deep' depth."""
        builder = InferenceChainBuilder(depth="deep")
        assert builder.depth == "deep"

    def test_invalid_depth_raises_error(self):
        """Invalid depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be 'shallow' or 'deep'"):
            InferenceChainBuilder(depth="medium")

    def test_invalid_depth_empty_string(self):
        """Empty string depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be 'shallow' or 'deep'"):
            InferenceChainBuilder(depth="")

    def test_max_hops_constant_is_3(self):
        """MAX_HOPS class constant is 3."""
        assert InferenceChainBuilder.MAX_HOPS == 3
        assert MAX_CHAIN_HOPS == 3


# =============================================================================
# build_shallow_chain Tests
# =============================================================================


class TestBuildShallowChain:
    """Tests for build_shallow_chain method."""

    def test_shallow_chain_produces_exactly_one_hop(self):
        """Shallow chain always has exactly 1 hop."""
        builder = InferenceChainBuilder(depth="shallow")
        chain = builder.build_shallow_chain(
            source_text="Patient mentions taking the bus",
            conclusion="Patient faces transportation barriers",
            confidence=0.75,
        )
        assert len(chain.hops) == 1

    def test_shallow_chain_hop_has_correct_fields(self):
        """The single hop records source_text, intermediate_conclusion, confidence."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="Patient mentions taking the bus",
            conclusion="Transportation barrier identified",
            confidence=0.8,
        )
        hop = chain.hops[0]
        assert hop.hop_number == 1
        assert hop.source_text == "Patient mentions taking the bus"
        assert hop.intermediate_conclusion == "Transportation barrier identified"
        assert hop.confidence == 0.8

    def test_shallow_chain_cumulative_confidence_equals_hop_confidence(self):
        """Cumulative confidence for 1-hop chain equals the hop's confidence."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="text",
            conclusion="conclusion",
            confidence=0.65,
        )
        assert chain.cumulative_confidence == pytest.approx(0.65)

    def test_shallow_chain_final_conclusion_matches(self):
        """Final conclusion matches the provided conclusion."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="text",
            conclusion="Patient has food insecurity",
            confidence=0.9,
        )
        assert chain.final_conclusion == "Patient has food insecurity"

    def test_shallow_chain_source_snippet_id(self):
        """Source snippet ID is correctly set."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="text",
            conclusion="conclusion",
            confidence=0.7,
            source_snippet_id="snippet-abc-123",
        )
        assert chain.source_snippet_id == "snippet-abc-123"

    def test_shallow_chain_has_chain_id(self):
        """Chain has a non-empty chain_id."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="text",
            conclusion="conclusion",
            confidence=0.5,
        )
        assert chain.chain_id
        assert chain.chain_id.startswith("chain-")

    def test_shallow_chain_returns_inference_chain_type(self):
        """Returns an InferenceChain instance."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="text",
            conclusion="conclusion",
            confidence=0.5,
        )
        assert isinstance(chain, InferenceChain)

    def test_shallow_chain_confidence_zero(self):
        """Confidence of 0.0 is valid."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="text",
            conclusion="conclusion",
            confidence=0.0,
        )
        assert chain.cumulative_confidence == 0.0

    def test_shallow_chain_confidence_one(self):
        """Confidence of 1.0 is valid."""
        builder = InferenceChainBuilder()
        chain = builder.build_shallow_chain(
            source_text="text",
            conclusion="conclusion",
            confidence=1.0,
        )
        assert chain.cumulative_confidence == 1.0

    def test_shallow_chain_invalid_confidence_too_high(self):
        """Confidence > 1.0 raises ValueError."""
        builder = InferenceChainBuilder()
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            builder.build_shallow_chain(
                source_text="text",
                conclusion="conclusion",
                confidence=1.5,
            )

    def test_shallow_chain_invalid_confidence_negative(self):
        """Negative confidence raises ValueError."""
        builder = InferenceChainBuilder()
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            builder.build_shallow_chain(
                source_text="text",
                conclusion="conclusion",
                confidence=-0.1,
            )


# =============================================================================
# build_deep_chain Tests
# =============================================================================


class TestBuildDeepChain:
    """Tests for build_deep_chain method."""

    def test_deep_chain_single_hop(self):
        """Deep chain with 1 hop input produces 1-hop chain."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {
                    "source_text": "Patient takes the bus",
                    "intermediate_conclusion": "Lacks personal transportation",
                    "confidence": 0.8,
                }
            ],
        )
        assert len(chain.hops) == 1

    def test_deep_chain_two_hops(self):
        """Deep chain with 2 hops produces 2-hop chain."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {
                    "source_text": "Patient takes the bus",
                    "intermediate_conclusion": "Lacks personal transportation",
                    "confidence": 0.85,
                },
                {
                    "source_text": "Lacks personal transportation",
                    "intermediate_conclusion": "May miss appointments due to transport barriers",
                    "confidence": 0.7,
                },
            ],
        )
        assert len(chain.hops) == 2

    def test_deep_chain_three_hops(self):
        """Deep chain with 3 hops produces 3-hop chain."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {
                    "source_text": "Patient reports difficulty with insulin storage",
                    "intermediate_conclusion": "May lack refrigeration",
                    "confidence": 0.8,
                },
                {
                    "source_text": "May lack refrigeration",
                    "intermediate_conclusion": "Housing instability possible",
                    "confidence": 0.7,
                },
                {
                    "source_text": "Housing instability possible",
                    "intermediate_conclusion": "Medication adherence at risk due to storage",
                    "confidence": 0.6,
                },
            ],
        )
        assert len(chain.hops) == 3

    def test_deep_chain_truncates_to_3_hops(self):
        """Deep chain with more than 3 hops is truncated to exactly 3."""
        builder = InferenceChainBuilder(depth="deep")
        hops_data = [
            {"source_text": f"hop {i}", "intermediate_conclusion": f"conc {i}", "confidence": 0.8}
            for i in range(5)
        ]
        chain = builder.build_deep_chain(hops_data=hops_data)
        assert len(chain.hops) == 3

    def test_deep_chain_truncates_to_3_hops_large_input(self):
        """Even with 10 hops, chain never exceeds 3."""
        builder = InferenceChainBuilder(depth="deep")
        hops_data = [
            {"source_text": f"hop {i}", "intermediate_conclusion": f"conc {i}", "confidence": 0.9}
            for i in range(10)
        ]
        chain = builder.build_deep_chain(hops_data=hops_data)
        assert len(chain.hops) == 3
        assert len(chain.hops) <= MAX_CHAIN_HOPS

    def test_deep_chain_cumulative_confidence_product(self):
        """Cumulative confidence is the product of hop confidences."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.8},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 0.7},
            ],
        )
        expected = 0.9 * 0.8 * 0.7
        assert chain.cumulative_confidence == pytest.approx(expected)

    def test_deep_chain_two_hop_cumulative_confidence(self):
        """Two-hop cumulative confidence is product of two hop confidences."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.6},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.5},
            ],
        )
        expected = 0.6 * 0.5
        assert chain.cumulative_confidence == pytest.approx(expected)

    def test_deep_chain_final_conclusion_from_last_hop(self):
        """If no final_conclusion provided, uses last hop's intermediate_conclusion."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "step 1", "confidence": 0.8},
                {"source_text": "t2", "intermediate_conclusion": "final step", "confidence": 0.7},
            ],
        )
        assert chain.final_conclusion == "final step"

    def test_deep_chain_explicit_final_conclusion(self):
        """Explicit final_conclusion overrides last hop's conclusion."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "step 1", "confidence": 0.8},
            ],
            final_conclusion="My explicit conclusion",
        )
        assert chain.final_conclusion == "My explicit conclusion"

    def test_deep_chain_hop_numbers_sequential(self):
        """Hop numbers are sequential starting from 1."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.8},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 0.7},
            ],
        )
        for i, hop in enumerate(chain.hops):
            assert hop.hop_number == i + 1

    def test_deep_chain_empty_hops_raises_error(self):
        """Empty hops_data raises ValueError."""
        builder = InferenceChainBuilder(depth="deep")
        with pytest.raises(ValueError, match="must contain at least one hop"):
            builder.build_deep_chain(hops_data=[])

    def test_deep_chain_invalid_confidence_raises_error(self):
        """Invalid confidence in hop data raises ValueError."""
        builder = InferenceChainBuilder(depth="deep")
        with pytest.raises(ValueError, match="Hop confidence must be between"):
            builder.build_deep_chain(
                hops_data=[
                    {"source_text": "t", "intermediate_conclusion": "c", "confidence": 1.5},
                ],
            )

    def test_deep_chain_non_dict_hop_raises_error(self):
        """Non-dict hop data raises ValueError."""
        builder = InferenceChainBuilder(depth="deep")
        with pytest.raises(ValueError, match="Each hop must be a dictionary"):
            builder.build_deep_chain(hops_data=["not a dict"])

    def test_deep_chain_source_snippet_id(self):
        """Source snippet ID is correctly stored."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.8},
            ],
            source_snippet_id="snippet-xyz",
        )
        assert chain.source_snippet_id == "snippet-xyz"


# =============================================================================
# build_chain (Depth-Aware) Tests
# =============================================================================


class TestBuildChainDepthAware:
    """Tests for the depth-aware build_chain method."""

    def test_shallow_builder_produces_1_hop_from_multiple_hops(self):
        """Shallow builder uses only first hop even if multiple provided."""
        builder = InferenceChainBuilder(depth="shallow")
        chain = builder.build_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.8},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 0.7},
            ],
        )
        assert len(chain.hops) == 1

    def test_deep_builder_uses_up_to_3_hops(self):
        """Deep builder uses up to 3 hops from input."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.8},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 0.7},
            ],
        )
        assert len(chain.hops) == 3

    def test_deep_builder_truncates_excess_hops(self):
        """Deep builder truncates to 3 hops when more provided."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_chain(
            hops_data=[
                {"source_text": f"t{i}", "intermediate_conclusion": f"c{i}", "confidence": 0.8}
                for i in range(7)
            ],
        )
        assert len(chain.hops) == 3
        assert len(chain.hops) <= MAX_CHAIN_HOPS

    def test_build_chain_empty_hops_raises_error(self):
        """Empty hops_data raises ValueError."""
        builder = InferenceChainBuilder(depth="shallow")
        with pytest.raises(ValueError, match="must contain at least one hop"):
            builder.build_chain(hops_data=[])

    def test_shallow_chain_confidence_from_first_hop(self):
        """Shallow chain confidence comes from first hop only."""
        builder = InferenceChainBuilder(depth="shallow")
        chain = builder.build_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.1},
            ],
        )
        # Should only use first hop's confidence
        assert chain.cumulative_confidence == pytest.approx(0.9)

    def test_deep_chain_confidence_product_of_all_hops(self):
        """Deep chain confidence is product of all hop confidences."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.8},
            ],
        )
        assert chain.cumulative_confidence == pytest.approx(0.9 * 0.8)


# =============================================================================
# 3-Hop Maximum Enforcement Tests
# =============================================================================


class TestMaxHopEnforcement:
    """Tests ensuring no chain ever exceeds 3 hops regardless of configuration."""

    def test_shallow_never_exceeds_3_hops(self):
        """Shallow chain never exceeds 3 hops (always 1)."""
        builder = InferenceChainBuilder(depth="shallow")
        chain = builder.build_chain(
            hops_data=[
                {"source_text": f"t{i}", "intermediate_conclusion": f"c{i}", "confidence": 0.9}
                for i in range(20)
            ],
        )
        assert len(chain.hops) <= 3
        assert len(chain.hops) == 1

    def test_deep_never_exceeds_3_hops_with_4_input(self):
        """Deep chain with 4 hops input produces at most 3 hops."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": f"t{i}", "intermediate_conclusion": f"c{i}", "confidence": 0.8}
                for i in range(4)
            ],
        )
        assert len(chain.hops) <= 3

    def test_deep_never_exceeds_3_hops_with_100_input(self):
        """Deep chain with 100 hops input produces at most 3 hops."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": f"t{i}", "intermediate_conclusion": f"c{i}", "confidence": 0.9}
                for i in range(100)
            ],
        )
        assert len(chain.hops) <= 3

    def test_build_chain_deep_never_exceeds_3(self):
        """build_chain with deep depth never exceeds 3 hops."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_chain(
            hops_data=[
                {"source_text": f"t{i}", "intermediate_conclusion": f"c{i}", "confidence": 0.7}
                for i in range(50)
            ],
        )
        assert len(chain.hops) <= MAX_CHAIN_HOPS


# =============================================================================
# Cumulative Confidence Computation Tests
# =============================================================================


class TestCumulativeConfidence:
    """Tests for cumulative confidence calculation (product of hop confidences)."""

    def test_single_hop_confidence_equals_hop(self):
        """Single hop: cumulative = hop confidence."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t", "intermediate_conclusion": "c", "confidence": 0.75},
            ],
        )
        assert chain.cumulative_confidence == pytest.approx(0.75)

    def test_two_hop_confidence_is_product(self):
        """Two hops: cumulative = hop1 * hop2."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.8},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.6},
            ],
        )
        assert chain.cumulative_confidence == pytest.approx(0.8 * 0.6)

    def test_three_hop_confidence_is_product(self):
        """Three hops: cumulative = hop1 * hop2 * hop3."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.8},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 0.7},
            ],
        )
        assert chain.cumulative_confidence == pytest.approx(0.9 * 0.8 * 0.7)

    def test_zero_confidence_hop_makes_chain_zero(self):
        """If any hop has 0.0 confidence, cumulative is 0.0."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.0},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 0.8},
            ],
        )
        assert chain.cumulative_confidence == pytest.approx(0.0)

    def test_all_perfect_confidence(self):
        """All hops with 1.0 confidence yields 1.0 cumulative."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 1.0},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 1.0},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 1.0},
            ],
        )
        assert chain.cumulative_confidence == pytest.approx(1.0)

    def test_cumulative_confidence_only_uses_included_hops(self):
        """When truncated, cumulative only includes the 3 retained hops."""
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=[
                {"source_text": "t1", "intermediate_conclusion": "c1", "confidence": 0.9},
                {"source_text": "t2", "intermediate_conclusion": "c2", "confidence": 0.8},
                {"source_text": "t3", "intermediate_conclusion": "c3", "confidence": 0.7},
                {"source_text": "t4", "intermediate_conclusion": "c4", "confidence": 0.1},  # truncated
                {"source_text": "t5", "intermediate_conclusion": "c5", "confidence": 0.1},  # truncated
            ],
        )
        # Only first 3 hops should be used
        expected = 0.9 * 0.8 * 0.7
        assert chain.cumulative_confidence == pytest.approx(expected)
        assert len(chain.hops) == 3
