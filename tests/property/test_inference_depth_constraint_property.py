"""Property-based tests for Inference Depth Constraint.

**Validates: Requirements 14.4**

Property 29: Inference Depth Constraint
- Shallow depth produces exactly 1 hop.
- Deep depth produces 1-3 hops.
- No chain exceeds 3 hops regardless of input size.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clinical_reasoning_fabric.inference.inference_chain_builder import (
    InferenceChainBuilder,
    MAX_CHAIN_HOPS,
)
from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    InferenceChain,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for a single valid hop dictionary
hop_strategy = st.fixed_dictionaries({
    "source_text": st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=100,
    ),
    "intermediate_conclusion": st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=100,
    ),
    "confidence": st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
})

# Strategy for generating random hop data with 1-20 hops (to stress-test truncation)
hops_data_strategy = st.lists(hop_strategy, min_size=1, max_size=20)

# Strategy for source snippet IDs
source_snippet_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=30,
)

# Strategy for final conclusion text
final_conclusion_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=100,
)


# =============================================================================
# Property 29: Inference Depth Constraint
# =============================================================================


@pytest.mark.property
class TestInferenceDepthConstraint:
    """Property 29: Inference Depth Constraint.

    **Validates: Requirements 14.4**

    Tests that:
    - Shallow depth produces exactly 1 hop
    - Deep depth produces 1-3 hops
    - No chain exceeds 3 hops regardless of configuration or input
    """

    @given(
        hops_data=hops_data_strategy,
        source_snippet_id=source_snippet_id_strategy,
        final_conclusion=final_conclusion_strategy,
    )
    @settings(max_examples=200, deadline=10000)
    def test_shallow_depth_produces_exactly_one_hop(
        self, hops_data, source_snippet_id, final_conclusion
    ):
        """Shallow depth produces exactly 1 hop regardless of input size.

        **Validates: Requirements 14.4**

        When configured with shallow depth, build_chain must produce a chain
        with exactly 1 hop, even if the input hops_data contains many hops.
        """
        builder = InferenceChainBuilder(depth="shallow")
        chain = builder.build_chain(
            hops_data=hops_data,
            source_snippet_id=source_snippet_id,
            final_conclusion=final_conclusion,
        )

        assert isinstance(chain, InferenceChain), (
            f"Expected InferenceChain, got {type(chain)}"
        )
        assert len(chain.hops) == 1, (
            f"Shallow depth must produce exactly 1 hop, got {len(chain.hops)} hops "
            f"with {len(hops_data)} input hops"
        )
        assert chain.hops[0].hop_number == 1, (
            f"Single hop must have hop_number=1, got {chain.hops[0].hop_number}"
        )

    @given(
        hops_data=hops_data_strategy,
        source_snippet_id=source_snippet_id_strategy,
        final_conclusion=final_conclusion_strategy,
    )
    @settings(max_examples=200, deadline=10000)
    def test_deep_depth_produces_one_to_three_hops(
        self, hops_data, source_snippet_id, final_conclusion
    ):
        """Deep depth produces 1-3 hops.

        **Validates: Requirements 14.4**

        When configured with deep depth, build_chain produces at least 1 hop
        and at most 3 hops. The number of hops is min(len(hops_data), 3).
        """
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_chain(
            hops_data=hops_data,
            source_snippet_id=source_snippet_id,
            final_conclusion=final_conclusion,
        )

        assert isinstance(chain, InferenceChain), (
            f"Expected InferenceChain, got {type(chain)}"
        )
        assert 1 <= len(chain.hops) <= 3, (
            f"Deep depth must produce 1-3 hops, got {len(chain.hops)} hops "
            f"with {len(hops_data)} input hops"
        )
        expected_hops = min(len(hops_data), MAX_CHAIN_HOPS)
        assert len(chain.hops) == expected_hops, (
            f"Deep depth with {len(hops_data)} input hops should produce "
            f"{expected_hops} hops, got {len(chain.hops)}"
        )

    @given(
        hops_data=hops_data_strategy,
        source_snippet_id=source_snippet_id_strategy,
        final_conclusion=final_conclusion_strategy,
        depth=st.sampled_from(["shallow", "deep"]),
    )
    @settings(max_examples=200, deadline=10000)
    def test_no_chain_exceeds_max_hops(
        self, hops_data, source_snippet_id, final_conclusion, depth
    ):
        """No chain exceeds 3 hops regardless of configuration or input.

        **Validates: Requirements 14.4**

        The absolute maximum chain length is MAX_CHAIN_HOPS (3). This must hold
        for both shallow and deep depth, and for any number of input hops (1-20).
        """
        builder = InferenceChainBuilder(depth=depth)
        chain = builder.build_chain(
            hops_data=hops_data,
            source_snippet_id=source_snippet_id,
            final_conclusion=final_conclusion,
        )

        assert len(chain.hops) <= MAX_CHAIN_HOPS, (
            f"Chain exceeds MAX_CHAIN_HOPS ({MAX_CHAIN_HOPS}): got {len(chain.hops)} hops "
            f"with depth='{depth}' and {len(hops_data)} input hops"
        )

    @given(
        hops_data=st.lists(hop_strategy, min_size=4, max_size=20),
        source_snippet_id=source_snippet_id_strategy,
        final_conclusion=final_conclusion_strategy,
    )
    @settings(max_examples=200, deadline=10000)
    def test_deep_depth_truncates_excess_hops(
        self, hops_data, source_snippet_id, final_conclusion
    ):
        """Deep depth truncates input to exactly 3 hops when given more than 3.

        **Validates: Requirements 14.4**

        When hops_data has more than 3 entries, deep mode uses only the first 3
        and discards the rest, ensuring the 3-hop maximum is enforced.
        """
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_chain(
            hops_data=hops_data,
            source_snippet_id=source_snippet_id,
            final_conclusion=final_conclusion,
        )

        assert len(chain.hops) == MAX_CHAIN_HOPS, (
            f"Deep depth with {len(hops_data)} input hops (>3) should produce "
            f"exactly {MAX_CHAIN_HOPS} hops, got {len(chain.hops)}"
        )
        # Verify hop numbers are sequential 1, 2, 3
        for i, hop in enumerate(chain.hops):
            assert hop.hop_number == i + 1, (
                f"Hop {i} should have hop_number={i + 1}, got {hop.hop_number}"
            )

    @given(
        hops_data=hops_data_strategy,
        source_snippet_id=source_snippet_id_strategy,
        final_conclusion=final_conclusion_strategy,
        depth=st.sampled_from(["shallow", "deep"]),
    )
    @settings(max_examples=200, deadline=10000)
    def test_chain_has_valid_cumulative_confidence(
        self, hops_data, source_snippet_id, final_conclusion, depth
    ):
        """Chain cumulative_confidence equals product of individual hop confidences.

        **Validates: Requirements 14.4**

        The cumulative_confidence on the chain must be the product of all hop
        confidence scores, and must be within [0.0, 1.0].
        """
        builder = InferenceChainBuilder(depth=depth)
        chain = builder.build_chain(
            hops_data=hops_data,
            source_snippet_id=source_snippet_id,
            final_conclusion=final_conclusion,
        )

        # Compute expected cumulative confidence
        expected_confidence = 1.0
        for hop in chain.hops:
            expected_confidence *= hop.confidence

        assert abs(chain.cumulative_confidence - expected_confidence) < 1e-10, (
            f"Cumulative confidence {chain.cumulative_confidence} does not match "
            f"expected product {expected_confidence} of hop confidences "
            f"{[h.confidence for h in chain.hops]}"
        )
        assert 0.0 <= chain.cumulative_confidence <= 1.0, (
            f"Cumulative confidence {chain.cumulative_confidence} is outside [0.0, 1.0]"
        )

    @given(
        hops_data=hops_data_strategy,
        source_snippet_id=source_snippet_id_strategy,
    )
    @settings(max_examples=200, deadline=10000)
    def test_build_deep_chain_directly_enforces_max_hops(
        self, hops_data, source_snippet_id
    ):
        """build_deep_chain directly enforces the 3-hop maximum.

        **Validates: Requirements 14.4**

        Calling build_deep_chain with arbitrary-length hops_data always produces
        a chain with at most 3 hops.
        """
        builder = InferenceChainBuilder(depth="deep")
        chain = builder.build_deep_chain(
            hops_data=hops_data,
            source_snippet_id=source_snippet_id,
        )

        assert 1 <= len(chain.hops) <= MAX_CHAIN_HOPS, (
            f"build_deep_chain must produce 1-{MAX_CHAIN_HOPS} hops, "
            f"got {len(chain.hops)} from {len(hops_data)} input hops"
        )
