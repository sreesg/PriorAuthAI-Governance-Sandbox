"""Property-based tests for Reciprocal Rank Fusion Correctness.

**Validates: Requirements 2.3**

Property 4: Reciprocal Rank Fusion Correctness
- For any two ranked lists: output is sorted descending by RRF score,
  bounded to top_k, and items appearing in both lists score higher than
  items in only one list at equivalent ranks.
- All items in the output came from at least one input list.
- All output scores are in [0, 1].
"""

import hashlib
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    ScoredChunk,
)
from clinical_reasoning_fabric.retrieval.hybrid_retrieval_service import (
    HybridRetrievalService,
    RRF_K_CONSTANT,
    DEFAULT_FINAL_TOP_K,
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


def _make_content_hash(chunk_id: str) -> str:
    """Create a valid 64-char hex content hash from a chunk_id."""
    return hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()


def _make_scored_chunk(chunk_id: str) -> ScoredChunk:
    """Create a valid ScoredChunk with the given chunk_id."""
    content_hash = _make_content_hash(chunk_id)
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"Clinical text for chunk {chunk_id}",
        score=0.5,  # Placeholder score; RRF will override
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


def _make_service() -> HybridRetrievalService:
    """Create a HybridRetrievalService instance for testing RRF (no external deps needed)."""
    # RRF is a pure function on the service; we just need an instance.
    # Pass None for clients since we won't call async methods.
    return HybridRetrievalService(
        qdrant_client=None,
        kms_client=None,
        embedding_model=None,
        collection_name="test_collection",
    )


# =============================================================================
# Hypothesis strategies
# =============================================================================


@st.composite
def scored_chunk_list_strategy(draw, min_size=0, max_size=50):
    """Generate a list of ScoredChunks with unique chunk_ids."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    chunk_ids = [f"chunk-{i:04d}" for i in range(size)]
    return [_make_scored_chunk(cid) for cid in chunk_ids]


@st.composite
def overlapping_ranked_lists_strategy(draw):
    """Generate two ranked lists with controlled overlap.

    Returns (dense_results, sparse_results, overlap_ids, dense_only_ids, sparse_only_ids)
    where overlap_ids appear in BOTH lists.
    """
    # Number of items unique to each list and in overlap
    n_overlap = draw(st.integers(min_value=0, max_value=20))
    n_dense_only = draw(st.integers(min_value=0, max_value=30))
    n_sparse_only = draw(st.integers(min_value=0, max_value=30))

    # Generate unique chunk IDs for each group
    overlap_ids = [f"overlap-{i:04d}" for i in range(n_overlap)]
    dense_only_ids = [f"dense-{i:04d}" for i in range(n_dense_only)]
    sparse_only_ids = [f"sparse-{i:04d}" for i in range(n_sparse_only)]

    # Build dense results: overlap items + dense-only items, shuffled
    dense_chunk_ids = overlap_ids + dense_only_ids
    # Shuffle to get random rank assignment
    dense_order = draw(st.permutations(range(len(dense_chunk_ids))))
    dense_results = [_make_scored_chunk(dense_chunk_ids[i]) for i in dense_order]

    # Build sparse results: overlap items + sparse-only items, shuffled
    sparse_chunk_ids = overlap_ids + sparse_only_ids
    sparse_order = draw(st.permutations(range(len(sparse_chunk_ids))))
    sparse_results = [_make_scored_chunk(sparse_chunk_ids[i]) for i in sparse_order]

    return dense_results, sparse_results, set(overlap_ids), set(dense_only_ids), set(sparse_only_ids)


@st.composite
def top_k_strategy(draw):
    """Generate a reasonable top_k value."""
    return draw(st.integers(min_value=1, max_value=50))


# =============================================================================
# Property tests
# =============================================================================


@pytest.mark.property
class TestRRFCorrectness:
    """Property 4: Reciprocal Rank Fusion Correctness.

    **Validates: Requirements 2.3**

    Tests that the RRF algorithm produces correctly sorted, bounded output
    with proper scoring behavior for overlapping items.
    """

    service = _make_service()

    @given(data=overlapping_ranked_lists_strategy(), top_k=top_k_strategy())
    @settings(max_examples=200)
    def test_output_sorted_descending_by_score(self, data, top_k):
        """Output RRF scores are in non-increasing order.

        **Validates: Requirements 2.3**

        The fused result list must be sorted by RRF score descending.
        """
        dense_results, sparse_results, _, _, _ = data

        result = self.service.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=RRF_K_CONSTANT,
            top_k=top_k,
        )

        # Verify descending order
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score, (
                f"Results not sorted descending at index {i}: "
                f"score[{i}]={result[i].score} < score[{i+1}]={result[i+1].score}"
            )

    @given(data=overlapping_ranked_lists_strategy(), top_k=top_k_strategy())
    @settings(max_examples=200)
    def test_output_bounded_to_top_k(self, data, top_k):
        """Output length is always <= top_k.

        **Validates: Requirements 2.3**

        The fused result must never exceed the specified top_k limit.
        """
        dense_results, sparse_results, _, _, _ = data

        result = self.service.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=RRF_K_CONSTANT,
            top_k=top_k,
        )

        assert len(result) <= top_k, (
            f"Output length {len(result)} exceeds top_k={top_k}"
        )

    @given(data=overlapping_ranked_lists_strategy())
    @settings(max_examples=200)
    def test_overlap_items_score_higher_than_single_list_at_equivalent_ranks(self, data):
        """Items appearing in BOTH lists score higher than items in only one list at same or later rank.

        **Validates: Requirements 2.3**

        For items at equivalent rank positions, an item in both lists should
        always have a higher RRF score than an item appearing in only one list.
        This property is tested by comparing the minimum score of any overlap item
        that appears at rank R in both lists against the maximum score of any
        single-list item at the same rank R in its only list.
        """
        dense_results, sparse_results, overlap_ids, dense_only_ids, sparse_only_ids = data

        # Need at least one overlap item and one single-list item to test
        assume(len(overlap_ids) > 0)
        assume(len(dense_only_ids) > 0 or len(sparse_only_ids) > 0)

        # Use a large top_k to include all items
        total_items = len(overlap_ids) + len(dense_only_ids) + len(sparse_only_ids)
        result = self.service.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=RRF_K_CONSTANT,
            top_k=total_items,
        )

        # Categorize results
        overlap_scores = []
        single_list_scores = []

        for chunk in result:
            if chunk.chunk_id in overlap_ids:
                overlap_scores.append(chunk.score)
            else:
                single_list_scores.append(chunk.score)

        # The minimum overlap score should be > maximum single-list score
        # when items are at equivalent ranks (rank 1 in both vs rank 1 in one).
        #
        # More precisely: an item at rank R in BOTH lists gets score
        # 2/(k+R), while an item at rank R in ONE list gets 1/(k+R).
        # So 2/(k+R) > 1/(k+R') for any R' >= R, meaning overlap items
        # at equivalent or better ranks always outscore single-list items.
        #
        # However, if an overlap item is at a very high rank (near the bottom)
        # in both lists, a single-list item at rank 1 could approach its score.
        # The property we test: for any pair where the overlap item's best rank
        # is <= the single-list item's rank, the overlap item scores higher.
        if overlap_scores and single_list_scores:
            # Build rank info for overlap and single-list items
            overlap_rank_info = {}
            single_rank_info = {}

            for chunk in result:
                best_rank = min(
                    r for r in [chunk.dense_rank, chunk.sparse_rank] if r is not None
                )
                if chunk.chunk_id in overlap_ids:
                    overlap_rank_info[chunk.chunk_id] = (chunk.score, best_rank)
                else:
                    single_rank_info[chunk.chunk_id] = (chunk.score, best_rank)

            # For each single-list item at rank R, any overlap item at rank <= R
            # in BOTH lists should have a higher score
            for s_id, (s_score, s_rank) in single_rank_info.items():
                for o_id, (o_score, o_best_rank) in overlap_rank_info.items():
                    if o_best_rank <= s_rank:
                        assert o_score > s_score, (
                            f"Overlap item '{o_id}' (best_rank={o_best_rank}, score={o_score}) "
                            f"should score higher than single-list item '{s_id}' "
                            f"(rank={s_rank}, score={s_score})"
                        )

    @given(data=overlapping_ranked_lists_strategy(), top_k=top_k_strategy())
    @settings(max_examples=200)
    def test_all_items_from_input_lists(self, data, top_k):
        """Every item in the output came from at least one input list.

        **Validates: Requirements 2.3**

        No phantom items should appear in the RRF output that weren't in
        either the dense or sparse result lists.
        """
        dense_results, sparse_results, overlap_ids, dense_only_ids, sparse_only_ids = data

        result = self.service.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=RRF_K_CONSTANT,
            top_k=top_k,
        )

        # Collect all input chunk_ids
        input_chunk_ids = set(c.chunk_id for c in dense_results) | set(c.chunk_id for c in sparse_results)

        for chunk in result:
            assert chunk.chunk_id in input_chunk_ids, (
                f"Output chunk '{chunk.chunk_id}' not found in either input list"
            )

    @given(data=overlapping_ranked_lists_strategy(), top_k=top_k_strategy())
    @settings(max_examples=200)
    def test_all_scores_in_unit_range(self, data, top_k):
        """All output scores are in [0, 1].

        **Validates: Requirements 2.3**

        RRF scores are normalized to the unit interval by dividing by the
        maximum possible score (2/(k+1)).
        """
        dense_results, sparse_results, _, _, _ = data

        result = self.service.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=RRF_K_CONSTANT,
            top_k=top_k,
        )

        for chunk in result:
            assert 0.0 <= chunk.score <= 1.0, (
                f"Score {chunk.score} for chunk '{chunk.chunk_id}' is outside [0, 1]"
            )

    @given(
        dense_list=scored_chunk_list_strategy(min_size=0, max_size=50),
        sparse_list=scored_chunk_list_strategy(min_size=0, max_size=50),
    )
    @settings(max_examples=100)
    def test_empty_inputs_produce_empty_or_valid_output(self, dense_list, sparse_list):
        """When both inputs are empty, output is empty. When one is empty, output is valid.

        **Validates: Requirements 2.3**

        Edge case: empty input lists should not cause errors and should produce
        appropriately sized output.
        """
        result = self.service.reciprocal_rank_fusion(
            dense_results=dense_list,
            sparse_results=sparse_list,
            k=RRF_K_CONSTANT,
            top_k=DEFAULT_FINAL_TOP_K,
        )

        if not dense_list and not sparse_list:
            assert len(result) == 0, (
                "Both inputs empty but output is non-empty"
            )
        else:
            # Output should be bounded and sorted
            assert len(result) <= DEFAULT_FINAL_TOP_K
            for i in range(len(result) - 1):
                assert result[i].score >= result[i + 1].score
