"""Unit tests for HybridRetrievalService.

Tests cover:
- Reciprocal Rank Fusion computation correctness
- Timeout handling and degraded mode
- Empty results / no_evidence_found indicator
- Dense search and sparse search result mapping
- KMS signature verification filter

Requirements referenced: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    RetrievalResult,
    ScoredChunk,
    TamperAlert,
)
from clinical_reasoning_fabric.retrieval.hybrid_retrieval_service import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_INDIVIDUAL_TOP_K,
    RRF_K_CONSTANT,
    SEARCH_TIMEOUT_SECONDS,
    HybridRetrievalService,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def make_kms_signature() -> KMSSignature:
    """Create a valid test KMS signature."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789012:key/test-key-id",
        signature="dGVzdC1zaWduYXR1cmU=",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    )


def make_provenance(document_id: str = "doc-001", chunk_index: int = 0) -> ChunkProvenance:
    """Create a valid test ChunkProvenance."""
    return ChunkProvenance(
        document_id=document_id,
        content_hash="a" * 64,
        kms_signature=make_kms_signature(),
        chunk_index=chunk_index,
        ingestion_timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


def make_scored_chunk(
    chunk_id: str,
    text: str = "test chunk text",
    score: float = 0.8,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> ScoredChunk:
    """Create a test ScoredChunk."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        provenance=make_provenance(),
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
    )


def make_qdrant_hit(chunk_id: str, score: float, text: str = "chunk text") -> MagicMock:
    """Create a mock Qdrant search hit."""
    hit = MagicMock()
    hit.id = chunk_id
    hit.score = score
    hit.payload = {
        "text": text,
        "document_id": "doc-001",
        "content_hash": "a" * 64,
        "kms_signature": {
            "key_id": "arn:aws:kms:us-east-1:123456789012:key/test-key-id",
            "signature": "dGVzdC1zaWduYXR1cmU=",
            "algorithm": "RSASSA_PKCS1_V1_5_SHA_256",
            "signed_at": "2024-01-15T10:30:00+00:00",
        },
        "chunk_index": 0,
        "ingestion_timestamp": "2024-01-15T12:00:00+00:00",
    }
    return hit


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.search_sparse = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_kms_client():
    """Create a mock KMS client."""
    client = AsyncMock()
    client.verify_signature = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock()
    model.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return model


@pytest.fixture
def service(mock_qdrant_client, mock_kms_client, mock_embedding_model):
    """Create a HybridRetrievalService instance with mocked dependencies."""
    return HybridRetrievalService(
        qdrant_client=mock_qdrant_client,
        kms_client=mock_kms_client,
        embedding_model=mock_embedding_model,
    )


# =============================================================================
# Tests: Reciprocal Rank Fusion Computation
# =============================================================================


class TestReciprocalRankFusion:
    """Tests for RRF scoring formula correctness."""

    def test_rrf_single_dense_result(self, service):
        """A single item in dense results gets score 1/(k+1)."""
        dense = [make_scored_chunk("chunk-1", dense_rank=1)]
        sparse: list[ScoredChunk] = []

        result = service.reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(result) == 1
        assert result[0].chunk_id == "chunk-1"
        assert result[0].dense_rank == 1
        assert result[0].sparse_rank is None
        # Score should be 1/(60+1) normalized by max_possible=2/(60+1)
        # = (1/61) / (2/61) = 0.5
        assert abs(result[0].score - 0.5) < 1e-6

    def test_rrf_single_sparse_result(self, service):
        """A single item in sparse results gets score 1/(k+1)."""
        dense: list[ScoredChunk] = []
        sparse = [make_scored_chunk("chunk-1", sparse_rank=1)]

        result = service.reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(result) == 1
        assert result[0].chunk_id == "chunk-1"
        assert result[0].dense_rank is None
        assert result[0].sparse_rank == 1
        assert abs(result[0].score - 0.5) < 1e-6

    def test_rrf_item_in_both_lists_scores_higher(self, service):
        """Item appearing in both lists scores higher than item in only one."""
        # chunk-1 appears in both at rank 1
        dense = [
            make_scored_chunk("chunk-1", dense_rank=1),
            make_scored_chunk("chunk-2", dense_rank=2),
        ]
        sparse = [
            make_scored_chunk("chunk-1", sparse_rank=1),
            make_scored_chunk("chunk-3", sparse_rank=2),
        ]

        result = service.reciprocal_rank_fusion(dense, sparse, k=60)

        # chunk-1 should be first (in both lists)
        assert result[0].chunk_id == "chunk-1"
        assert result[0].dense_rank == 1
        assert result[0].sparse_rank == 1
        # Its score should be 2/(k+1) / (2/(k+1)) = 1.0
        assert abs(result[0].score - 1.0) < 1e-6

        # chunk-2 and chunk-3 should have lower scores (only in one list)
        other_chunks = [r for r in result if r.chunk_id != "chunk-1"]
        for chunk in other_chunks:
            assert chunk.score < result[0].score

    def test_rrf_preserves_descending_order(self, service):
        """RRF results are sorted in descending order by score."""
        dense = [make_scored_chunk(f"d-{i}", dense_rank=i) for i in range(1, 6)]
        sparse = [make_scored_chunk(f"s-{i}", sparse_rank=i) for i in range(1, 6)]
        # Overlap: d-1 and s-1 are different chunks, no overlap here

        result = service.reciprocal_rank_fusion(dense, sparse, k=60)

        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_rrf_respects_top_k_limit(self, service):
        """RRF output is bounded to top_k results."""
        dense = [make_scored_chunk(f"d-{i}", dense_rank=i) for i in range(1, 30)]
        sparse = [make_scored_chunk(f"s-{i}", sparse_rank=i) for i in range(1, 30)]

        result = service.reciprocal_rank_fusion(dense, sparse, k=60, top_k=10)

        assert len(result) <= 10

    def test_rrf_default_top_k_is_20(self, service):
        """Default top_k is 20."""
        dense = [make_scored_chunk(f"d-{i}", dense_rank=i) for i in range(1, 51)]
        sparse = [make_scored_chunk(f"s-{i}", sparse_rank=i) for i in range(1, 51)]

        result = service.reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(result) <= DEFAULT_FINAL_TOP_K

    def test_rrf_empty_inputs(self, service):
        """RRF with empty inputs returns empty list."""
        result = service.reciprocal_rank_fusion([], [], k=60)
        assert result == []

    def test_rrf_k_constant_is_60(self, service):
        """Verify the k constant used in RRF formula is 60."""
        assert RRF_K_CONSTANT == 60

    def test_rrf_formula_computation(self, service):
        """Verify RRF formula: score(d) = Σ 1/(k + rank_i(d))."""
        # chunk-A at dense rank 3, sparse rank 5
        dense = [
            make_scored_chunk("chunk-X", dense_rank=1),
            make_scored_chunk("chunk-Y", dense_rank=2),
            make_scored_chunk("chunk-A", dense_rank=3),
        ]
        sparse = [
            make_scored_chunk("chunk-P", sparse_rank=1),
            make_scored_chunk("chunk-Q", sparse_rank=2),
            make_scored_chunk("chunk-R", sparse_rank=3),
            make_scored_chunk("chunk-S", sparse_rank=4),
            make_scored_chunk("chunk-A", sparse_rank=5),
        ]

        result = service.reciprocal_rank_fusion(dense, sparse, k=60)

        # Find chunk-A in results
        chunk_a = next(r for r in result if r.chunk_id == "chunk-A")

        # Expected raw RRF score: 1/(60+3) + 1/(60+5) = 1/63 + 1/65
        expected_raw = 1.0 / 63 + 1.0 / 65
        max_possible = 2.0 / 61  # max when rank 1 in both
        expected_normalized = expected_raw / max_possible

        assert abs(chunk_a.score - expected_normalized) < 1e-6
        assert chunk_a.dense_rank == 3
        assert chunk_a.sparse_rank == 5


# =============================================================================
# Tests: Timeout Handling and Degraded Mode
# =============================================================================


class TestTimeoutHandling:
    """Tests for search timeout handling (10s) and degraded mode."""

    @pytest.mark.asyncio
    async def test_dense_timeout_proceeds_with_sparse(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When dense search times out, proceed with sparse results."""

        async def slow_search(*args, **kwargs):
            await asyncio.sleep(20)  # Exceeds 10s timeout
            return []

        mock_qdrant_client.search = slow_search
        mock_qdrant_client.search_sparse = AsyncMock(
            return_value=[make_qdrant_hit("sparse-1", 0.9)]
        )

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.degraded_search is True
        assert len(result.verified_chunks) > 0

    @pytest.mark.asyncio
    async def test_sparse_timeout_proceeds_with_dense(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When sparse search times out, proceed with dense results."""

        async def slow_sparse(*args, **kwargs):
            await asyncio.sleep(20)  # Exceeds 10s timeout
            return []

        mock_qdrant_client.search = AsyncMock(
            return_value=[make_qdrant_hit("dense-1", 0.9)]
        )
        mock_qdrant_client.search_sparse = slow_sparse

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.degraded_search is True
        assert len(result.verified_chunks) > 0

    @pytest.mark.asyncio
    async def test_both_searches_succeed_not_degraded(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When both searches succeed, degraded_search is False."""
        mock_qdrant_client.search = AsyncMock(
            return_value=[make_qdrant_hit("dense-1", 0.9)]
        )
        mock_qdrant_client.search_sparse = AsyncMock(
            return_value=[make_qdrant_hit("sparse-1", 0.8)]
        )

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.degraded_search is False
        assert len(result.verified_chunks) > 0

    @pytest.mark.asyncio
    async def test_dense_exception_triggers_degraded(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When dense search raises an exception, proceed degraded."""
        mock_qdrant_client.search = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )
        mock_qdrant_client.search_sparse = AsyncMock(
            return_value=[make_qdrant_hit("sparse-1", 0.8)]
        )

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.degraded_search is True
        assert len(result.verified_chunks) > 0

    @pytest.mark.asyncio
    async def test_sparse_exception_triggers_degraded(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When sparse search raises an exception, proceed degraded."""
        mock_qdrant_client.search = AsyncMock(
            return_value=[make_qdrant_hit("dense-1", 0.9)]
        )
        mock_qdrant_client.search_sparse = AsyncMock(
            side_effect=RuntimeError("service unavailable")
        )

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.degraded_search is True
        assert len(result.verified_chunks) > 0

    @pytest.mark.asyncio
    async def test_timeout_constant_is_10_seconds(self):
        """Verify the timeout constant is 10 seconds per spec."""
        assert SEARCH_TIMEOUT_SECONDS == 10


# =============================================================================
# Tests: No Evidence Found
# =============================================================================


class TestNoEvidenceFound:
    """Tests for empty results and no_evidence_found indicator."""

    @pytest.mark.asyncio
    async def test_both_empty_returns_no_evidence_found(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When both searches return zero matches, set no_evidence_found=True."""
        mock_qdrant_client.search = AsyncMock(return_value=[])
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.no_evidence_found is True
        assert result.verified_chunks == []
        assert result.total_candidates == 0

    @pytest.mark.asyncio
    async def test_both_timeout_returns_no_evidence_found(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When both searches time out, return empty with no_evidence_found."""

        async def slow_search(*args, **kwargs):
            await asyncio.sleep(20)
            return []

        mock_qdrant_client.search = slow_search
        mock_qdrant_client.search_sparse = slow_search

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.no_evidence_found is True
        assert result.degraded_search is True

    @pytest.mark.asyncio
    async def test_results_present_no_evidence_found_false(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When results are found, no_evidence_found is False."""
        mock_qdrant_client.search = AsyncMock(
            return_value=[make_qdrant_hit("chunk-1", 0.9)]
        )
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.no_evidence_found is False
        assert len(result.verified_chunks) > 0


# =============================================================================
# Tests: Dense Search
# =============================================================================


class TestDenseSearch:
    """Tests for dense vector search functionality."""

    @pytest.mark.asyncio
    async def test_dense_search_returns_scored_chunks(self, service, mock_qdrant_client):
        """Dense search converts Qdrant hits to ScoredChunks."""
        hits = [make_qdrant_hit(f"chunk-{i}", 0.9 - i * 0.1) for i in range(3)]
        mock_qdrant_client.search = AsyncMock(return_value=hits)

        result = await service.dense_search([0.1] * 1024, top_k=50)

        assert len(result) == 3
        for i, chunk in enumerate(result, start=1):
            assert chunk.dense_rank == i
            assert chunk.sparse_rank is None

    @pytest.mark.asyncio
    async def test_dense_search_uses_correct_top_k(self, service, mock_qdrant_client):
        """Dense search passes the correct limit to Qdrant."""
        mock_qdrant_client.search = AsyncMock(return_value=[])

        await service.dense_search([0.1] * 1024, top_k=50)

        mock_qdrant_client.search.assert_called_once_with(
            collection_name="clinical_documents",
            query_vector=[0.1] * 1024,
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_dense_search_default_top_k_is_50(self):
        """Default individual search top_k is 50."""
        assert DEFAULT_INDIVIDUAL_TOP_K == 50


# =============================================================================
# Tests: Sparse Search
# =============================================================================


class TestSparseSearch:
    """Tests for BM25 sparse index search functionality."""

    @pytest.mark.asyncio
    async def test_sparse_search_returns_scored_chunks(self, service, mock_qdrant_client):
        """Sparse search converts Qdrant hits to ScoredChunks."""
        hits = [make_qdrant_hit(f"chunk-{i}", 0.8 - i * 0.1) for i in range(3)]
        mock_qdrant_client.search_sparse = AsyncMock(return_value=hits)

        result = await service.sparse_search(["test", "query"], top_k=50)

        assert len(result) == 3
        for i, chunk in enumerate(result, start=1):
            assert chunk.sparse_rank == i
            assert chunk.dense_rank is None

    @pytest.mark.asyncio
    async def test_sparse_search_uses_correct_params(self, service, mock_qdrant_client):
        """Sparse search passes correct terms and limit to Qdrant."""
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])

        await service.sparse_search(["diabetes", "insulin"], top_k=50)

        mock_qdrant_client.search_sparse.assert_called_once_with(
            collection_name="clinical_documents",
            query_terms=["diabetes", "insulin"],
            limit=50,
        )


# =============================================================================
# Tests: Query Tokenization
# =============================================================================


class TestQueryTokenization:
    """Tests for query tokenization helper."""

    def test_tokenize_basic_query(self, service):
        """Tokenizes a basic query into lowercase terms."""
        result = service._tokenize_query("Patient Diabetes Type 2")
        assert result == ["patient", "diabetes", "type", "2"]

    def test_tokenize_empty_query(self, service):
        """Empty query produces empty terms list."""
        result = service._tokenize_query("")
        assert result == []

    def test_tokenize_strips_whitespace(self, service):
        """Extra whitespace is handled gracefully."""
        result = service._tokenize_query("  hello   world  ")
        assert result == ["hello", "world"]


# =============================================================================
# Tests: Full Retrieve Flow
# =============================================================================


class TestRetrieveFlow:
    """Tests for the full retrieve() orchestration."""

    @pytest.mark.asyncio
    async def test_retrieve_calls_embedding_model(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """Retrieve generates an embedding for the query."""
        mock_qdrant_client.search = AsyncMock(return_value=[])
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        await service.retrieve("test query")

        mock_embedding_model.embed_query.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieval_result(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """Retrieve returns a properly structured RetrievalResult."""
        mock_qdrant_client.search = AsyncMock(
            return_value=[make_qdrant_hit("chunk-1", 0.9)]
        )
        mock_qdrant_client.search_sparse = AsyncMock(
            return_value=[make_qdrant_hit("chunk-2", 0.8)]
        )

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert isinstance(result, RetrievalResult)
        assert result.no_evidence_found is False
        assert result.degraded_search is False
        assert len(result.verified_chunks) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k_parameter(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """Retrieve limits final output to top_k parameter."""
        hits = [make_qdrant_hit(f"chunk-{i}", 0.9 - i * 0.01) for i in range(30)]
        mock_qdrant_client.search = AsyncMock(return_value=hits)
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query", top_k=5)

        assert len(result.verified_chunks) <= 5

    @pytest.mark.asyncio
    async def test_retrieve_tracks_total_candidates(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """Retrieve tracks total candidates from both search methods."""
        dense_hits = [make_qdrant_hit(f"d-{i}", 0.9) for i in range(5)]
        sparse_hits = [make_qdrant_hit(f"s-{i}", 0.8) for i in range(3)]
        mock_qdrant_client.search = AsyncMock(return_value=dense_hits)
        mock_qdrant_client.search_sparse = AsyncMock(return_value=sparse_hits)

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.total_candidates == 8



# =============================================================================
# Tests: KMS Signature Verification Filter
# =============================================================================


class TestVerifySignatures:
    """Tests for KMS signature verification filter (Requirements 2.4, 2.5)."""

    @pytest.mark.asyncio
    async def test_valid_signatures_pass_through(self, service, mock_kms_client):
        """Chunks with valid KMS signatures are included in verified results."""
        mock_kms_client.verify_signature = AsyncMock(return_value=True)

        chunks = [
            make_scored_chunk("chunk-1", text="valid chunk 1"),
            make_scored_chunk("chunk-2", text="valid chunk 2"),
            make_scored_chunk("chunk-3", text="valid chunk 3"),
        ]

        verified, alerts = await service.verify_signatures(chunks)

        assert len(verified) == 3
        assert len(alerts) == 0
        assert [c.chunk_id for c in verified] == ["chunk-1", "chunk-2", "chunk-3"]

    @pytest.mark.asyncio
    async def test_invalid_signatures_excluded_with_tamper_alert(
        self, service, mock_kms_client
    ):
        """Chunks with invalid KMS signatures are excluded and produce tamper alerts."""
        mock_kms_client.verify_signature = AsyncMock(return_value=False)

        chunks = [
            make_scored_chunk("chunk-1", text="tampered chunk 1"),
            make_scored_chunk("chunk-2", text="tampered chunk 2"),
        ]

        verified, alerts = await service.verify_signatures(chunks)

        assert len(verified) == 0
        assert len(alerts) == 2
        assert alerts[0].chunk_id == "chunk-1"
        assert alerts[0].reason == "KMS signature verification failed"
        assert alerts[0].expected_signature == "dGVzdC1zaWduYXR1cmU="
        assert alerts[1].chunk_id == "chunk-2"
        assert alerts[1].reason == "KMS signature verification failed"

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_signatures(self, service, mock_kms_client):
        """Mix of valid and invalid signatures: only valid are returned."""
        # First call returns True, second returns False, third returns True
        mock_kms_client.verify_signature = AsyncMock(
            side_effect=[True, False, True]
        )

        chunks = [
            make_scored_chunk("chunk-valid-1", text="valid"),
            make_scored_chunk("chunk-invalid", text="invalid"),
            make_scored_chunk("chunk-valid-2", text="valid too"),
        ]

        verified, alerts = await service.verify_signatures(chunks)

        assert len(verified) == 2
        assert len(alerts) == 1
        assert verified[0].chunk_id == "chunk-valid-1"
        assert verified[1].chunk_id == "chunk-valid-2"
        assert alerts[0].chunk_id == "chunk-invalid"

    @pytest.mark.asyncio
    async def test_kms_verification_exception_treated_as_invalid(
        self, service, mock_kms_client
    ):
        """When KMS client raises an exception, chunk is treated as invalid."""
        mock_kms_client.verify_signature = AsyncMock(
            side_effect=RuntimeError("KMS service unavailable")
        )

        chunks = [make_scored_chunk("chunk-error", text="error chunk")]

        verified, alerts = await service.verify_signatures(chunks)

        assert len(verified) == 0
        assert len(alerts) == 1
        assert alerts[0].chunk_id == "chunk-error"
        assert alerts[0].reason == "KMS signature verification failed"

    @pytest.mark.asyncio
    async def test_tamper_alert_contains_required_fields(
        self, service, mock_kms_client
    ):
        """Tamper alerts contain all required fields: chunk_id, document_id, content_hash, reason, detected_at."""
        mock_kms_client.verify_signature = AsyncMock(return_value=False)

        chunks = [make_scored_chunk("chunk-1")]

        _, alerts = await service.verify_signatures(chunks)

        alert = alerts[0]
        assert alert.chunk_id == "chunk-1"
        assert alert.document_id == "doc-001"
        assert alert.content_hash == "a" * 64
        assert alert.expected_signature is not None
        assert alert.reason != ""
        assert alert.detected_at is not None

    @pytest.mark.asyncio
    async def test_verify_signatures_calls_kms_with_correct_params(
        self, service, mock_kms_client
    ):
        """verify_signatures passes content_hash, signature, and key_id to KMS client."""
        mock_kms_client.verify_signature = AsyncMock(return_value=True)

        chunks = [make_scored_chunk("chunk-1")]

        await service.verify_signatures(chunks)

        mock_kms_client.verify_signature.assert_called_once_with(
            content_hash="a" * 64,
            signature="dGVzdC1zaWduYXR1cmU=",
            key_id="arn:aws:kms:us-east-1:123456789012:key/test-key-id",
        )

    @pytest.mark.asyncio
    async def test_empty_chunks_list_returns_empty(self, service):
        """Verifying an empty list returns empty verified and alerts."""
        verified, alerts = await service.verify_signatures([])

        assert verified == []
        assert alerts == []

    @pytest.mark.asyncio
    async def test_verify_signatures_logs_tamper_alerts(
        self, service, mock_kms_client, caplog
    ):
        """Tamper alerts are logged to the observability layer (logging)."""
        mock_kms_client.verify_signature = AsyncMock(return_value=False)
        chunks = [make_scored_chunk("chunk-1")]

        with caplog.at_level(logging.WARNING):
            await service.verify_signatures(chunks)

        assert any("Tamper alert" in record.message for record in caplog.records)
        assert any("chunk-1" in record.message for record in caplog.records)


class TestRetrieveWithSignatureVerification:
    """Tests for retrieve() integration with verify_signatures()."""

    @pytest.mark.asyncio
    async def test_retrieve_filters_invalid_signatures(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """retrieve() excludes chunks with invalid signatures from results."""
        # Return 2 hits from dense search
        hits = [
            make_qdrant_hit("chunk-valid", 0.9),
            make_qdrant_hit("chunk-invalid", 0.8),
        ]
        mock_qdrant_client.search = AsyncMock(return_value=hits)
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])
        # First signature valid, second invalid
        mock_kms_client.verify_signature = AsyncMock(side_effect=[True, False])

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert len(result.verified_chunks) == 1
        assert result.verified_chunks[0].chunk_id == "chunk-valid"
        assert len(result.tamper_alerts) == 1
        assert result.tamper_alerts[0].chunk_id == "chunk-invalid"

    @pytest.mark.asyncio
    async def test_retrieve_all_invalid_signatures_marks_no_evidence(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """When all chunks fail verification, result shows no_evidence_found."""
        hits = [make_qdrant_hit("chunk-1", 0.9)]
        mock_qdrant_client.search = AsyncMock(return_value=hits)
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])
        mock_kms_client.verify_signature = AsyncMock(return_value=False)

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert result.no_evidence_found is True
        assert len(result.verified_chunks) == 0
        assert len(result.tamper_alerts) == 1

    @pytest.mark.asyncio
    async def test_retrieve_includes_tamper_alerts_in_result(
        self, mock_qdrant_client, mock_kms_client, mock_embedding_model
    ):
        """retrieve() returns tamper alerts in the RetrievalResult."""
        hits = [
            make_qdrant_hit("chunk-1", 0.9),
            make_qdrant_hit("chunk-2", 0.8),
            make_qdrant_hit("chunk-3", 0.7),
        ]
        mock_qdrant_client.search = AsyncMock(return_value=hits)
        mock_qdrant_client.search_sparse = AsyncMock(return_value=[])
        # Only first is valid
        mock_kms_client.verify_signature = AsyncMock(
            side_effect=[True, False, False]
        )

        service = HybridRetrievalService(
            mock_qdrant_client, mock_kms_client, mock_embedding_model
        )
        result = await service.retrieve("test query")

        assert len(result.verified_chunks) == 1
        assert len(result.tamper_alerts) == 2
        alert_ids = [a.chunk_id for a in result.tamper_alerts]
        assert "chunk-2" in alert_ids
        assert "chunk-3" in alert_ids
