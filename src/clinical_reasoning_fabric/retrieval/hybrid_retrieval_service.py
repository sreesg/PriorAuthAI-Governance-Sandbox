"""Hybrid Retrieval Service combining dense vector search and BM25 sparse search.

Implements Reciprocal Rank Fusion (RRF) to merge results from both retrieval
methods. Handles search timeouts gracefully, proceeding with available results
when one method fails. Verifies KMS signatures on all chunks before returning.

Requirements referenced: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    RetrievalResult,
    ScoredChunk,
    TamperAlert,
)

logger = logging.getLogger(__name__)

# Constants
SEARCH_TIMEOUT_SECONDS = 10
DEFAULT_INDIVIDUAL_TOP_K = 50
DEFAULT_FINAL_TOP_K = 20
RRF_K_CONSTANT = 60


class EmbeddingModel(Protocol):
    """Protocol for embedding model that generates query vectors."""

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a query string."""
        ...


class QdrantClientProtocol(Protocol):
    """Protocol for Qdrant client operations."""

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute dense vector search."""
        ...

    async def search_sparse(
        self,
        collection_name: str,
        query_terms: list[str],
        limit: int,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute BM25 sparse index search."""
        ...


class KMSClientProtocol(Protocol):
    """Protocol for KMS signature verification."""

    async def verify_signature(
        self, content_hash: str, signature: str, key_id: str
    ) -> bool:
        """Verify a KMS signature against a content hash."""
        ...


class HybridRetrievalService:
    """Executes hybrid dense+BM25 retrieval with signature verification.

    Combines dense vector cosine similarity search and BM25 sparse index search
    using Reciprocal Rank Fusion (RRF) scoring. Verifies KMS signatures on all
    results before returning.

    RRF formula: score(d) = Σ 1/(k + rank_i(d)) where k=60, rank is 1-indexed.
    Items appearing in both lists receive higher combined scores.

    Requirements:
        2.1: Dense vector similarity search returning top 50 by cosine similarity
        2.2: BM25 sparse index search returning top 50 by BM25 score
        2.3: Reciprocal Rank Fusion combination, top 20 output
        2.6: 10-second timeout; proceed with available method on failure
        2.7: Return empty result with no_evidence_found if both return zero
    """

    def __init__(
        self,
        qdrant_client: Any,
        kms_client: Any,
        embedding_model: Any,
        collection_name: str = "clinical_documents",
    ):
        """Initialize HybridRetrievalService.

        Args:
            qdrant_client: Client for Qdrant vector database operations.
            kms_client: Client for KMS signature verification.
            embedding_model: Model for generating query embeddings.
            collection_name: Name of the Qdrant collection to search.
        """
        self.qdrant = qdrant_client
        self.kms = kms_client
        self.embedder = embedding_model
        self.collection_name = collection_name

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
        min_score: float = 0.5,
    ) -> RetrievalResult:
        """Execute hybrid retrieval: dense top-50 + BM25 top-50 → RRF → top-k.

        Runs dense and sparse searches concurrently with a 10-second timeout
        per method. If one method fails/times out, proceeds with available
        results and sets degraded_search=True.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return (default 20).
            min_score: Minimum RRF score threshold for inclusion.

        Returns:
            RetrievalResult with verified chunks, tamper alerts, and status flags.
        """
        # Generate embedding for dense search
        query_embedding = await self.embedder.embed_query(query)

        # Tokenize query for sparse search
        query_terms = self._tokenize_query(query)

        # Execute both searches concurrently with timeout handling
        dense_results: list[ScoredChunk] = []
        sparse_results: list[ScoredChunk] = []
        degraded_search = False

        # Run dense search with timeout
        dense_task = asyncio.create_task(
            self.dense_search(query_embedding, top_k=DEFAULT_INDIVIDUAL_TOP_K)
        )
        # Run sparse search with timeout
        sparse_task = asyncio.create_task(
            self.sparse_search(query_terms, top_k=DEFAULT_INDIVIDUAL_TOP_K)
        )

        # Gather dense results with timeout
        try:
            dense_results = await asyncio.wait_for(
                dense_task, timeout=SEARCH_TIMEOUT_SECONDS
            )
        except (asyncio.TimeoutError, Exception) as e:
            degraded_search = True
            logger.warning(
                "Dense search failed or timed out, proceeding with sparse results only: %s",
                str(e),
            )
            dense_task.cancel()

        # Gather sparse results with timeout
        try:
            sparse_results = await asyncio.wait_for(
                sparse_task, timeout=SEARCH_TIMEOUT_SECONDS
            )
        except (asyncio.TimeoutError, Exception) as e:
            degraded_search = True
            logger.warning(
                "Sparse search failed or timed out, proceeding with dense results only: %s",
                str(e),
            )
            sparse_task.cancel()

        # Apply Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(
            dense_results, sparse_results, k=RRF_K_CONSTANT
        )

        # Limit to top_k results
        top_results = fused_results[:top_k]

        # Check for no evidence
        if not top_results:
            return RetrievalResult(
                verified_chunks=[],
                tamper_alerts=[],
                no_evidence_found=True,
                degraded_search=degraded_search,
                total_candidates=0,
            )

        total_candidates = len(dense_results) + len(sparse_results)

        # Verify KMS signatures on all results before returning
        verified_chunks, tamper_alerts = await self.verify_signatures(top_results)

        # If all chunks were excluded by signature verification, mark no evidence
        no_evidence = len(verified_chunks) == 0

        return RetrievalResult(
            verified_chunks=verified_chunks,
            tamper_alerts=tamper_alerts,
            no_evidence_found=no_evidence,
            degraded_search=degraded_search,
            total_candidates=total_candidates,
        )

    async def dense_search(
        self, query_embedding: list[float], top_k: int = DEFAULT_INDIVIDUAL_TOP_K
    ) -> list[ScoredChunk]:
        """Execute dense vector cosine similarity search.

        Queries Qdrant with the embedding vector and returns the top results
        ranked by cosine similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Maximum number of results to return (default 50).

        Returns:
            List of ScoredChunk objects with dense_rank populated.
        """
        results = await self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        scored_chunks: list[ScoredChunk] = []
        for rank, hit in enumerate(results, start=1):
            chunk = self._hit_to_scored_chunk(hit, dense_rank=rank)
            scored_chunks.append(chunk)

        return scored_chunks

    async def sparse_search(
        self, query_terms: list[str], top_k: int = DEFAULT_INDIVIDUAL_TOP_K
    ) -> list[ScoredChunk]:
        """Execute BM25 sparse index search.

        Queries Qdrant's BM25 index with the query terms and returns
        the top results ranked by BM25 score.

        Args:
            query_terms: Tokenized query terms for BM25 matching.
            top_k: Maximum number of results to return (default 50).

        Returns:
            List of ScoredChunk objects with sparse_rank populated.
        """
        results = await self.qdrant.search_sparse(
            collection_name=self.collection_name,
            query_terms=query_terms,
            limit=top_k,
        )

        scored_chunks: list[ScoredChunk] = []
        for rank, hit in enumerate(results, start=1):
            chunk = self._hit_to_scored_chunk(hit, sparse_rank=rank)
            scored_chunks.append(chunk)

        return scored_chunks

    def reciprocal_rank_fusion(
        self,
        dense_results: list[ScoredChunk],
        sparse_results: list[ScoredChunk],
        k: int = RRF_K_CONSTANT,
        top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> list[ScoredChunk]:
        """Combine dense and sparse results using Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ 1/(k + rank_i(d))
        where k=60 (default) and rank is 1-indexed.

        Items appearing in both dense and sparse results receive contributions
        from both rankings, giving them higher combined scores than items
        appearing in only one list at equivalent ranks.

        Args:
            dense_results: Results from dense vector search with dense_rank set.
            sparse_results: Results from BM25 sparse search with sparse_rank set.
            k: RRF constant (default 60).
            top_k: Maximum results to return (default 20).

        Returns:
            Combined list of ScoredChunk sorted descending by RRF score,
            limited to top_k.
        """
        # Build a map of chunk_id -> aggregated RRF score and chunk data
        chunk_map: dict[str, dict[str, Any]] = {}

        # Process dense results
        for rank, chunk in enumerate(dense_results, start=1):
            rrf_score = 1.0 / (k + rank)
            if chunk.chunk_id in chunk_map:
                chunk_map[chunk.chunk_id]["rrf_score"] += rrf_score
                chunk_map[chunk.chunk_id]["dense_rank"] = rank
            else:
                chunk_map[chunk.chunk_id] = {
                    "rrf_score": rrf_score,
                    "chunk": chunk,
                    "dense_rank": rank,
                    "sparse_rank": None,
                }

        # Process sparse results
        for rank, chunk in enumerate(sparse_results, start=1):
            rrf_score = 1.0 / (k + rank)
            if chunk.chunk_id in chunk_map:
                chunk_map[chunk.chunk_id]["rrf_score"] += rrf_score
                chunk_map[chunk.chunk_id]["sparse_rank"] = rank
            else:
                chunk_map[chunk.chunk_id] = {
                    "rrf_score": rrf_score,
                    "chunk": chunk,
                    "dense_rank": None,
                    "sparse_rank": rank,
                }

        # Sort by RRF score descending
        sorted_items = sorted(
            chunk_map.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        # Build final ScoredChunk list with normalized RRF scores
        # Max possible RRF score is 2/(k+1) when an item is rank 1 in both lists
        max_possible_score = 2.0 / (k + 1)

        fused_chunks: list[ScoredChunk] = []
        for item in sorted_items[:top_k]:
            chunk = item["chunk"]
            # Normalize score to [0, 1] range
            normalized_score = min(item["rrf_score"] / max_possible_score, 1.0)
            fused_chunk = ScoredChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=normalized_score,
                provenance=chunk.provenance,
                dense_rank=item["dense_rank"],
                sparse_rank=item["sparse_rank"],
            )
            fused_chunks.append(fused_chunk)

        return fused_chunks

    async def verify_signatures(
        self, chunks: list[ScoredChunk]
    ) -> tuple[list[ScoredChunk], list[TamperAlert]]:
        """Verify KMS signatures on chunks, excluding invalid/missing ones.

        For each chunk, verifies the KMS signature against the content_hash
        using the KMS client. Chunks with valid signatures are included in
        the verified results. Chunks with invalid or missing signatures are
        excluded and produce TamperAlert entries logged to the observability layer.

        Args:
            chunks: List of ScoredChunk objects to verify.

        Returns:
            A tuple of (verified_chunks, tamper_alerts) where verified_chunks
            contains only chunks with valid signatures, and tamper_alerts
            contains alerts for each excluded chunk.

        Requirements:
            2.4: Return only chunks whose KMS signatures are verified as valid
            2.5: Exclude failed chunks and log tamper alerts to observability
        """
        verified_chunks: list[ScoredChunk] = []
        tamper_alerts: list[TamperAlert] = []

        for chunk in chunks:
            provenance = chunk.provenance

            # Check for missing signature
            if not provenance.kms_signature:
                alert = TamperAlert(
                    chunk_id=chunk.chunk_id,
                    document_id=provenance.document_id,
                    content_hash=provenance.content_hash,
                    expected_signature=None,
                    reason="Missing KMS signature",
                    detected_at=datetime.now(timezone.utc),
                )
                tamper_alerts.append(alert)
                logger.warning(
                    "Tamper alert: chunk %s from document %s has missing KMS signature",
                    chunk.chunk_id,
                    provenance.document_id,
                )
                continue

            # Verify the signature against the content hash
            try:
                is_valid = await self.kms.verify_signature(
                    content_hash=provenance.content_hash,
                    signature=provenance.kms_signature.signature,
                    key_id=provenance.kms_signature.key_id,
                )
            except Exception as e:
                # Treat verification errors as invalid signatures
                is_valid = False
                logger.error(
                    "KMS signature verification error for chunk %s: %s",
                    chunk.chunk_id,
                    str(e),
                )

            if is_valid:
                verified_chunks.append(chunk)
            else:
                alert = TamperAlert(
                    chunk_id=chunk.chunk_id,
                    document_id=provenance.document_id,
                    content_hash=provenance.content_hash,
                    expected_signature=provenance.kms_signature.signature,
                    reason="KMS signature verification failed",
                    detected_at=datetime.now(timezone.utc),
                )
                tamper_alerts.append(alert)
                logger.warning(
                    "Tamper alert: chunk %s from document %s failed KMS signature verification",
                    chunk.chunk_id,
                    provenance.document_id,
                )

        return verified_chunks, tamper_alerts

    def _tokenize_query(self, query: str) -> list[str]:
        """Simple whitespace tokenization for BM25 query terms.

        Args:
            query: The query string to tokenize.

        Returns:
            List of lowercase query terms.
        """
        return [term.lower().strip() for term in query.split() if term.strip()]

    def _hit_to_scored_chunk(
        self,
        hit: Any,
        dense_rank: int | None = None,
        sparse_rank: int | None = None,
    ) -> ScoredChunk:
        """Convert a Qdrant search hit to a ScoredChunk.

        Args:
            hit: The search result from Qdrant (expected to have id, payload, score).
            dense_rank: The rank in dense search results (1-indexed).
            sparse_rank: The rank in sparse search results (1-indexed).

        Returns:
            A ScoredChunk with provenance extracted from the hit payload.
        """
        payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
        score = hit.score if hasattr(hit, "score") else hit.get("score", 0.0)
        chunk_id = str(hit.id if hasattr(hit, "id") else hit.get("id", ""))

        provenance = ChunkProvenance(
            document_id=payload.get("document_id", ""),
            content_hash=payload.get("content_hash", ""),
            kms_signature=payload.get("kms_signature", {}),
            chunk_index=payload.get("chunk_index", 0),
            ingestion_timestamp=payload.get("ingestion_timestamp", "2024-01-01T00:00:00Z"),
        )

        return ScoredChunk(
            chunk_id=chunk_id,
            text=payload.get("text", ""),
            score=min(score, 1.0),
            provenance=provenance,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
        )
