"""Property-based tests for Evidence Chunk Rendering Completeness.

**Validates: Requirements 15.3**

Property 31: Evidence Chunk Rendering Completeness
- For any list of evidence chunks stored via the audit trail, all chunks are
  represented in the API response with required provenance fields:
  chunk_id, document_id, content_hash, relevance_score, kms_status
- No required field is omitted from any chunk in the response.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from hypothesis import given, settings
from hypothesis import strategies as st

from src.clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    create_frontend_router,
)
from src.clinical_reasoning_fabric.models.core import TraceCategory


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Non-empty identifier strings
identifier_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=40,
)

# SHA-256 content hash (64 hex characters)
content_hash_strategy = st.text(
    alphabet="0123456789abcdef",
    min_size=64,
    max_size=64,
)

# Relevance score between 0.0 and 1.0
relevance_score_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# KMS status: either 'valid' or 'invalid'
kms_status_strategy = st.sampled_from(["valid", "invalid"])

# Chunk index (non-negative integer)
chunk_index_strategy = st.integers(min_value=0, max_value=999)

# ISO-8601 timestamp string
timestamp_strategy = st.datetimes(
    min_value=__import__("datetime").datetime(2020, 1, 1),
    max_value=__import__("datetime").datetime(2030, 12, 31),
).map(lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")

# Chunk text content
chunk_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z", "P")),
    min_size=1,
    max_size=200,
)


@st.composite
def evidence_chunk_strategy(draw):
    """Generate a single valid evidence chunk dictionary."""
    return {
        "chunk_id": draw(identifier_strategy),
        "text": draw(chunk_text_strategy),
        "document_id": draw(identifier_strategy),
        "content_hash": draw(content_hash_strategy),
        "relevance_score": draw(relevance_score_strategy),
        "kms_status": draw(kms_status_strategy),
        "chunk_index": draw(chunk_index_strategy),
        "ingestion_timestamp": draw(timestamp_strategy),
    }


@st.composite
def evidence_chunks_list_strategy(draw):
    """Generate a list of 1-20 evidence chunks with unique chunk_ids."""
    num_chunks = draw(st.integers(min_value=1, max_value=20))
    chunks = []
    used_ids = set()
    for _ in range(num_chunks):
        chunk = draw(evidence_chunk_strategy())
        # Ensure unique chunk_id
        while chunk["chunk_id"] in used_ids:
            chunk["chunk_id"] = draw(identifier_strategy)
        used_ids.add(chunk["chunk_id"])
        chunks.append(chunk)
    return chunks


# =============================================================================
# Helpers
# =============================================================================


def _create_app_and_service():
    """Create a fresh FastAPI app and AuditTrailService for each test."""
    storage = InMemoryAppendOnlyStorage()
    audit_service = AuditTrailService(storage_backend=storage)
    app = FastAPI()
    router = create_frontend_router(audit_service)
    app.include_router(router)
    return app, audit_service


async def _store_chunks_and_query(chunks: list[dict]) -> dict:
    """Store evidence chunks via audit trail and query the API endpoint.

    Returns the API response as a parsed dictionary.
    """
    app, audit_service = _create_app_and_service()
    request_id = "test-request-prop31"

    # Store chunks in audit trail as context_retrieval entry
    await audit_service.record_entry(
        request_id=request_id,
        identity_id="test-user",
        category=TraceCategory.CONTEXT_RETRIEVAL,
        details={"context_planner": "retrieval", "chunks": chunks},
    )

    # Query the API endpoint
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"/api/axisweave/context/{request_id}")
        assert response.status_code == 200
        return response.json()


# =============================================================================
# Property 31: Evidence Chunk Rendering Completeness
# =============================================================================


@pytest.mark.property
class TestEvidenceChunkRenderingCompleteness:
    """Property 31: Evidence Chunk Rendering Completeness.

    **Validates: Requirements 15.3**

    For any list of evidence chunks from the API, all chunks are represented
    in the response with required provenance fields (chunk_id, document_id,
    content_hash, relevance_score, kms_status).
    """

    @given(chunks=evidence_chunks_list_strategy())
    @settings(max_examples=100)
    def test_all_chunks_present_in_response(self, chunks: list[dict]):
        """All stored evidence chunks appear in the API response.

        **Validates: Requirements 15.3**

        For any list of N evidence chunks stored in the audit trail,
        the API response must contain exactly N chunks.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_chunks_and_query(chunks)
        )

        response_chunks = result["chunks"]
        assert len(response_chunks) == len(chunks), (
            f"Expected {len(chunks)} chunks in response, got {len(response_chunks)}"
        )

    @given(chunks=evidence_chunks_list_strategy())
    @settings(max_examples=100)
    def test_chunk_id_present_in_all_response_chunks(self, chunks: list[dict]):
        """Every chunk in the response has a non-empty chunk_id field.

        **Validates: Requirements 15.3**

        The chunk_id field identifies each evidence chunk and must be
        present and non-empty in every response item.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_chunks_and_query(chunks)
        )

        for i, resp_chunk in enumerate(result["chunks"]):
            assert "chunk_id" in resp_chunk, (
                f"chunk[{i}] is missing 'chunk_id' field"
            )
            assert resp_chunk["chunk_id"] is not None and len(resp_chunk["chunk_id"]) > 0, (
                f"chunk[{i}] has empty chunk_id"
            )

    @given(chunks=evidence_chunks_list_strategy())
    @settings(max_examples=100)
    def test_document_id_present_in_all_response_chunks(self, chunks: list[dict]):
        """Every chunk in the response has a non-empty document_id field.

        **Validates: Requirements 15.3**

        The document_id field links the chunk to its source document and
        must be present and non-empty in every response item.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_chunks_and_query(chunks)
        )

        for i, resp_chunk in enumerate(result["chunks"]):
            assert "document_id" in resp_chunk, (
                f"chunk[{i}] is missing 'document_id' field"
            )
            assert resp_chunk["document_id"] is not None and len(resp_chunk["document_id"]) > 0, (
                f"chunk[{i}] has empty document_id"
            )

    @given(chunks=evidence_chunks_list_strategy())
    @settings(max_examples=100)
    def test_content_hash_present_in_all_response_chunks(self, chunks: list[dict]):
        """Every chunk in the response has a non-empty content_hash field.

        **Validates: Requirements 15.3**

        The content_hash provides cryptographic provenance verification
        and must be present and non-empty in every response item.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_chunks_and_query(chunks)
        )

        for i, resp_chunk in enumerate(result["chunks"]):
            assert "content_hash" in resp_chunk, (
                f"chunk[{i}] is missing 'content_hash' field"
            )
            assert resp_chunk["content_hash"] is not None and len(resp_chunk["content_hash"]) > 0, (
                f"chunk[{i}] has empty content_hash"
            )

    @given(chunks=evidence_chunks_list_strategy())
    @settings(max_examples=100)
    def test_relevance_score_present_and_valid_in_all_chunks(self, chunks: list[dict]):
        """Every chunk in the response has a relevance_score between 0.0 and 1.0.

        **Validates: Requirements 15.3**

        The relevance_score indicates the retrieval relevance and must be
        a float in [0.0, 1.0] for every response item.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_chunks_and_query(chunks)
        )

        for i, resp_chunk in enumerate(result["chunks"]):
            assert "relevance_score" in resp_chunk, (
                f"chunk[{i}] is missing 'relevance_score' field"
            )
            score = resp_chunk["relevance_score"]
            assert score is not None, f"chunk[{i}] has null relevance_score"
            assert 0.0 <= score <= 1.0, (
                f"chunk[{i}] relevance_score {score} not in [0.0, 1.0]"
            )

    @given(chunks=evidence_chunks_list_strategy())
    @settings(max_examples=100)
    def test_kms_status_present_and_valid_in_all_chunks(self, chunks: list[dict]):
        """Every chunk in the response has a kms_status of 'valid' or 'invalid'.

        **Validates: Requirements 15.3**

        The kms_status field indicates KMS signature verification result
        and must be either 'valid' or 'invalid' for every response item.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_chunks_and_query(chunks)
        )

        for i, resp_chunk in enumerate(result["chunks"]):
            assert "kms_status" in resp_chunk, (
                f"chunk[{i}] is missing 'kms_status' field"
            )
            assert resp_chunk["kms_status"] in ("valid", "invalid"), (
                f"chunk[{i}] kms_status '{resp_chunk['kms_status']}' not in ('valid', 'invalid')"
            )

    @given(chunks=evidence_chunks_list_strategy())
    @settings(max_examples=100)
    def test_stored_chunk_ids_match_response_chunk_ids(self, chunks: list[dict]):
        """The set of chunk_ids in the response matches the set of stored chunk_ids.

        **Validates: Requirements 15.3**

        Every chunk stored is faithfully represented in the response — no
        chunks are lost or duplicated in transit.
        """
        result = asyncio.get_event_loop().run_until_complete(
            _store_chunks_and_query(chunks)
        )

        stored_ids = {c["chunk_id"] for c in chunks}
        response_ids = {c["chunk_id"] for c in result["chunks"]}
        assert stored_ids == response_ids, (
            f"Chunk ID mismatch. Missing: {stored_ids - response_ids}, "
            f"Extra: {response_ids - stored_ids}"
        )
