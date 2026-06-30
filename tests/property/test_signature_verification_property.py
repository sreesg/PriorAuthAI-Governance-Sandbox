"""Property-based tests for Signature Verification Filter.

**Validates: Requirements 2.4, 7.1, 11.2**

Property 5: Signature Verification Filter
- For any set of chunks with mixed valid/invalid signatures, the filter
  includes exactly valid-signature chunks and excludes exactly invalid ones,
  producing tamper alerts for each exclusion.
- The number of verified chunks equals the number of valid-signature chunks.
- The number of tamper alerts equals the number of invalid-signature chunks.
- Every verified chunk_id is in the set of valid chunk_ids.
- Every tamper alert chunk_id is in the set of invalid chunk_ids.
- No overlap between verified chunk_ids and alert chunk_ids.
- verified_count + alert_count = input_count (partition invariant).
"""

import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    ScoredChunk,
    TamperAlert,
)
from clinical_reasoning_fabric.retrieval.hybrid_retrieval_service import (
    HybridRetrievalService,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_content_hash(chunk_id: str) -> str:
    """Create a valid 64-char hex content hash from a chunk_id."""
    return hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()


def _make_kms_signature(chunk_id: str) -> KMSSignature:
    """Create a valid KMSSignature for a given chunk."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789012:key/test-key-id",
        signature=f"sig-{chunk_id}-base64encoded",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_scored_chunk(chunk_id: str) -> ScoredChunk:
    """Create a valid ScoredChunk with the given chunk_id."""
    content_hash = _make_content_hash(chunk_id)
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"Clinical note content for chunk {chunk_id}",
        score=0.75,
        provenance=ChunkProvenance(
            document_id=f"doc-{chunk_id}",
            content_hash=content_hash,
            kms_signature=_make_kms_signature(chunk_id),
            chunk_index=0,
            ingestion_timestamp=datetime.now(timezone.utc),
        ),
        dense_rank=1,
        sparse_rank=None,
    )


def _make_mock_kms_client(valid_chunk_ids: set[str], chunks: list[ScoredChunk]):
    """Create a mock KMS client that returns True for valid chunks, False for invalid.

    The mock determines validity based on the content_hash: if the chunk_id
    corresponding to a content_hash is in the valid_chunk_ids set, returns True.
    """
    # Build a mapping from content_hash to chunk_id for lookup
    hash_to_chunk_id = {
        chunk.provenance.content_hash: chunk.chunk_id for chunk in chunks
    }

    async def mock_verify_signature(content_hash: str, signature: str, key_id: str) -> bool:
        chunk_id = hash_to_chunk_id.get(content_hash)
        if chunk_id is None:
            return False
        return chunk_id in valid_chunk_ids

    kms_client = AsyncMock()
    kms_client.verify_signature = AsyncMock(side_effect=mock_verify_signature)
    return kms_client


def _make_service_with_kms(kms_client) -> HybridRetrievalService:
    """Create a HybridRetrievalService with a mock KMS client."""
    return HybridRetrievalService(
        qdrant_client=None,
        kms_client=kms_client,
        embedding_model=None,
        collection_name="test_collection",
    )


# =============================================================================
# Hypothesis strategies
# =============================================================================


@st.composite
def chunks_with_validity_mask_strategy(draw):
    """Generate a list of ScoredChunks with a boolean mask indicating validity.

    Returns (chunks, valid_chunk_ids, invalid_chunk_ids) where:
    - chunks is a list of 1-20 ScoredChunks with unique chunk_ids
    - valid_chunk_ids is the set of chunk_ids that should pass verification
    - invalid_chunk_ids is the set that should fail verification
    """
    size = draw(st.integers(min_value=1, max_value=20))
    chunk_ids = [f"chunk-{i:04d}" for i in range(size)]
    chunks = [_make_scored_chunk(cid) for cid in chunk_ids]

    # Generate a boolean mask: True = valid signature, False = invalid
    validity_mask = draw(st.lists(st.booleans(), min_size=size, max_size=size))

    valid_chunk_ids = {cid for cid, valid in zip(chunk_ids, validity_mask) if valid}
    invalid_chunk_ids = {cid for cid, valid in zip(chunk_ids, validity_mask) if not valid}

    return chunks, valid_chunk_ids, invalid_chunk_ids


# =============================================================================
# Property tests
# =============================================================================


@pytest.mark.property
class TestSignatureVerificationFilter:
    """Property 5: Signature Verification Filter.

    **Validates: Requirements 2.4, 7.1, 11.2**

    Tests that the verify_signatures method correctly partitions chunks into
    verified (valid signature) and excluded (invalid signature with tamper alert)
    groups, maintaining the partition invariant.
    """

    @given(data=chunks_with_validity_mask_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_verified_count_equals_valid_signature_count(self, data):
        """The number of verified chunks equals the number of valid-signature chunks.

        **Validates: Requirements 2.4**
        """
        chunks, valid_chunk_ids, invalid_chunk_ids = data

        kms_client = _make_mock_kms_client(valid_chunk_ids, chunks)
        service = _make_service_with_kms(kms_client)

        verified, alerts = await service.verify_signatures(chunks)

        assert len(verified) == len(valid_chunk_ids), (
            f"Expected {len(valid_chunk_ids)} verified chunks, got {len(verified)}"
        )

    @given(data=chunks_with_validity_mask_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_tamper_alert_count_equals_invalid_signature_count(self, data):
        """The number of tamper alerts equals the number of invalid-signature chunks.

        **Validates: Requirements 2.4, 7.1**
        """
        chunks, valid_chunk_ids, invalid_chunk_ids = data

        kms_client = _make_mock_kms_client(valid_chunk_ids, chunks)
        service = _make_service_with_kms(kms_client)

        verified, alerts = await service.verify_signatures(chunks)

        assert len(alerts) == len(invalid_chunk_ids), (
            f"Expected {len(invalid_chunk_ids)} tamper alerts, got {len(alerts)}"
        )

    @given(data=chunks_with_validity_mask_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_every_verified_chunk_id_in_valid_set(self, data):
        """Every verified chunk's chunk_id is in the set of valid chunk_ids.

        **Validates: Requirements 2.4**
        """
        chunks, valid_chunk_ids, invalid_chunk_ids = data

        kms_client = _make_mock_kms_client(valid_chunk_ids, chunks)
        service = _make_service_with_kms(kms_client)

        verified, alerts = await service.verify_signatures(chunks)

        for chunk in verified:
            assert chunk.chunk_id in valid_chunk_ids, (
                f"Verified chunk '{chunk.chunk_id}' not in valid set: {valid_chunk_ids}"
            )

    @given(data=chunks_with_validity_mask_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_every_tamper_alert_chunk_id_in_invalid_set(self, data):
        """Every tamper alert's chunk_id is in the set of invalid chunk_ids.

        **Validates: Requirements 7.1, 11.2**
        """
        chunks, valid_chunk_ids, invalid_chunk_ids = data

        kms_client = _make_mock_kms_client(valid_chunk_ids, chunks)
        service = _make_service_with_kms(kms_client)

        verified, alerts = await service.verify_signatures(chunks)

        for alert in alerts:
            assert alert.chunk_id in invalid_chunk_ids, (
                f"Tamper alert chunk '{alert.chunk_id}' not in invalid set: {invalid_chunk_ids}"
            )

    @given(data=chunks_with_validity_mask_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_no_overlap_between_verified_and_alert_chunk_ids(self, data):
        """No overlap between verified chunk_ids and alert chunk_ids.

        **Validates: Requirements 2.4, 7.1**
        """
        chunks, valid_chunk_ids, invalid_chunk_ids = data

        kms_client = _make_mock_kms_client(valid_chunk_ids, chunks)
        service = _make_service_with_kms(kms_client)

        verified, alerts = await service.verify_signatures(chunks)

        verified_ids = {chunk.chunk_id for chunk in verified}
        alert_ids = {alert.chunk_id for alert in alerts}

        overlap = verified_ids & alert_ids
        assert len(overlap) == 0, (
            f"Overlap between verified and alert chunk_ids: {overlap}"
        )

    @given(data=chunks_with_validity_mask_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_partition_invariant(self, data):
        """verified_count + alert_count = input_count (partition invariant).

        **Validates: Requirements 2.4, 7.1, 11.2**

        The filter partitions the input into exactly two disjoint sets:
        verified chunks and tamper alerts. No chunk is lost or duplicated.
        """
        chunks, valid_chunk_ids, invalid_chunk_ids = data

        kms_client = _make_mock_kms_client(valid_chunk_ids, chunks)
        service = _make_service_with_kms(kms_client)

        verified, alerts = await service.verify_signatures(chunks)

        total_output = len(verified) + len(alerts)
        assert total_output == len(chunks), (
            f"Partition invariant violated: {len(verified)} verified + "
            f"{len(alerts)} alerts = {total_output}, but input had {len(chunks)} chunks"
        )
