"""Property-based tests for Chunking Content Preservation with Provenance.

**Validates: Requirements 1.5**

Property 3: Chunking Content Preservation with Provenance
- For any non-empty text, chunking produces >= 1 chunk.
- The union of all chunk texts contains all words from the original text
  (content preservation).
- When chunks are stored via QdrantStorageService, every DocumentChunk has:
  - A non-empty chunk_id
  - Non-empty text
  - Valid ChunkProvenance with: document_id set, content_hash is 64-char hex,
    kms_signature present, chunk_index >= 0, ingestion_timestamp is set
  - Sequential chunk_index values starting from 0
"""

import hashlib
import re
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clinical_reasoning_fabric.ingestion.chunker import (
    QdrantStorageService,
    SemanticChunker,
)
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    DocumentChunk,
    KMSSignature,
)


# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_kms_signature() -> KMSSignature:
    """Create a valid KMSSignature fixture for testing."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789012:key/test-key-id",
        signature="dGVzdC1zaWduYXR1cmUtYmFzZTY0",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_mock_qdrant_client() -> MagicMock:
    """Create a mock Qdrant client that accepts upserts without error."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.upsert.return_value = None
    return client


def _compute_content_hash(text: str) -> str:
    """Compute a valid SHA-256 content hash for test text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# Hypothesis strategies
# =============================================================================


def non_empty_text_strategy() -> st.SearchStrategy[str]:
    """Generate arbitrary non-empty text strings of various sizes."""
    return st.text(min_size=1, max_size=5000).filter(lambda t: t.strip())


def clinical_paragraph_strategy() -> st.SearchStrategy[str]:
    """Generate multi-paragraph clinical text to test paragraph chunking."""
    sentences = st.sampled_from([
        "Patient presents with chronic lower back pain radiating to left leg.",
        "Assessment indicates moderate to severe lumbar spinal stenosis.",
        "MRI findings show disc herniation at L4-L5 with nerve root compression.",
        "Treatment plan includes physical therapy and epidural steroid injection.",
        "Lab results show elevated CRP indicating ongoing inflammation.",
        "Patient reports difficulty walking more than 100 meters without rest.",
        "History of failed conservative management over 6 months.",
        "Current medications include gabapentin 300mg TID and ibuprofen PRN.",
        "Referral to neurosurgery for surgical decompression evaluation.",
        "BMI 32.4 with comorbid type 2 diabetes under good glycemic control.",
    ])
    return st.lists(sentences, min_size=1, max_size=10).map(
        lambda sents: "\n\n".join(sents)
    )


def large_text_strategy() -> st.SearchStrategy[str]:
    """Generate larger texts to test chunking behavior with sizeable input."""
    words = st.sampled_from([
        "clinical", "patient", "diagnosis", "treatment", "medication",
        "assessment", "evidence", "procedure", "condition", "therapy",
        "referral", "imaging", "lab", "results", "history",
        "chronic", "acute", "bilateral", "progressive", "moderate",
    ])
    return st.lists(words, min_size=20, max_size=500).map(lambda ws: " ".join(ws))


# =============================================================================
# Property tests
# =============================================================================


@pytest.mark.property
class TestChunkingContentPreservation:
    """Property 3: Chunking Content Preservation with Provenance.

    **Validates: Requirements 1.5**

    Tests that the semantic chunking and storage pipeline preserves content
    and produces valid provenance for every chunk.
    """

    chunker = SemanticChunker(max_chunk_size=1000, similarity_threshold=0.5)

    @given(text=non_empty_text_strategy())
    @settings(max_examples=100)
    def test_chunking_produces_at_least_one_chunk(self, text: str):
        """For any non-empty text, chunking always produces >= 1 chunk.

        **Validates: Requirements 1.5**

        The SemanticChunker must never return an empty list for non-empty input.
        """
        chunks = self.chunker.chunk(text)

        assert len(chunks) >= 1, (
            f"Chunking produced 0 chunks for non-empty text of length {len(text)}"
        )
        # Every chunk must be a non-empty string
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, str), f"Chunk {i} is not a string: {type(chunk)}"
            assert len(chunk) > 0, f"Chunk {i} is empty"

    @given(text=non_empty_text_strategy())
    @settings(max_examples=100)
    def test_chunking_preserves_all_words(self, text: str):
        """The concatenation of all chunk texts contains all words from the original.

        **Validates: Requirements 1.5**

        Content preservation: every word in the original text must appear
        in the union of all chunk texts. Words are defined as sequences of
        non-whitespace characters.
        """
        chunks = self.chunker.chunk(text)
        concatenated = " ".join(chunks)

        # Extract words from original text
        original_words = set(text.split())
        # Extract words from concatenated chunks
        chunk_words = set(concatenated.split())

        # Every word in the original must appear in the chunks
        missing_words = original_words - chunk_words
        assert not missing_words, (
            f"Content preservation violated: {len(missing_words)} words missing "
            f"from chunks. Sample missing: {list(missing_words)[:5]}"
        )

    @given(text=clinical_paragraph_strategy())
    @settings(max_examples=50)
    def test_clinical_text_chunking_preserves_content(self, text: str):
        """Clinical paragraph text is preserved through chunking.

        **Validates: Requirements 1.5**

        Multi-paragraph clinical notes must have all content preserved after
        semantic chunking.
        """
        chunks = self.chunker.chunk(text)
        concatenated = " ".join(chunks)

        original_words = set(text.split())
        chunk_words = set(concatenated.split())

        missing_words = original_words - chunk_words
        assert not missing_words, (
            f"Clinical content preservation failed: {len(missing_words)} words missing"
        )

    @given(text=large_text_strategy())
    @settings(max_examples=50)
    def test_large_text_chunking_produces_multiple_chunks(self, text: str):
        """Large texts are split into multiple chunks while preserving content.

        **Validates: Requirements 1.5**

        Texts exceeding max_chunk_size should be split into multiple chunks.
        """
        chunks = self.chunker.chunk(text)

        assert len(chunks) >= 1, "Must produce at least one chunk"

        # Verify content preservation for large texts
        concatenated = " ".join(chunks)
        original_words = set(text.split())
        chunk_words = set(concatenated.split())

        missing_words = original_words - chunk_words
        assert not missing_words, (
            f"Large text content preservation failed: {len(missing_words)} words lost"
        )


@pytest.mark.property
class TestChunkStorageProvenance:
    """Property 3 (Storage): Provenance metadata validity for stored chunks.

    **Validates: Requirements 1.5**

    Tests that when chunks are stored via QdrantStorageService, every
    DocumentChunk has valid provenance fields.
    """

    @given(text=non_empty_text_strategy())
    @settings(max_examples=100)
    def test_stored_chunks_have_valid_provenance(self, text: str):
        """Every stored DocumentChunk has all required provenance fields with valid values.

        **Validates: Requirements 1.5**

        Provenance fields checked:
        - document_id is set (non-empty)
        - content_hash is a 64-character hex string
        - kms_signature is present with non-empty signature
        - chunk_index >= 0
        - ingestion_timestamp is set
        """
        chunker = SemanticChunker(max_chunk_size=1000)
        chunks = chunker.chunk(text)

        mock_client = _make_mock_qdrant_client()
        storage = QdrantStorageService(qdrant_client=mock_client)

        document_id = "test-doc-001"
        content_hash = _compute_content_hash(text)
        kms_signature = _make_kms_signature()

        stored_chunks = storage.store_chunks(
            chunks=chunks,
            document_id=document_id,
            content_hash=content_hash,
            kms_signature=kms_signature,
            namespace="test-namespace",
            document_category="clinical-notes",
        )

        assert len(stored_chunks) == len(chunks), (
            f"Expected {len(chunks)} stored chunks, got {len(stored_chunks)}"
        )

        for i, doc_chunk in enumerate(stored_chunks):
            # Non-empty chunk_id
            assert doc_chunk.chunk_id, (
                f"Chunk {i} has empty chunk_id"
            )
            assert len(doc_chunk.chunk_id) > 0, (
                f"Chunk {i} chunk_id is empty string"
            )

            # Non-empty text
            assert doc_chunk.text, (
                f"Chunk {i} has empty text"
            )
            assert len(doc_chunk.text) > 0, (
                f"Chunk {i} text is empty string"
            )

            # Valid provenance
            prov = doc_chunk.provenance
            assert isinstance(prov, ChunkProvenance), (
                f"Chunk {i} provenance is not ChunkProvenance: {type(prov)}"
            )

            # document_id set
            assert prov.document_id == document_id, (
                f"Chunk {i} document_id mismatch: '{prov.document_id}' != '{document_id}'"
            )

            # content_hash is 64-char hex
            assert len(prov.content_hash) == 64, (
                f"Chunk {i} content_hash length is {len(prov.content_hash)}, expected 64"
            )
            assert re.fullmatch(r"[0-9a-f]{64}", prov.content_hash), (
                f"Chunk {i} content_hash '{prov.content_hash}' is not valid 64-char hex"
            )

            # kms_signature present
            assert prov.kms_signature is not None, (
                f"Chunk {i} kms_signature is None"
            )
            assert prov.kms_signature.signature, (
                f"Chunk {i} kms_signature.signature is empty"
            )
            assert prov.kms_signature.key_id, (
                f"Chunk {i} kms_signature.key_id is empty"
            )

            # chunk_index >= 0
            assert prov.chunk_index >= 0, (
                f"Chunk {i} chunk_index is {prov.chunk_index}, expected >= 0"
            )

            # ingestion_timestamp is set
            assert prov.ingestion_timestamp is not None, (
                f"Chunk {i} ingestion_timestamp is None"
            )
            assert isinstance(prov.ingestion_timestamp, datetime), (
                f"Chunk {i} ingestion_timestamp is not datetime: {type(prov.ingestion_timestamp)}"
            )

    @given(text=non_empty_text_strategy())
    @settings(max_examples=100)
    def test_stored_chunks_have_sequential_indices(self, text: str):
        """Stored chunks have sequential chunk_index values starting from 0.

        **Validates: Requirements 1.5**

        chunk_index must be 0, 1, 2, ..., n-1 for n chunks.
        """
        chunker = SemanticChunker(max_chunk_size=1000)
        chunks = chunker.chunk(text)

        mock_client = _make_mock_qdrant_client()
        storage = QdrantStorageService(qdrant_client=mock_client)

        document_id = "test-doc-sequential"
        content_hash = _compute_content_hash(text)
        kms_signature = _make_kms_signature()

        stored_chunks = storage.store_chunks(
            chunks=chunks,
            document_id=document_id,
            content_hash=content_hash,
            kms_signature=kms_signature,
        )

        # Verify sequential indices starting from 0
        expected_indices = list(range(len(stored_chunks)))
        actual_indices = [chunk.provenance.chunk_index for chunk in stored_chunks]

        assert actual_indices == expected_indices, (
            f"Chunk indices are not sequential from 0: {actual_indices} "
            f"(expected {expected_indices})"
        )

    @given(text=non_empty_text_strategy())
    @settings(max_examples=100)
    def test_stored_chunks_text_matches_input(self, text: str):
        """Stored DocumentChunk text matches the chunker output.

        **Validates: Requirements 1.5**

        The text field in each stored DocumentChunk must equal the
        corresponding chunk text from the chunker.
        """
        chunker = SemanticChunker(max_chunk_size=1000)
        chunks = chunker.chunk(text)

        mock_client = _make_mock_qdrant_client()
        storage = QdrantStorageService(qdrant_client=mock_client)

        document_id = "test-doc-text-match"
        content_hash = _compute_content_hash(text)
        kms_signature = _make_kms_signature()

        stored_chunks = storage.store_chunks(
            chunks=chunks,
            document_id=document_id,
            content_hash=content_hash,
            kms_signature=kms_signature,
        )

        for i, (original_chunk, stored_chunk) in enumerate(zip(chunks, stored_chunks)):
            assert stored_chunk.text == original_chunk, (
                f"Chunk {i} text mismatch: stored '{stored_chunk.text[:50]}...' "
                f"!= original '{original_chunk[:50]}...'"
            )

    @given(text=clinical_paragraph_strategy())
    @settings(max_examples=50)
    def test_stored_chunks_unique_ids(self, text: str):
        """All stored chunks have unique chunk_id values.

        **Validates: Requirements 1.5**

        Each chunk must have a distinct identifier within a document.
        """
        chunker = SemanticChunker(max_chunk_size=1000)
        chunks = chunker.chunk(text)

        mock_client = _make_mock_qdrant_client()
        storage = QdrantStorageService(qdrant_client=mock_client)

        document_id = "test-doc-unique-ids"
        content_hash = _compute_content_hash(text)
        kms_signature = _make_kms_signature()

        stored_chunks = storage.store_chunks(
            chunks=chunks,
            document_id=document_id,
            content_hash=content_hash,
            kms_signature=kms_signature,
        )

        chunk_ids = [chunk.chunk_id for chunk in stored_chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), (
            f"Duplicate chunk_ids found: {len(chunk_ids)} total, "
            f"{len(set(chunk_ids))} unique"
        )
