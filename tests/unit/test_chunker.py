"""Unit tests for SemanticChunker and QdrantStorageService.

Tests the semantic chunking and Qdrant storage components:
- SemanticChunker: paragraph fallback, long paragraph splitting, error handling
- QdrantStorageService: collection creation, chunk storage with provenance,
  multi-tenant payload fields

Validates:
    - Requirements 1.5: Semantic chunking and provenance metadata storage
    - Requirements 13.2: Multi-tenant namespace/category/tenant_id support
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from clinical_reasoning_fabric.ingestion.chunker import (
    DENSE_VECTOR_DISTANCE,
    DENSE_VECTOR_SIZE,
    DEFAULT_COLLECTION_NAME,
    QdrantStorageService,
    SemanticChunker,
)
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    DocumentChunk,
    KMSSignature,
)
from clinical_reasoning_fabric.models.exceptions import IngestionError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def chunker():
    """Create a SemanticChunker with Chonkie disabled for deterministic testing."""
    c = SemanticChunker(max_chunk_size=200)
    c._chonkie_available = False  # Force paragraph fallback for deterministic tests
    return c


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client conforming to QdrantClientProtocol."""
    client = MagicMock()
    client.collection_exists.return_value = False
    client.create_collection = MagicMock()
    client.upsert = MagicMock()
    return client


@pytest.fixture
def kms_signature():
    """Sample KMS signature for testing."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789:key/test-key",
        signature="dGVzdC1zaWduYXR1cmUtYnl0ZXM=",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def storage_service(mock_qdrant_client):
    """Create a QdrantStorageService with mocked client."""
    return QdrantStorageService(
        qdrant_client=mock_qdrant_client,
        collection_name="test_collection",
    )


# =============================================================================
# Tests for SemanticChunker
# =============================================================================


class TestSemanticChunker:
    """Tests for SemanticChunker.chunk() method."""

    def test_chunk_single_paragraph(self, chunker):
        """Single paragraph text returns one chunk."""
        text = "Patient presents with chronic lower back pain lasting three months."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_multiple_paragraphs(self, chunker):
        """Multiple paragraphs are split into separate chunks."""
        text = "First paragraph about diagnosis.\n\nSecond paragraph about treatment."
        chunks = chunker.chunk(text)
        assert len(chunks) == 2
        assert "First paragraph" in chunks[0]
        assert "Second paragraph" in chunks[1]

    def test_chunk_preserves_content(self, chunker):
        """All original content is preserved across chunks."""
        text = "Part one of the note.\n\nPart two of the note.\n\nPart three."
        chunks = chunker.chunk(text)
        combined = " ".join(chunks)
        assert "Part one" in combined
        assert "Part two" in combined
        assert "Part three" in combined

    def test_chunk_splits_long_paragraph(self):
        """Paragraphs exceeding max_chunk_size are further split."""
        chunker = SemanticChunker(max_chunk_size=50)
        chunker._chonkie_available = False
        text = "This is a sentence. Another sentence here. A third sentence follows. And a fourth one too."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            # Each chunk should be reasonable size (may slightly exceed due to sentence boundaries)
            assert len(chunk) > 0

    def test_chunk_empty_text_raises_error(self, chunker):
        """Empty text raises ValueError."""
        with pytest.raises(ValueError, match="Cannot chunk empty"):
            chunker.chunk("")

    def test_chunk_whitespace_only_raises_error(self, chunker):
        """Whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Cannot chunk empty"):
            chunker.chunk("   \n\n  \t  ")

    def test_chunk_always_returns_at_least_one(self, chunker):
        """Non-empty text always produces at least one chunk."""
        text = "A simple note."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_chunk_strips_whitespace(self, chunker):
        """Chunks have leading/trailing whitespace stripped."""
        text = "  First paragraph.  \n\n  Second paragraph.  "
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert chunk == chunk.strip()

    def test_chunk_handles_text_without_paragraph_breaks(self, chunker):
        """Text without double-newlines returns as a single chunk."""
        text = "Single line text without paragraph breaks anywhere in it."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chonkie_fallback_on_import_error(self):
        """When Chonkie is not available, falls back to paragraph splitting."""
        chunker = SemanticChunker()
        chunker._chonkie_available = False
        text = "Paragraph one.\n\nParagraph two."
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_chunk_default_configuration(self):
        """Default chunker has expected configuration values."""
        chunker = SemanticChunker()
        assert chunker.max_chunk_size == 1000
        assert chunker.similarity_threshold == 0.5


# =============================================================================
# Tests for QdrantStorageService.ensure_collection()
# =============================================================================


class TestQdrantStorageServiceCollection:
    """Tests for QdrantStorageService collection management."""

    def test_ensure_collection_creates_when_not_exists(
        self, storage_service, mock_qdrant_client
    ):
        """Creates collection with correct config when it doesn't exist."""
        mock_qdrant_client.collection_exists.return_value = False

        storage_service.ensure_collection()

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args
        assert call_kwargs[1]["collection_name"] == "test_collection"

        # Verify vector configs were passed
        vectors_config = call_kwargs[1]["vectors_config"]
        sparse_config = call_kwargs[1]["sparse_vectors_config"]
        assert "dense" in vectors_config
        assert "bm25" in sparse_config

    def test_ensure_collection_skips_when_exists(
        self, storage_service, mock_qdrant_client
    ):
        """Does not create collection when it already exists."""
        mock_qdrant_client.collection_exists.return_value = True

        storage_service.ensure_collection()

        mock_qdrant_client.create_collection.assert_not_called()

    def test_ensure_collection_dense_vector_size(
        self, storage_service, mock_qdrant_client
    ):
        """Dense vector config uses 1024 dimensions."""
        assert DENSE_VECTOR_SIZE == 1024

    def test_ensure_collection_cosine_distance(
        self, storage_service, mock_qdrant_client
    ):
        """Dense vector config uses cosine similarity."""
        assert DENSE_VECTOR_DISTANCE == "Cosine"


# =============================================================================
# Tests for QdrantStorageService.store_chunks()
# =============================================================================


class TestQdrantStorageServiceStoreChunks:
    """Tests for QdrantStorageService.store_chunks() method."""

    def test_store_chunks_returns_document_chunks(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """store_chunks() returns a list of DocumentChunk objects."""
        chunks = ["Clinical note chunk one.", "Clinical note chunk two."]
        result = storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-001",
            content_hash="a" * 64,
            kms_signature=kms_signature,
        )

        assert len(result) == 2
        assert all(isinstance(c, DocumentChunk) for c in result)

    def test_store_chunks_provenance_metadata(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Each stored chunk has correct provenance metadata."""
        chunks = ["First chunk.", "Second chunk."]
        content_hash = "b" * 64

        result = storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-002",
            content_hash=content_hash,
            kms_signature=kms_signature,
        )

        for i, doc_chunk in enumerate(result):
            assert doc_chunk.provenance.document_id == "doc-002"
            assert doc_chunk.provenance.content_hash == content_hash
            assert doc_chunk.provenance.kms_signature == kms_signature
            assert doc_chunk.provenance.chunk_index == i
            assert doc_chunk.provenance.ingestion_timestamp is not None

    def test_store_chunks_multi_tenant_fields(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Namespace, document_category, and tenant_id are stored in payload."""
        chunks = ["Tenant-isolated chunk."]

        result = storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-003",
            content_hash="c" * 64,
            kms_signature=kms_signature,
            namespace="hedis-gap-closure",
            document_category="clinical-note",
            tenant_id="tenant-alpha",
        )

        assert result[0].namespace == "hedis-gap-closure"
        assert result[0].document_category == "clinical-note"

        # Verify payload sent to Qdrant includes tenant fields
        upsert_call = mock_qdrant_client.upsert.call_args
        points = upsert_call[1]["points"]
        payload = points[0]["payload"]
        assert payload["namespace"] == "hedis-gap-closure"
        assert payload["document_category"] == "clinical-note"
        assert payload["tenant_id"] == "tenant-alpha"

    def test_store_chunks_payload_contains_provenance(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Qdrant payload contains all provenance fields."""
        chunks = ["Evidence text."]
        content_hash = "d" * 64

        storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-004",
            content_hash=content_hash,
            kms_signature=kms_signature,
        )

        upsert_call = mock_qdrant_client.upsert.call_args
        points = upsert_call[1]["points"]
        payload = points[0]["payload"]

        assert payload["document_id"] == "doc-004"
        assert payload["content_hash"] == content_hash
        assert payload["kms_signature"] == kms_signature.signature
        assert payload["kms_key_id"] == kms_signature.key_id
        assert payload["kms_algorithm"] == kms_signature.algorithm
        assert payload["chunk_index"] == 0
        assert "ingestion_timestamp" in payload
        assert payload["text"] == "Evidence text."

    def test_store_chunks_sequential_indices(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Chunk indices are sequential starting from 0."""
        chunks = ["Chunk A.", "Chunk B.", "Chunk C."]

        result = storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-005",
            content_hash="e" * 64,
            kms_signature=kms_signature,
        )

        indices = [c.provenance.chunk_index for c in result]
        assert indices == [0, 1, 2]

    def test_store_chunks_unique_ids(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Each chunk gets a unique chunk_id."""
        chunks = ["Chunk 1.", "Chunk 2.", "Chunk 3."]

        result = storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-006",
            content_hash="f" * 64,
            kms_signature=kms_signature,
        )

        chunk_ids = [c.chunk_id for c in result]
        assert len(set(chunk_ids)) == 3  # All unique

    def test_store_chunks_empty_list_raises_error(
        self, storage_service, kms_signature
    ):
        """Empty chunk list raises IngestionError."""
        with pytest.raises(IngestionError, match="No chunks to store"):
            storage_service.store_chunks(
                chunks=[],
                document_id="doc-007",
                content_hash="0" * 64,
                kms_signature=kms_signature,
            )

    def test_store_chunks_qdrant_failure_raises_ingestion_error(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Qdrant upsert failure raises IngestionError."""
        mock_qdrant_client.upsert.side_effect = RuntimeError("Connection refused")

        with pytest.raises(IngestionError) as exc_info:
            storage_service.store_chunks(
                chunks=["Test chunk."],
                document_id="doc-008",
                content_hash="1" * 64,
                kms_signature=kms_signature,
            )

        assert "Failed to store chunks" in exc_info.value.reason

    def test_store_chunks_with_embedding_model(
        self, kms_signature, mock_qdrant_client
    ):
        """When embedding model is provided, vectors are included in points."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1] * 1024

        service = QdrantStorageService(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            embedding_model=mock_embedder,
        )

        result = service.store_chunks(
            chunks=["Embeddable text."],
            document_id="doc-009",
            content_hash="2" * 64,
            kms_signature=kms_signature,
        )

        # Embedding should be set on the DocumentChunk
        assert result[0].embedding == [0.1] * 1024

        # Verify vector is in the Qdrant point
        upsert_call = mock_qdrant_client.upsert.call_args
        points = upsert_call[1]["points"]
        assert "vector" in points[0]
        assert points[0]["vector"]["dense"] == [0.1] * 1024

    def test_store_chunks_without_embedding_model(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Without embedding model, chunks are stored without vectors."""
        result = storage_service.store_chunks(
            chunks=["No embedding chunk."],
            document_id="doc-010",
            content_hash="3" * 64,
            kms_signature=kms_signature,
        )

        assert result[0].embedding is None

        # Verify no vector in Qdrant point
        upsert_call = mock_qdrant_client.upsert.call_args
        points = upsert_call[1]["points"]
        assert "vector" not in points[0]

    def test_store_chunks_ingestion_timestamp_is_utc(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Ingestion timestamp is in UTC."""
        result = storage_service.store_chunks(
            chunks=["UTC timestamp test."],
            document_id="doc-011",
            content_hash="4" * 64,
            kms_signature=kms_signature,
        )

        ts = result[0].provenance.ingestion_timestamp
        assert ts.tzinfo is not None
        assert ts.tzinfo == timezone.utc

    def test_store_chunks_consistent_timestamp_across_batch(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """All chunks in a batch share the same ingestion timestamp."""
        chunks = ["Chunk A.", "Chunk B.", "Chunk C."]

        result = storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-012",
            content_hash="5" * 64,
            kms_signature=kms_signature,
        )

        timestamps = [c.provenance.ingestion_timestamp for c in result]
        assert all(t == timestamps[0] for t in timestamps)

    def test_store_chunks_calls_upsert_once_with_all_points(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """All chunks are upserted in a single call for efficiency."""
        chunks = ["One.", "Two.", "Three."]

        storage_service.store_chunks(
            chunks=chunks,
            document_id="doc-013",
            content_hash="6" * 64,
            kms_signature=kms_signature,
        )

        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert len(call_kwargs["points"]) == 3

    def test_store_chunks_null_optional_fields(
        self, storage_service, kms_signature, mock_qdrant_client
    ):
        """Optional fields (namespace, category, tenant_id) default to None in payload."""
        storage_service.store_chunks(
            chunks=["Minimal chunk."],
            document_id="doc-014",
            content_hash="7" * 64,
            kms_signature=kms_signature,
        )

        upsert_call = mock_qdrant_client.upsert.call_args
        payload = upsert_call[1]["points"][0]["payload"]
        assert payload["namespace"] is None
        assert payload["document_category"] is None
        assert payload["tenant_id"] is None
