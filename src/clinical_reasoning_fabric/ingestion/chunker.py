"""Semantic chunking and Qdrant storage for the Axisweave Retrieval Stack.

Implements:
- SemanticChunker: Wraps Chonkie library for semantic text chunking with fallback
  to simple paragraph splitting if Chonkie is unavailable.
- QdrantStorageService: Manages Qdrant collection creation, chunk storage with
  full provenance metadata, and multi-tenant namespace isolation.

Requirements:
    1.5: Chunk text into semantic segments using Chonkie and store each chunk in
         Qdrant with provenance metadata (document_id, content_hash, kms_signature,
         chunk_index, ingestion_timestamp).
    13.2: Accept document ingestion with namespace and document_category for
          multi-tenant support.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    DocumentChunk,
    KMSSignature,
)
from clinical_reasoning_fabric.models.exceptions import IngestionError

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols for dependency injection
# =============================================================================


class QdrantClientProtocol(Protocol):
    """Protocol for Qdrant client operations enabling test mocking."""

    def get_collections(self) -> Any:
        """List existing collections."""
        ...

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        ...

    def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        """Create a new collection."""
        ...

    def upsert(self, collection_name: str, points: Any, **kwargs: Any) -> None:
        """Upsert points into a collection."""
        ...


# =============================================================================
# SemanticChunker
# =============================================================================


class SemanticChunker:
    """Semantic text chunker wrapping the Chonkie library.

    Uses Chonkie's semantic chunking to split clinical text at natural semantic
    boundaries. Falls back to simple paragraph splitting if Chonkie is not
    installed.

    Args:
        max_chunk_size: Maximum number of characters per chunk (default 1000).
        similarity_threshold: Semantic similarity threshold for splitting (default 0.5).
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.5,
    ):
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self._chonkie_available = self._check_chonkie_available()

    def _check_chonkie_available(self) -> bool:
        """Check if the Chonkie library is importable."""
        try:
            import chonkie  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "Chonkie library not available; falling back to paragraph-based chunking"
            )
            return False

    def chunk(self, text: str) -> list[str]:
        """Split text into semantic chunks.

        Uses Chonkie semantic chunking if available, otherwise falls back to
        paragraph-based splitting.

        Args:
            text: The text content to chunk.

        Returns:
            A list of text chunks. Always returns at least one chunk for
            non-empty input.

        Raises:
            ValueError: If text is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty or whitespace-only text")

        if self._chonkie_available:
            return self._chunk_with_chonkie(text)
        else:
            return self._chunk_by_paragraphs(text)

    def _chunk_with_chonkie(self, text: str) -> list[str]:
        """Chunk text using Chonkie's semantic chunking.

        Args:
            text: The text to chunk.

        Returns:
            List of semantically chunked text segments.
        """
        try:
            from chonkie import SemanticChunker as ChonkieSemanticChunker

            chunker = ChonkieSemanticChunker(
                max_chunk_size=self.max_chunk_size,
                similarity_threshold=self.similarity_threshold,
            )
            chunks = chunker.chunk(text)

            # Chonkie returns Chunk objects; extract text
            result = []
            for chunk_obj in chunks:
                if hasattr(chunk_obj, "text"):
                    chunk_text = chunk_obj.text.strip()
                elif isinstance(chunk_obj, str):
                    chunk_text = chunk_obj.strip()
                else:
                    chunk_text = str(chunk_obj).strip()

                if chunk_text:
                    result.append(chunk_text)

            # Ensure at least one chunk is returned
            if not result:
                result = [text.strip()]

            return result

        except Exception as e:
            logger.warning(
                "Chonkie chunking failed (%s), falling back to paragraph splitting: %s",
                type(e).__name__,
                e,
            )
            return self._chunk_by_paragraphs(text)

    def _chunk_by_paragraphs(self, text: str) -> list[str]:
        """Fallback: split text by paragraph boundaries.

        Splits on double newlines and respects max_chunk_size by further
        splitting long paragraphs.

        Args:
            text: The text to chunk.

        Returns:
            List of paragraph-based text chunks.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            # Single block of text without paragraph breaks
            paragraphs = [text.strip()]

        # Further split paragraphs that exceed max_chunk_size
        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.max_chunk_size:
                chunks.append(paragraph)
            else:
                # Split long paragraphs by sentences or at max_chunk_size
                sub_chunks = self._split_long_paragraph(paragraph)
                chunks.extend(sub_chunks)

        return chunks if chunks else [text.strip()]

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """Split a long paragraph into smaller chunks respecting sentence boundaries.

        Args:
            paragraph: A paragraph that exceeds max_chunk_size.

        Returns:
            List of sub-chunks from the paragraph.
        """
        import re

        # Split by sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_size:
                current_chunk = (
                    f"{current_chunk} {sentence}" if current_chunk else sentence
                )
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If a single sentence exceeds max_chunk_size, include it as-is
                current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [paragraph]


# =============================================================================
# QdrantStorageService
# =============================================================================


# Collection configuration constants
DENSE_VECTOR_SIZE = 1024
DENSE_VECTOR_DISTANCE = "Cosine"
DEFAULT_COLLECTION_NAME = "clinical_documents"


class QdrantStorageService:
    """Manages Qdrant vector storage for clinical document chunks.

    Creates and manages collections with dense vector (1024 dims, cosine) and
    BM25 sparse vector configuration. Stores chunks with full provenance metadata
    and multi-tenant isolation via namespace, document_category, and tenant_id.

    Requirements:
        1.5: Store chunks with provenance (document_id, content_hash, kms_signature,
             chunk_index, ingestion_timestamp).
        13.2: Support namespace and document_category for multi-tenant isolation.
    """

    def __init__(
        self,
        qdrant_client: QdrantClientProtocol,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: Any = None,
    ):
        """Initialize the QdrantStorageService.

        Args:
            qdrant_client: Qdrant client instance (or mock conforming to protocol).
            collection_name: Name of the Qdrant collection to use.
            embedding_model: Optional embedding model for generating dense vectors.
                If None, chunks are stored without embeddings (useful for testing).
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def ensure_collection(self) -> None:
        """Create or verify the Qdrant collection with proper vector configuration.

        Creates a collection with:
        - Dense vector: 1024 dimensions, cosine similarity
        - Sparse vector: BM25 configuration for hybrid search

        If the collection already exists, this method is a no-op.
        """
        try:
            if hasattr(self.client, "collection_exists"):
                if self.client.collection_exists(self.collection_name):
                    logger.info(
                        "Collection '%s' already exists", self.collection_name
                    )
                    return
            else:
                # Fallback: try to get collection info
                try:
                    self.client.get_collection(self.collection_name)
                    logger.info(
                        "Collection '%s' already exists", self.collection_name
                    )
                    return
                except Exception:
                    pass  # Collection doesn't exist, create it
        except Exception:
            pass  # Cannot check, attempt to create

        try:
            from qdrant_client.models import (
                Distance,
                SparseVectorParams,
                VectorParams,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=DENSE_VECTOR_SIZE,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                },
            )
            logger.info(
                "Created collection '%s' with dense (1024, cosine) and BM25 sparse config",
                self.collection_name,
            )
        except ImportError:
            # qdrant_client not available — create with dict-based config
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": {
                        "size": DENSE_VECTOR_SIZE,
                        "distance": DENSE_VECTOR_DISTANCE,
                    }
                },
                sparse_vectors_config={
                    "bm25": {}
                },
            )
            logger.info(
                "Created collection '%s' with dict-based config (qdrant models unavailable)",
                self.collection_name,
            )

    def store_chunks(
        self,
        chunks: list[str],
        document_id: str,
        content_hash: str,
        kms_signature: KMSSignature,
        namespace: str | None = None,
        document_category: str | None = None,
        tenant_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Store text chunks in Qdrant with full provenance metadata.

        Each chunk is stored with:
        - Provenance: document_id, content_hash, kms_signature, chunk_index,
          ingestion_timestamp
        - Multi-tenant fields: namespace, document_category, tenant_id

        Args:
            chunks: List of text strings to store.
            document_id: Source document identifier.
            content_hash: SHA-256 hash of the full document text.
            kms_signature: KMS signature of the content hash.
            namespace: Tenant namespace for isolation (Requirement 13.2).
            document_category: Document category (1-64 chars, Requirement 13.2).
            tenant_id: Tenant identifier for multi-tenant support.

        Returns:
            List of DocumentChunk objects representing the stored chunks.

        Raises:
            IngestionError: If storage fails for any chunk.
        """
        if not chunks:
            raise IngestionError(
                reason="No chunks to store",
                document_id=document_id,
                details={"content_hash": content_hash},
            )

        ingestion_timestamp = datetime.now(timezone.utc)
        stored_chunks: list[DocumentChunk] = []
        points_to_upsert = []

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{chunk_index}")
            )

            # Build provenance metadata
            provenance = ChunkProvenance(
                document_id=document_id,
                content_hash=content_hash,
                kms_signature=kms_signature,
                chunk_index=chunk_index,
                ingestion_timestamp=ingestion_timestamp,
            )

            # Generate embedding if model is available
            embedding = None
            if self.embedding_model is not None:
                try:
                    embedding = self.embedding_model.encode(chunk_text)
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()
                except Exception as e:
                    logger.warning(
                        "Embedding generation failed for chunk %d: %s",
                        chunk_index,
                        e,
                    )

            # Build the DocumentChunk model
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                embedding=embedding,
                provenance=provenance,
                namespace=namespace,
                document_category=document_category,
            )

            # Build payload for Qdrant storage
            payload = {
                "text": chunk_text,
                "document_id": document_id,
                "content_hash": content_hash,
                "kms_signature": kms_signature.signature,
                "kms_key_id": kms_signature.key_id,
                "kms_algorithm": kms_signature.algorithm,
                "kms_signed_at": kms_signature.signed_at.isoformat(),
                "chunk_index": chunk_index,
                "ingestion_timestamp": ingestion_timestamp.isoformat(),
                "namespace": namespace,
                "document_category": document_category,
                "tenant_id": tenant_id,
            }

            # Build the point for Qdrant
            point = {
                "id": chunk_id,
                "payload": payload,
            }

            # Add vector if available
            if embedding is not None:
                point["vector"] = {"dense": embedding}

            points_to_upsert.append(point)
            stored_chunks.append(doc_chunk)

        # Upsert all points to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_upsert,
            )
            logger.info(
                "Stored %d chunks for document_id=%s in collection '%s'",
                len(stored_chunks),
                document_id,
                self.collection_name,
            )
        except Exception as e:
            raise IngestionError(
                reason=f"Failed to store chunks in Qdrant: {e}",
                document_id=document_id,
                details={
                    "chunk_count": len(chunks),
                    "collection": self.collection_name,
                    "error_type": type(e).__name__,
                },
            )

        return stored_chunks
