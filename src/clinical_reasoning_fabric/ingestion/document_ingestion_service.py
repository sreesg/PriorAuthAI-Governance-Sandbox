"""Document Ingestion Service for the Axisweave Retrieval Stack.

Orchestrates the full ingestion pipeline: parse PDF → scrub PII → hash → sign → chunk → store.
Uses Docling for PDF parsing, SHA-256 for content hashing, and AWS KMS for cryptographic signing.

Requirements:
    1.1: Parse clinical PDF documents using Docling PDF parsing
    1.3: Compute SHA-256 content hash of cleaned text
    1.4: Sign hash with AWS KMS asymmetric key
    1.6: Reject corrupted/unsupported formats with IngestionError
    1.7: Halt ingestion on KMS failure, discard unsigned chunks, log to audit trail
"""

import base64
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from clinical_reasoning_fabric.ingestion.pii_scrubber import PIIScrubber
from clinical_reasoning_fabric.models.core import (
    IngestionResult,
    KMSSignature,
)
from clinical_reasoning_fabric.models.exceptions import (
    IngestionError,
    KMSUnavailableError,
    PIIScrubError,
)

logger = logging.getLogger(__name__)


class PDFParser(Protocol):
    """Protocol for PDF parsing implementations."""

    def parse(self, document_bytes: bytes) -> str:
        """Parse PDF bytes into structured text."""
        ...


class ChunkerProtocol(Protocol):
    """Protocol for text chunking implementations."""

    def chunk(self, text: str) -> list[str]:
        """Split text into semantic chunks."""
        ...


class DocumentIngestionService:
    """Orchestrates PDF parsing, PII scrubbing, signing, chunking, and storage.

    Full ingestion pipeline: parse → scrub → hash → sign → chunk → store.
    On KMS failure, halts ingestion, discards unsigned chunks, and logs to audit trail.
    """

    def __init__(
        self,
        kms_client: Any,
        qdrant_client: Any,
        pii_scrubber: PIIScrubber,
        chunker: Any,
        kms_key_id: str = "alias/clinical-document-signing",
        pdf_parser: PDFParser | None = None,
        audit_logger: Any | None = None,
    ):
        """Initialize the DocumentIngestionService.

        Args:
            kms_client: boto3 KMS client for cryptographic signing.
            qdrant_client: Qdrant vector database client for storage.
            pii_scrubber: PIIScrubber instance for HIPAA Safe Harbor compliance.
            chunker: Semantic chunker (Chonkie) for splitting text.
            kms_key_id: AWS KMS key identifier for signing operations.
            pdf_parser: Optional PDF parser implementation. Defaults to Docling.
            audit_logger: Optional audit trail logger.
        """
        self.kms = kms_client
        self.qdrant = qdrant_client
        self.pii_scrubber = pii_scrubber
        self.chunker = chunker
        self.kms_key_id = kms_key_id
        self.pdf_parser = pdf_parser
        self.audit_logger = audit_logger

    async def ingest_document(
        self,
        document_bytes: bytes,
        document_id: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Execute the full ingestion pipeline: parse → scrub → hash → sign → chunk → store.

        Args:
            document_bytes: Raw PDF document bytes.
            document_id: Unique identifier for the document.
            source_metadata: Optional metadata about the document source.

        Returns:
            IngestionResult with document_id, content_hash, signature,
            chunk_count, and ingestion_timestamp.

        Raises:
            IngestionError: On corrupted/unsupported document format.
            PIIScrubError: If PII scrubbing fails.
            KMSUnavailableError: If KMS signing fails (halts ingestion,
                discards unsigned chunks).
        """
        if source_metadata is None:
            source_metadata = {}

        # Step 1: Parse PDF into structured text
        logger.info("Starting ingestion for document_id=%s", document_id)
        text = await self.parse_pdf(document_bytes)

        # Step 2: Scrub PII (HIPAA Safe Harbor compliance)
        try:
            scrubbed_text = self.pii_scrubber.scrub(text)
        except PIIScrubError:
            logger.error(
                "PII scrubbing failed for document_id=%s; halting ingestion",
                document_id,
            )
            self._log_audit(
                event="pii_scrub_failure",
                document_id=document_id,
                details={"reason": "PII scrubbing failed"},
            )
            raise

        # Step 3: Compute SHA-256 content hash
        content_hash = await self.compute_hash(scrubbed_text)

        # Step 4: Sign hash with KMS
        try:
            signature = await self.sign_hash(content_hash)
        except KMSUnavailableError:
            # Requirement 1.7: Halt ingestion, discard unsigned chunks, log to audit trail
            logger.error(
                "KMS signing failed for document_id=%s; halting ingestion, "
                "discarding unsigned chunks",
                document_id,
            )
            self._log_audit(
                event="kms_signing_failure",
                document_id=document_id,
                details={
                    "content_hash": content_hash,
                    "reason": "KMS unavailable or signing failed",
                },
            )
            raise

        # Step 5: Chunk and store
        chunk_count = await self.chunk_and_store(
            text=scrubbed_text,
            document_id=document_id,
            content_hash=content_hash,
            signature=signature,
            source_metadata=source_metadata,
        )

        ingestion_timestamp = datetime.now(timezone.utc)

        result = IngestionResult(
            document_id=document_id,
            content_hash=content_hash,
            signature=signature,
            chunk_count=chunk_count,
            ingestion_timestamp=ingestion_timestamp,
        )

        logger.info(
            "Ingestion complete for document_id=%s: %d chunks stored",
            document_id,
            chunk_count,
        )
        self._log_audit(
            event="ingestion_complete",
            document_id=document_id,
            details={
                "content_hash": content_hash,
                "chunk_count": chunk_count,
                "ingestion_timestamp": ingestion_timestamp.isoformat(),
            },
        )

        return result

    async def parse_pdf(self, document_bytes: bytes) -> str:
        """Parse PDF document bytes into structured text using Docling.

        Args:
            document_bytes: Raw PDF binary content.

        Returns:
            Extracted text content from the PDF.

        Raises:
            IngestionError: If the document is corrupted, empty, or in an
                unsupported format.
        """
        if not document_bytes:
            raise IngestionError(
                reason="Document is empty (zero bytes)",
                details={"byte_length": 0},
            )

        # Check for PDF magic bytes (%PDF-)
        if not document_bytes[:5] == b"%PDF-":
            raise IngestionError(
                reason="Unsupported document format: not a valid PDF file",
                details={
                    "magic_bytes": document_bytes[:10].hex(),
                    "expected": "25504446 (%PDF-)",
                },
            )

        # Use injected parser if available, otherwise attempt Docling
        if self.pdf_parser is not None:
            try:
                text = self.pdf_parser.parse(document_bytes)
            except Exception as e:
                raise IngestionError(
                    reason=f"PDF parsing failed: {e}",
                    details={"error_type": type(e).__name__, "error_msg": str(e)},
                )
        else:
            text = self._parse_with_docling(document_bytes)

        if not text or not text.strip():
            raise IngestionError(
                reason="PDF parsing produced empty text; document may be corrupted or image-only",
                details={"byte_length": len(document_bytes)},
            )

        return text

    def _parse_with_docling(self, document_bytes: bytes) -> str:
        """Parse PDF using the Docling library.

        Falls back to basic pypdf extraction if Docling is not available.

        Args:
            document_bytes: Raw PDF binary content.

        Returns:
            Extracted text content.

        Raises:
            IngestionError: If parsing fails.
        """
        try:
            from docling.document_converter import DocumentConverter
            import tempfile
            import os

            # Docling requires a file path, write to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(document_bytes)
                tmp_path = tmp.name

            try:
                converter = DocumentConverter()
                result = converter.convert(tmp_path)
                text = result.document.export_to_markdown()
                return text
            finally:
                os.unlink(tmp_path)

        except ImportError:
            # Docling not available, fall back to pypdf
            logger.warning(
                "Docling not available, falling back to pypdf for PDF extraction"
            )
            return self._parse_with_pypdf(document_bytes)
        except Exception as e:
            raise IngestionError(
                reason=f"Docling PDF parsing failed: {e}",
                details={"error_type": type(e).__name__, "error_msg": str(e)},
            )

    def _parse_with_pypdf(self, document_bytes: bytes) -> str:
        """Fallback PDF parsing using pypdf.

        Args:
            document_bytes: Raw PDF binary content.

        Returns:
            Extracted text content.

        Raises:
            IngestionError: If parsing fails.
        """
        try:
            import io
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(document_bytes))
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            return "\n".join(pages_text)
        except ImportError:
            raise IngestionError(
                reason="No PDF parsing library available (neither docling nor pypdf installed)",
                details={},
            )
        except Exception as e:
            raise IngestionError(
                reason=f"PDF parsing failed: {e}",
                details={"error_type": type(e).__name__, "error_msg": str(e)},
            )

    async def compute_hash(self, text: str) -> str:
        """Compute SHA-256 content hash of the given text.

        Args:
            text: The text content to hash (typically scrubbed text).

        Returns:
            Lowercase hexadecimal SHA-256 digest (64 characters).
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def sign_hash(self, content_hash: str) -> KMSSignature:
        """Sign content hash with AWS KMS asymmetric key.

        Args:
            content_hash: The SHA-256 hex digest to sign.

        Returns:
            KMSSignature with key_id, base64-encoded signature, algorithm, and timestamp.

        Raises:
            KMSUnavailableError: If KMS is unavailable or signing fails.
        """
        try:
            message_bytes = content_hash.encode("utf-8")

            response = self.kms.sign(
                KeyId=self.kms_key_id,
                Message=message_bytes,
                MessageType="RAW",
                SigningAlgorithm="RSASSA_PKCS1_V1_5_SHA_256",
            )

            signature_bytes = response["Signature"]
            signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")

            return KMSSignature(
                key_id=response.get("KeyId", self.kms_key_id),
                signature=signature_b64,
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            # Check if it's a boto3/botocore client error
            error_type = type(e).__name__
            if error_type in (
                "KMSInternalException",
                "KMSInvalidStateException",
                "DisabledException",
                "NotFoundException",
                "DependencyTimeoutException",
                "InvalidKeyUsageException",
                "KeyUnavailableException",
                "ClientError",
                "EndpointConnectionError",
                "ConnectTimeoutError",
                "ReadTimeoutError",
                "ConnectionError",
                "BotoCoreError",
            ) or "KMS" in error_type or "Connection" in error_type or "Timeout" in error_type:
                raise KMSUnavailableError(
                    reason=f"KMS signing failed: {e}",
                    details={
                        "error_type": error_type,
                        "key_id": self.kms_key_id,
                        "content_hash": content_hash,
                    },
                )
            # For unexpected errors, still wrap as KMSUnavailableError
            raise KMSUnavailableError(
                reason=f"Unexpected error during KMS signing: {e}",
                details={
                    "error_type": error_type,
                    "key_id": self.kms_key_id,
                },
            )

    async def chunk_and_store(
        self,
        text: str,
        document_id: str,
        content_hash: str,
        signature: KMSSignature,
        source_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk text into semantic segments and store in Qdrant with provenance.

        Args:
            text: The scrubbed text to chunk.
            document_id: The document identifier.
            content_hash: SHA-256 hash of the scrubbed text.
            signature: KMS signature of the content hash.
            source_metadata: Optional metadata about the document source.

        Returns:
            Number of chunks stored.
        """
        if source_metadata is None:
            source_metadata = {}

        # Use the chunker to split text into semantic segments
        if hasattr(self.chunker, "chunk"):
            chunks = self.chunker.chunk(text)
        elif callable(self.chunker):
            chunks = self.chunker(text)
        else:
            # Simple fallback: split by paragraphs
            chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not chunks:
            chunks = [text]

        ingestion_timestamp = datetime.now(timezone.utc)

        # Store each chunk with provenance metadata
        for chunk_index, chunk_text in enumerate(chunks):
            provenance = {
                "document_id": document_id,
                "content_hash": content_hash,
                "kms_signature": signature.signature,
                "kms_key_id": signature.key_id,
                "kms_algorithm": signature.algorithm,
                "chunk_index": chunk_index,
                "ingestion_timestamp": ingestion_timestamp.isoformat(),
            }

            # Merge source metadata
            payload = {**provenance, **source_metadata}

            # Store in Qdrant if client is available
            if self.qdrant is not None and hasattr(self.qdrant, "upsert"):
                try:
                    import uuid

                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{chunk_index}"))
                    self.qdrant.upsert(
                        collection_name=source_metadata.get("collection", "clinical_documents"),
                        points=[
                            {
                                "id": point_id,
                                "payload": {**payload, "text": chunk_text},
                            }
                        ],
                    )
                except Exception as e:
                    logger.error(
                        "Failed to store chunk %d for document_id=%s: %s",
                        chunk_index,
                        document_id,
                        e,
                    )
                    raise IngestionError(
                        reason=f"Failed to store chunk {chunk_index} in Qdrant: {e}",
                        document_id=document_id,
                        details={"chunk_index": chunk_index, "error": str(e)},
                    )

        return len(chunks)

    def _log_audit(
        self, event: str, document_id: str, details: dict[str, Any] | None = None
    ) -> None:
        """Log an event to the audit trail.

        Args:
            event: The audit event type.
            document_id: The associated document ID.
            details: Additional event details.
        """
        audit_entry = {
            "event": event,
            "document_id": document_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }

        if self.audit_logger is not None:
            try:
                if hasattr(self.audit_logger, "record_entry"):
                    self.audit_logger.record_entry(audit_entry)
                elif callable(self.audit_logger):
                    self.audit_logger(audit_entry)
            except Exception as e:
                logger.warning("Failed to write audit log entry: %s", e)

        logger.info("AUDIT: %s document_id=%s", event, document_id)
