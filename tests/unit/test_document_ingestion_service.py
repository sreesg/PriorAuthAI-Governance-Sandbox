"""Unit tests for DocumentIngestionService.

Tests the orchestration of the full ingestion pipeline:
parse → scrub → hash → sign → chunk → store.

Validates:
    - parse_pdf() raises IngestionError on corrupted/unsupported formats
    - compute_hash() produces consistent SHA-256 hashes
    - sign_hash() raises KMSUnavailableError on KMS failures
    - ingest_document() orchestrates the full pipeline correctly
    - On KMS failure: halts ingestion and discards unsigned chunks
"""

import base64
import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_reasoning_fabric.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)
from clinical_reasoning_fabric.ingestion.pii_scrubber import PIIScrubber
from clinical_reasoning_fabric.models.core import KMSSignature
from clinical_reasoning_fabric.models.exceptions import (
    IngestionError,
    KMSUnavailableError,
    PIIScrubError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_kms_client():
    """Mock boto3 KMS client that returns valid sign responses."""
    client = MagicMock()
    client.sign.return_value = {
        "KeyId": "arn:aws:kms:us-east-1:123456789:key/test-key-id",
        "Signature": base64.b64decode(
            base64.b64encode(b"mock-signature-bytes")
        ),
        "SigningAlgorithm": "RSASSA_PKCS1_V1_5_SHA_256",
    }
    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = MagicMock()
    client.upsert = MagicMock()
    return client


@pytest.fixture
def pii_scrubber():
    """Real PIIScrubber instance."""
    return PIIScrubber()


@pytest.fixture
def mock_chunker():
    """Mock chunker that splits text into paragraphs."""
    chunker = MagicMock()
    chunker.chunk.side_effect = lambda text: [
        p.strip() for p in text.split("\n\n") if p.strip()
    ] or [text]
    return chunker


@pytest.fixture
def mock_pdf_parser():
    """Mock PDF parser that returns text from valid PDF bytes."""
    parser = MagicMock()
    parser.parse.return_value = "This is extracted clinical text from a PDF document."
    return parser


@pytest.fixture
def service(mock_kms_client, mock_qdrant_client, pii_scrubber, mock_chunker, mock_pdf_parser):
    """Create a DocumentIngestionService with mocked dependencies."""
    return DocumentIngestionService(
        kms_client=mock_kms_client,
        qdrant_client=mock_qdrant_client,
        pii_scrubber=pii_scrubber,
        chunker=mock_chunker,
        pdf_parser=mock_pdf_parser,
    )


@pytest.fixture
def valid_pdf_bytes():
    """Simulated valid PDF bytes (starts with %PDF- magic)."""
    return b"%PDF-1.4 fake pdf content for testing purposes"


# =============================================================================
# Tests for parse_pdf()
# =============================================================================


class TestParsePdf:
    """Tests for DocumentIngestionService.parse_pdf()."""

    async def test_parse_pdf_with_valid_pdf(self, service, valid_pdf_bytes):
        """Valid PDF bytes are parsed successfully."""
        result = await service.parse_pdf(valid_pdf_bytes)
        assert result == "This is extracted clinical text from a PDF document."

    async def test_parse_pdf_raises_on_empty_bytes(self, service):
        """Empty document bytes raise IngestionError."""
        with pytest.raises(IngestionError) as exc_info:
            await service.parse_pdf(b"")
        assert "empty" in exc_info.value.reason.lower()

    async def test_parse_pdf_raises_on_non_pdf_format(self, service):
        """Non-PDF bytes (wrong magic bytes) raise IngestionError."""
        with pytest.raises(IngestionError) as exc_info:
            await service.parse_pdf(b"This is not a PDF file at all")
        assert "unsupported" in exc_info.value.reason.lower() or "not a valid PDF" in exc_info.value.reason

    async def test_parse_pdf_raises_on_corrupted_pdf(self, service, mock_pdf_parser):
        """Corrupted PDF that fails parsing raises IngestionError."""
        mock_pdf_parser.parse.side_effect = RuntimeError("Corrupted PDF structure")
        corrupted = b"%PDF-1.4 corrupted content"

        with pytest.raises(IngestionError) as exc_info:
            await service.parse_pdf(corrupted)
        assert "parsing failed" in exc_info.value.reason.lower()

    async def test_parse_pdf_raises_on_empty_extracted_text(self, service, mock_pdf_parser):
        """PDF that produces empty text raises IngestionError."""
        mock_pdf_parser.parse.return_value = "   "
        pdf_bytes = b"%PDF-1.4 image-only pdf"

        with pytest.raises(IngestionError) as exc_info:
            await service.parse_pdf(pdf_bytes)
        assert "empty text" in exc_info.value.reason.lower()


# =============================================================================
# Tests for compute_hash()
# =============================================================================


class TestComputeHash:
    """Tests for DocumentIngestionService.compute_hash()."""

    async def test_compute_hash_produces_valid_sha256(self, service):
        """compute_hash() returns a 64-char hex SHA-256 digest."""
        result = await service.compute_hash("test clinical text")
        assert len(result) == 64
        # Verify it's valid hex
        int(result, 16)

    async def test_compute_hash_deterministic(self, service):
        """Same input always produces same hash."""
        text = "Patient presents with chronic lower back pain."
        hash1 = await service.compute_hash(text)
        hash2 = await service.compute_hash(text)
        assert hash1 == hash2

    async def test_compute_hash_matches_hashlib(self, service):
        """Output matches direct hashlib computation."""
        text = "Clinical note text for hashing"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result = await service.compute_hash(text)
        assert result == expected

    async def test_compute_hash_different_for_different_inputs(self, service):
        """Different texts produce different hashes."""
        hash1 = await service.compute_hash("text one")
        hash2 = await service.compute_hash("text two")
        assert hash1 != hash2

    async def test_compute_hash_empty_string(self, service):
        """Empty string still produces a valid hash."""
        result = await service.compute_hash("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected


# =============================================================================
# Tests for sign_hash()
# =============================================================================


class TestSignHash:
    """Tests for DocumentIngestionService.sign_hash()."""

    async def test_sign_hash_returns_kms_signature(self, service):
        """Successful signing returns a KMSSignature object."""
        content_hash = hashlib.sha256(b"test").hexdigest()
        result = await service.sign_hash(content_hash)

        assert isinstance(result, KMSSignature)
        assert result.key_id == "arn:aws:kms:us-east-1:123456789:key/test-key-id"
        assert result.algorithm == "RSASSA_PKCS1_V1_5_SHA_256"
        assert result.signature  # non-empty
        assert result.signed_at is not None

    async def test_sign_hash_calls_kms_with_correct_params(self, service, mock_kms_client):
        """sign_hash() passes correct parameters to KMS."""
        content_hash = "a" * 64
        await service.sign_hash(content_hash)

        mock_kms_client.sign.assert_called_once_with(
            KeyId="alias/clinical-document-signing",
            Message=content_hash.encode("utf-8"),
            MessageType="RAW",
            SigningAlgorithm="RSASSA_PKCS1_V1_5_SHA_256",
        )

    async def test_sign_hash_raises_kms_unavailable_on_client_error(self, service, mock_kms_client):
        """KMS client error raises KMSUnavailableError."""
        mock_kms_client.sign.side_effect = Exception("Connection refused")

        with pytest.raises(KMSUnavailableError) as exc_info:
            await service.sign_hash("a" * 64)
        assert "signing failed" in exc_info.value.reason.lower() or "error" in exc_info.value.reason.lower()

    async def test_sign_hash_raises_kms_unavailable_on_timeout(self, service, mock_kms_client):
        """KMS timeout raises KMSUnavailableError."""

        class ReadTimeoutError(Exception):
            pass

        mock_kms_client.sign.side_effect = ReadTimeoutError("Read timed out")

        with pytest.raises(KMSUnavailableError):
            await service.sign_hash("b" * 64)


# =============================================================================
# Tests for ingest_document() orchestration
# =============================================================================


class TestIngestDocument:
    """Tests for DocumentIngestionService.ingest_document() pipeline orchestration."""

    async def test_ingest_document_success(self, service, valid_pdf_bytes):
        """Full pipeline succeeds with valid inputs."""
        result = await service.ingest_document(
            document_bytes=valid_pdf_bytes,
            document_id="doc-001",
            source_metadata={"source": "test"},
        )

        assert result.document_id == "doc-001"
        assert len(result.content_hash) == 64
        assert isinstance(result.signature, KMSSignature)
        assert result.chunk_count >= 1
        assert result.ingestion_timestamp is not None

    async def test_ingest_document_halts_on_parse_failure(self, service, mock_pdf_parser):
        """Pipeline halts on parsing failure without continuing."""
        mock_pdf_parser.parse.side_effect = RuntimeError("Corrupted")
        pdf_bytes = b"%PDF-1.4 corrupted"

        with pytest.raises(IngestionError):
            await service.ingest_document(pdf_bytes, "doc-002")

    async def test_ingest_document_halts_on_pii_scrub_failure(
        self, service, valid_pdf_bytes, mock_pdf_parser
    ):
        """Pipeline halts when PII scrubbing fails."""
        # Make the parser return text that will cause scrub to fail
        mock_pdf_parser.parse.return_value = "Valid text"
        service.pii_scrubber = MagicMock()
        service.pii_scrubber.scrub.side_effect = PIIScrubError(
            reason="Scrubbing failed"
        )

        with pytest.raises(PIIScrubError):
            await service.ingest_document(valid_pdf_bytes, "doc-003")

    async def test_ingest_document_halts_on_kms_failure(
        self, service, valid_pdf_bytes, mock_kms_client, mock_qdrant_client
    ):
        """Pipeline halts on KMS failure, discards unsigned chunks.

        Requirement 1.7: On KMS failure, halt ingestion, discard unsigned chunks.
        """
        mock_kms_client.sign.side_effect = Exception("KMS service unavailable")

        with pytest.raises(KMSUnavailableError):
            await service.ingest_document(valid_pdf_bytes, "doc-004")

        # Verify no chunks were stored (Qdrant upsert should not be called)
        mock_qdrant_client.upsert.assert_not_called()

    async def test_ingest_document_pipeline_order(
        self, service, valid_pdf_bytes, mock_kms_client, mock_pdf_parser
    ):
        """Verify pipeline executes in correct order: parse → scrub → hash → sign → chunk → store."""
        call_order = []

        # Set up parse mock with a fresh side_effect (no recursion)
        mock_pdf_parser.parse.side_effect = lambda *a, **kw: (
            call_order.append("parse"),
            "Parsed clinical text for ordering test",
        )[-1]

        # Replace scrubber with a mock that records calls
        service.pii_scrubber = MagicMock()
        service.pii_scrubber.scrub.side_effect = lambda text: (
            call_order.append("scrub"),
            text,
        )[-1]

        # Capture the original return value for sign
        sign_return = {
            "KeyId": "arn:aws:kms:us-east-1:123456789:key/test-key-id",
            "Signature": b"mock-signature-bytes",
            "SigningAlgorithm": "RSASSA_PKCS1_V1_5_SHA_256",
        }

        def sign_side_effect(*args, **kwargs):
            call_order.append("sign")
            return sign_return

        mock_kms_client.sign.side_effect = sign_side_effect

        def chunk_side_effect(text):
            call_order.append("chunk")
            return [text]

        service.chunker.chunk.side_effect = chunk_side_effect

        await service.ingest_document(valid_pdf_bytes, "doc-005")

        assert call_order == ["parse", "scrub", "sign", "chunk"]

    async def test_ingest_document_logs_kms_failure_to_audit(
        self, service, valid_pdf_bytes, mock_kms_client
    ):
        """On KMS failure, audit trail is logged."""
        audit_entries = []
        service.audit_logger = lambda entry: audit_entries.append(entry)
        mock_kms_client.sign.side_effect = Exception("KMS down")

        with pytest.raises(KMSUnavailableError):
            await service.ingest_document(valid_pdf_bytes, "doc-006")

        # Verify audit was logged
        assert len(audit_entries) >= 1
        kms_failure_entry = next(
            (e for e in audit_entries if e["event"] == "kms_signing_failure"), None
        )
        assert kms_failure_entry is not None
        assert kms_failure_entry["document_id"] == "doc-006"

    async def test_ingest_document_with_no_metadata(self, service, valid_pdf_bytes):
        """Pipeline works when source_metadata is None."""
        result = await service.ingest_document(
            document_bytes=valid_pdf_bytes,
            document_id="doc-007",
            source_metadata=None,
        )
        assert result.document_id == "doc-007"
        assert result.chunk_count >= 1
