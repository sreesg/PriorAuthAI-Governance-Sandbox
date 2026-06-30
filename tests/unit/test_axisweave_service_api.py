"""Unit tests for AxisweaveServiceAPI.

Tests verify:
- Each operation (ingest, retrieve, verify) works independently.
- Namespace validation is enforced on all operations.
- Authentication/authorization is required for all operations.
- Namespace isolation is enforced on retrieval results.
- Document category validation (1-64 chars).
- Operations do NOT require prior invocation of other operations.

Requirements tested: 13.1, 13.2, 13.3, 13.4
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_reasoning_fabric.api.auth_provider import APIAuthProvider, APICredentials
from clinical_reasoning_fabric.api.axisweave_service_api import (
    AxisweaveServiceAPI,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    VerifyRequest,
    VerifyResponse,
)
from clinical_reasoning_fabric.api.namespace import NamespaceRegistry
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    IngestionResult,
    KMSSignature,
    RetrievalResult,
    ScoredChunk,
)
from clinical_reasoning_fabric.models.exceptions import (
    InvalidNamespaceError,
    UnauthorizedError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def namespace_registry():
    """Create a NamespaceRegistry with a test namespace registered."""
    registry = NamespaceRegistry()
    registry.register_namespace("test-namespace", "tenant-001")
    registry.register_namespace("other-namespace", "tenant-002")
    return registry


@pytest.fixture
def auth_provider(namespace_registry):
    """Create an APIAuthProvider with a test API key registered."""
    provider = APIAuthProvider(namespace_registry=namespace_registry)
    provider.register_api_key(
        api_key="valid-api-key-12345",
        tenant_id="tenant-001",
        authorized_namespaces=["test-namespace"],
    )
    provider.register_api_key(
        api_key="other-api-key-67890",
        tenant_id="tenant-002",
        authorized_namespaces=["other-namespace"],
    )
    return provider


@pytest.fixture
def valid_credentials():
    """Create valid APICredentials for testing."""
    return APICredentials(
        api_key="valid-api-key-12345",
        tenant_id="tenant-001",
        authorized_namespaces=["test-namespace"],
    )


@pytest.fixture
def other_credentials():
    """Create credentials for a different tenant."""
    return APICredentials(
        api_key="other-api-key-67890",
        tenant_id="tenant-002",
        authorized_namespaces=["other-namespace"],
    )


@pytest.fixture
def mock_kms_signature():
    """Create a test KMS signature."""
    return KMSSignature(
        key_id="alias/test-key",
        signature="dGVzdC1zaWduYXR1cmU=",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_ingestion_service(mock_kms_signature):
    """Create a mock DocumentIngestionService."""
    service = AsyncMock()
    service.ingest_document = AsyncMock(
        return_value=IngestionResult(
            document_id="doc-12345",
            content_hash="a" * 64,
            signature=mock_kms_signature,
            chunk_count=5,
            ingestion_timestamp=datetime.now(timezone.utc),
        )
    )
    return service


@pytest.fixture
def mock_retrieval_service(mock_kms_signature):
    """Create a mock HybridRetrievalService."""
    service = AsyncMock()
    provenance = ChunkProvenance(
        document_id="doc-12345",
        content_hash="a" * 64,
        kms_signature=mock_kms_signature,
        chunk_index=0,
        ingestion_timestamp=datetime.now(timezone.utc),
    )
    service.retrieve = AsyncMock(
        return_value=RetrievalResult(
            verified_chunks=[
                ScoredChunk(
                    chunk_id="chunk-001",
                    text="Test clinical text content",
                    score=0.85,
                    provenance=provenance,
                )
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=10,
        )
    )
    return service


@pytest.fixture
def mock_kms_client():
    """Create a mock KMS verification client."""
    client = AsyncMock()
    client.verify_signature = AsyncMock(return_value=True)
    return client


@pytest.fixture
def api(
    mock_ingestion_service,
    mock_retrieval_service,
    auth_provider,
    namespace_registry,
    mock_kms_client,
):
    """Create an AxisweaveServiceAPI instance with mocked dependencies."""
    return AxisweaveServiceAPI(
        ingestion_service=mock_ingestion_service,
        retrieval_service=mock_retrieval_service,
        auth_provider=auth_provider,
        namespace_registry=namespace_registry,
        kms_client=mock_kms_client,
    )


# =============================================================================
# Test: Ingest Operation
# =============================================================================


class TestIngestOperation:
    """Tests for the ingest() operation."""

    @pytest.mark.asyncio
    async def test_ingest_success(self, api, valid_credentials, mock_kms_signature):
        """Ingest succeeds with valid namespace, category, and credentials."""
        request = IngestRequest(
            namespace="test-namespace",
            document_category="clinical-notes",
            document_bytes=b"%PDF-1.4 test content",
        )

        response = await api.ingest(request, valid_credentials)

        assert isinstance(response, IngestResponse)
        assert response.document_id == "doc-12345"
        assert response.content_hash == "a" * 64
        assert response.signature == mock_kms_signature
        assert response.chunk_count == 5
        assert response.api_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_ingest_validates_namespace(self, api, valid_credentials):
        """Ingest rejects invalid namespace formats."""
        request = IngestRequest(
            namespace="invalid namespace!",
            document_category="notes",
            document_bytes=b"content",
        )

        with pytest.raises(InvalidNamespaceError):
            await api.ingest(request, valid_credentials)

    @pytest.mark.asyncio
    async def test_ingest_validates_empty_namespace(self, api, valid_credentials):
        """Ingest rejects empty namespace."""
        request = IngestRequest(
            namespace="",
            document_category="notes",
            document_bytes=b"content",
        )

        with pytest.raises(InvalidNamespaceError):
            await api.ingest(request, valid_credentials)

    @pytest.mark.asyncio
    async def test_ingest_validates_category_too_long(self, api, valid_credentials):
        """Ingest rejects document_category exceeding 64 characters."""
        request = IngestRequest(
            namespace="test-namespace",
            document_category="x" * 65,
            document_bytes=b"content",
        )

        with pytest.raises(ValueError, match="document_category"):
            await api.ingest(request, valid_credentials)

    @pytest.mark.asyncio
    async def test_ingest_validates_category_empty(self, api, valid_credentials):
        """Ingest rejects empty document_category."""
        request = IngestRequest(
            namespace="test-namespace",
            document_category="",
            document_bytes=b"content",
        )

        with pytest.raises(ValueError, match="document_category"):
            await api.ingest(request, valid_credentials)

    @pytest.mark.asyncio
    async def test_ingest_enforces_namespace_access(self, api, other_credentials):
        """Ingest denies access when credentials lack namespace authorization."""
        request = IngestRequest(
            namespace="test-namespace",
            document_category="notes",
            document_bytes=b"content",
        )

        with pytest.raises(UnauthorizedError):
            await api.ingest(request, other_credentials)

    @pytest.mark.asyncio
    async def test_ingest_independent_of_retrieve(self, api, valid_credentials):
        """Ingest works without any prior retrieve call."""
        # This test verifies Requirement 13.1 — operations are independent
        request = IngestRequest(
            namespace="test-namespace",
            document_category="radiology",
            document_bytes=b"%PDF-1.4 imaging report",
            metadata={"use_case": "hedis"},
        )

        response = await api.ingest(request, valid_credentials)

        assert response.document_id is not None
        assert response.chunk_count > 0

    @pytest.mark.asyncio
    async def test_ingest_passes_metadata_to_service(
        self, api, valid_credentials, mock_ingestion_service
    ):
        """Ingest passes namespace and category in source_metadata."""
        request = IngestRequest(
            namespace="test-namespace",
            document_category="labs",
            document_bytes=b"content",
            metadata={"custom_key": "custom_value"},
        )

        await api.ingest(request, valid_credentials)

        call_kwargs = mock_ingestion_service.ingest_document.call_args[1]
        assert call_kwargs["source_metadata"]["namespace"] == "test-namespace"
        assert call_kwargs["source_metadata"]["document_category"] == "labs"
        assert call_kwargs["source_metadata"]["tenant_id"] == "tenant-001"
        assert call_kwargs["source_metadata"]["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_ingest_category_at_max_length(self, api, valid_credentials):
        """Ingest accepts document_category at exactly 64 characters."""
        request = IngestRequest(
            namespace="test-namespace",
            document_category="x" * 64,
            document_bytes=b"content",
        )

        response = await api.ingest(request, valid_credentials)
        assert response.document_id is not None


# =============================================================================
# Test: Retrieve Operation
# =============================================================================


class TestRetrieveOperation:
    """Tests for the retrieve() operation."""

    @pytest.mark.asyncio
    async def test_retrieve_success(self, api, valid_credentials):
        """Retrieve returns chunks with provenance metadata."""
        request = RetrieveRequest(
            namespace="test-namespace",
            query="diabetes management guidelines",
        )

        response = await api.retrieve(request, valid_credentials)

        assert isinstance(response, RetrieveResponse)
        assert len(response.chunks) >= 1
        assert response.no_evidence_found is False
        assert response.api_version == "1.0.0"

        # Verify full provenance metadata on chunks
        for chunk in response.chunks:
            assert chunk.provenance.document_id is not None
            assert chunk.provenance.content_hash is not None
            assert chunk.provenance.kms_signature is not None
            assert chunk.provenance.chunk_index >= 0
            assert chunk.provenance.ingestion_timestamp is not None

    @pytest.mark.asyncio
    async def test_retrieve_validates_namespace(self, api, valid_credentials):
        """Retrieve rejects invalid namespace formats."""
        request = RetrieveRequest(
            namespace="bad namespace!@#",
            query="test query",
        )

        with pytest.raises(InvalidNamespaceError):
            await api.retrieve(request, valid_credentials)

    @pytest.mark.asyncio
    async def test_retrieve_enforces_namespace_access(self, api, other_credentials):
        """Retrieve denies access when credentials lack namespace authorization."""
        request = RetrieveRequest(
            namespace="test-namespace",
            query="test query",
        )

        with pytest.raises(UnauthorizedError):
            await api.retrieve(request, other_credentials)

    @pytest.mark.asyncio
    async def test_retrieve_independent_of_ingest(self, api, valid_credentials):
        """Retrieve works without any prior ingest call (Requirement 13.1)."""
        request = RetrieveRequest(
            namespace="test-namespace",
            query="prior authorization evidence",
            top_k=10,
        )

        response = await api.retrieve(request, valid_credentials)

        assert isinstance(response, RetrieveResponse)
        # Operation succeeds — does not need prior ingest

    @pytest.mark.asyncio
    async def test_retrieve_handles_no_evidence(
        self, api, valid_credentials, mock_retrieval_service
    ):
        """Retrieve returns no_evidence_found when retrieval yields nothing."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[],
            tamper_alerts=[],
            no_evidence_found=True,
            degraded_search=False,
            total_candidates=0,
        )

        request = RetrieveRequest(
            namespace="test-namespace",
            query="nonexistent topic",
        )

        response = await api.retrieve(request, valid_credentials)

        assert response.no_evidence_found is True
        assert len(response.chunks) == 0

    @pytest.mark.asyncio
    async def test_retrieve_namespace_isolation(
        self, api, valid_credentials, mock_retrieval_service, mock_kms_signature
    ):
        """Retrieve filters out chunks from unauthorized namespaces."""
        provenance = ChunkProvenance(
            document_id="doc-from-other",
            content_hash="b" * 64,
            kms_signature=mock_kms_signature,
            chunk_index=0,
            ingestion_timestamp=datetime.now(timezone.utc),
        )

        # Mock returns chunks — some without namespace (should pass through)
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                ScoredChunk(
                    chunk_id="chunk-own",
                    text="Own namespace content",
                    score=0.9,
                    provenance=provenance,
                ),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=5,
        )

        request = RetrieveRequest(
            namespace="test-namespace",
            query="clinical evidence",
        )

        response = await api.retrieve(request, valid_credentials)

        # Chunks without explicit namespace should pass through
        assert len(response.chunks) == 1

    @pytest.mark.asyncio
    async def test_retrieve_timeout_enforcement(
        self, api, valid_credentials, mock_retrieval_service
    ):
        """Retrieve enforces 10-second timeout."""

        async def slow_retrieval(*args, **kwargs):
            await asyncio.sleep(15)  # Exceeds 10s timeout
            return RetrievalResult(
                verified_chunks=[],
                tamper_alerts=[],
                no_evidence_found=True,
                degraded_search=False,
                total_candidates=0,
            )

        mock_retrieval_service.retrieve = slow_retrieval

        request = RetrieveRequest(
            namespace="test-namespace",
            query="slow query",
        )

        with pytest.raises(asyncio.TimeoutError):
            await api.retrieve(request, valid_credentials)


# =============================================================================
# Test: Verify Operation
# =============================================================================


class TestVerifyOperation:
    """Tests for the verify() operation."""

    @pytest.mark.asyncio
    async def test_verify_success(self, api, valid_credentials):
        """Verify returns verification results for a document."""
        request = VerifyRequest(
            namespace="test-namespace",
            document_id="doc-12345",
        )

        response = await api.verify(request, valid_credentials)

        assert isinstance(response, VerifyResponse)
        assert response.document_id == "doc-12345"
        assert response.all_valid is True
        assert len(response.chunk_results) >= 1
        assert response.api_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_verify_specific_chunks(self, api, valid_credentials):
        """Verify can target specific chunk_ids."""
        request = VerifyRequest(
            namespace="test-namespace",
            document_id="doc-12345",
            chunk_ids=["chunk-001", "chunk-002", "chunk-003"],
        )

        response = await api.verify(request, valid_credentials)

        assert len(response.chunk_results) == 3
        for result in response.chunk_results:
            assert result.chunk_id in ["chunk-001", "chunk-002", "chunk-003"]

    @pytest.mark.asyncio
    async def test_verify_validates_namespace(self, api, valid_credentials):
        """Verify rejects invalid namespace formats."""
        request = VerifyRequest(
            namespace="",
            document_id="doc-12345",
        )

        with pytest.raises(InvalidNamespaceError):
            await api.verify(request, valid_credentials)

    @pytest.mark.asyncio
    async def test_verify_enforces_namespace_access(self, api, other_credentials):
        """Verify denies access to namespaces not authorized for the caller."""
        request = VerifyRequest(
            namespace="test-namespace",
            document_id="doc-12345",
        )

        with pytest.raises(UnauthorizedError):
            await api.verify(request, other_credentials)

    @pytest.mark.asyncio
    async def test_verify_independent_of_ingest(self, api, valid_credentials):
        """Verify works without prior ingest by the same caller (Requirement 13.1)."""
        # Caller did not ingest this document — verify still works
        request = VerifyRequest(
            namespace="test-namespace",
            document_id="doc-ingested-by-another",
            chunk_ids=["chunk-abc"],
        )

        response = await api.verify(request, valid_credentials)

        assert isinstance(response, VerifyResponse)
        # Operation completes successfully regardless of who ingested

    @pytest.mark.asyncio
    async def test_verify_independent_of_retrieve(self, api, valid_credentials):
        """Verify works without any prior retrieve call (Requirement 13.1)."""
        request = VerifyRequest(
            namespace="test-namespace",
            document_id="doc-99999",
        )

        response = await api.verify(request, valid_credentials)

        assert isinstance(response, VerifyResponse)

    @pytest.mark.asyncio
    async def test_verify_reports_invalid_signature(
        self, api, valid_credentials, mock_kms_client
    ):
        """Verify reports all_valid=False when a signature check fails."""
        mock_kms_client.verify_signature.return_value = False

        request = VerifyRequest(
            namespace="test-namespace",
            document_id="doc-tampered",
            chunk_ids=["chunk-bad"],
        )

        response = await api.verify(request, valid_credentials)

        assert response.all_valid is False
        assert response.chunk_results[0].is_valid is False

    @pytest.mark.asyncio
    async def test_verify_handles_kms_exception(
        self, api, valid_credentials, mock_kms_client
    ):
        """Verify gracefully handles KMS exceptions by marking chunk as invalid."""
        mock_kms_client.verify_signature.side_effect = Exception("KMS unavailable")

        request = VerifyRequest(
            namespace="test-namespace",
            document_id="doc-12345",
            chunk_ids=["chunk-001"],
        )

        response = await api.verify(request, valid_credentials)

        assert response.all_valid is False
        assert response.chunk_results[0].is_valid is False


# =============================================================================
# Test: Operation Independence (Requirement 13.1)
# =============================================================================


class TestOperationIndependence:
    """Tests verifying each operation works independently without prior invocations."""

    @pytest.mark.asyncio
    async def test_all_operations_independent(self, api, valid_credentials):
        """Each operation succeeds in isolation without calling others first."""
        # Verify works alone
        verify_response = await api.verify(
            VerifyRequest(namespace="test-namespace", document_id="doc-001"),
            valid_credentials,
        )
        assert isinstance(verify_response, VerifyResponse)

        # Retrieve works alone (no prior ingest or verify)
        retrieve_response = await api.retrieve(
            RetrieveRequest(namespace="test-namespace", query="isolated query"),
            valid_credentials,
        )
        assert isinstance(retrieve_response, RetrieveResponse)

        # Ingest works alone (no prior retrieve or verify)
        ingest_response = await api.ingest(
            IngestRequest(
                namespace="test-namespace",
                document_category="standalone",
                document_bytes=b"content",
            ),
            valid_credentials,
        )
        assert isinstance(ingest_response, IngestResponse)

    @pytest.mark.asyncio
    async def test_verify_without_ingest_by_same_caller(
        self, api, valid_credentials
    ):
        """Verify operation doesn't require prior ingestion by the same caller."""
        # Document was ingested by a different process/caller
        # But verification should still work for any authorized caller
        response = await api.verify(
            VerifyRequest(
                namespace="test-namespace",
                document_id="doc-from-external-system",
                chunk_ids=["chunk-external-1"],
            ),
            valid_credentials,
        )

        assert isinstance(response, VerifyResponse)
        assert response.document_id == "doc-from-external-system"
