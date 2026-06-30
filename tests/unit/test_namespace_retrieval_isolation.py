"""Unit tests for namespace-scoped retrieval isolation and cross-namespace access.

Tests verify:
- All returned chunks have namespace metadata matching caller's authorized namespaces
- Shared vector index partitioning: namespace-scoped access control applied before returning results
- Cross-namespace access requires explicit grant in caller token/API key scope
- No chunks from unauthorized namespaces included in responses regardless of shared index config

Requirements tested: 13.5, 13.7
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical_reasoning_fabric.api.auth_provider import APIAuthProvider, APICredentials
from clinical_reasoning_fabric.api.axisweave_service_api import (
    AxisweaveServiceAPI,
    RetrieveRequest,
    RetrieveResponse,
)
from clinical_reasoning_fabric.api.namespace import NamespaceRegistry
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    IngestionResult,
    KMSSignature,
    RetrievalResult,
    ScoredChunk,
)
from clinical_reasoning_fabric.models.exceptions import UnauthorizedError


# =============================================================================
# Helpers
# =============================================================================


def _make_kms_signature() -> KMSSignature:
    """Create a valid KMS signature for testing."""
    return KMSSignature(
        signature="dGVzdHNpZ25hdHVyZQ==",
        key_id="arn:aws:kms:us-east-1:123456789012:key/test-key",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_provenance(doc_id: str = "doc-001") -> ChunkProvenance:
    """Create a valid ChunkProvenance for testing."""
    return ChunkProvenance(
        document_id=doc_id,
        content_hash="a" * 64,
        kms_signature=_make_kms_signature(),
        chunk_index=0,
        ingestion_timestamp=datetime.now(timezone.utc),
    )


def _make_chunk(
    chunk_id: str,
    namespace: str | None = None,
    score: float = 0.8,
    doc_id: str = "doc-001",
) -> ScoredChunk:
    """Create a ScoredChunk with optional namespace for testing."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"Content of {chunk_id}",
        score=score,
        provenance=_make_provenance(doc_id),
        namespace=namespace,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def namespace_registry():
    """Registry with multiple namespaces for isolation testing."""
    registry = NamespaceRegistry()
    registry.register_namespace("namespace-a", "tenant-alpha")
    registry.register_namespace("namespace-b", "tenant-beta")
    registry.register_namespace("namespace-c", "tenant-gamma")
    registry.register_namespace("shared-ns", "tenant-alpha")
    return registry


@pytest.fixture
def auth_provider(namespace_registry):
    """Auth provider with multiple tenants and API keys."""
    provider = APIAuthProvider(namespace_registry=namespace_registry)
    # tenant-alpha has access to namespace-a and shared-ns
    provider.register_api_key(
        api_key="alpha-key-123456789",
        tenant_id="tenant-alpha",
        authorized_namespaces=["namespace-a"],
    )
    # tenant-beta has access to namespace-b only
    provider.register_api_key(
        api_key="beta-key-987654321",
        tenant_id="tenant-beta",
        authorized_namespaces=["namespace-b"],
    )
    # tenant-gamma has explicit cross-namespace grant to namespace-a and namespace-c
    provider.register_api_key(
        api_key="gamma-key-111222333",
        tenant_id="tenant-gamma",
        authorized_namespaces=["namespace-c", "namespace-a"],
    )
    return provider


@pytest.fixture
def credentials_alpha():
    """Credentials for tenant-alpha (authorized: namespace-a)."""
    return APICredentials(
        api_key="alpha-key-123456789",
        tenant_id="tenant-alpha",
        authorized_namespaces=["namespace-a"],
    )


@pytest.fixture
def credentials_beta():
    """Credentials for tenant-beta (authorized: namespace-b)."""
    return APICredentials(
        api_key="beta-key-987654321",
        tenant_id="tenant-beta",
        authorized_namespaces=["namespace-b"],
    )


@pytest.fixture
def credentials_gamma():
    """Credentials for tenant-gamma (authorized: namespace-c, namespace-a)."""
    return APICredentials(
        api_key="gamma-key-111222333",
        tenant_id="tenant-gamma",
        authorized_namespaces=["namespace-c", "namespace-a"],
    )


@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service returning configurable results."""
    service = AsyncMock()
    service.retrieve = AsyncMock(
        return_value=RetrievalResult(
            verified_chunks=[],
            tamper_alerts=[],
            no_evidence_found=True,
            degraded_search=False,
            total_candidates=0,
        )
    )
    return service


@pytest.fixture
def mock_ingestion_service():
    """Mock ingestion service."""
    return AsyncMock()


@pytest.fixture
def api(mock_ingestion_service, mock_retrieval_service, auth_provider, namespace_registry):
    """Configured AxisweaveServiceAPI for testing."""
    return AxisweaveServiceAPI(
        ingestion_service=mock_ingestion_service,
        retrieval_service=mock_retrieval_service,
        auth_provider=auth_provider,
        namespace_registry=namespace_registry,
    )


# =============================================================================
# Test: Namespace-Scoped Retrieval Isolation
# =============================================================================


class TestNamespaceScopedRetrievalIsolation:
    """Tests that retrieval only returns chunks from authorized namespaces.

    Requirement 13.5: Enforce tenant and namespace isolation such that retrieval
    queries return only documents ingested under that use case namespace.
    Requirement 13.7: Shared vector index — partition results by namespace metadata.
    """

    @pytest.mark.asyncio
    async def test_chunks_from_authorized_namespace_are_returned(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Chunks matching caller's namespace are included in response."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-1", namespace="namespace-a"),
                _make_chunk("chunk-2", namespace="namespace-a"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=2,
        )

        request = RetrieveRequest(namespace="namespace-a", query="clinical data")
        response = await api.retrieve(request, credentials_alpha)

        assert len(response.chunks) == 2
        assert all(c.namespace == "namespace-a" for c in response.chunks)

    @pytest.mark.asyncio
    async def test_chunks_from_unauthorized_namespace_are_excluded(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Chunks from unauthorized namespaces are excluded from response.

        Simulates shared vector index where chunks from multiple namespaces
        exist in the same index.
        """
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-own", namespace="namespace-a"),
                _make_chunk("chunk-other-b", namespace="namespace-b"),
                _make_chunk("chunk-other-c", namespace="namespace-c"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=3,
        )

        request = RetrieveRequest(namespace="namespace-a", query="clinical data")
        response = await api.retrieve(request, credentials_alpha)

        # Only chunk from namespace-a should be returned
        assert len(response.chunks) == 1
        assert response.chunks[0].chunk_id == "chunk-own"
        assert response.chunks[0].namespace == "namespace-a"

    @pytest.mark.asyncio
    async def test_all_unauthorized_chunks_excluded_in_shared_index(
        self, api, credentials_beta, mock_retrieval_service
    ):
        """In a shared index, ALL chunks from unauthorized namespaces are excluded."""
        # Simulate shared index returning chunks from multiple namespaces
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-a1", namespace="namespace-a"),
                _make_chunk("chunk-a2", namespace="namespace-a"),
                _make_chunk("chunk-b1", namespace="namespace-b"),
                _make_chunk("chunk-b2", namespace="namespace-b"),
                _make_chunk("chunk-c1", namespace="namespace-c"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=5,
        )

        request = RetrieveRequest(namespace="namespace-b", query="patient records")
        response = await api.retrieve(request, credentials_beta)

        # Only namespace-b chunks should be returned
        assert len(response.chunks) == 2
        assert all(c.namespace == "namespace-b" for c in response.chunks)
        chunk_ids = [c.chunk_id for c in response.chunks]
        assert "chunk-b1" in chunk_ids
        assert "chunk-b2" in chunk_ids

    @pytest.mark.asyncio
    async def test_no_evidence_found_when_all_chunks_unauthorized(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Response indicates no evidence when all chunks are from unauthorized namespaces."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-b1", namespace="namespace-b"),
                _make_chunk("chunk-c1", namespace="namespace-c"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=2,
        )

        request = RetrieveRequest(namespace="namespace-a", query="clinical data")
        response = await api.retrieve(request, credentials_alpha)

        assert len(response.chunks) == 0
        assert response.no_evidence_found is True

    @pytest.mark.asyncio
    async def test_chunks_without_namespace_pass_through(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Chunks without namespace metadata pass through (query-level scoping assumed)."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-no-ns-1", namespace=None),
                _make_chunk("chunk-no-ns-2", namespace=None),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=2,
        )

        request = RetrieveRequest(namespace="namespace-a", query="clinical data")
        response = await api.retrieve(request, credentials_alpha)

        # Chunks without namespace pass through (retrieval was already scoped)
        assert len(response.chunks) == 2

    @pytest.mark.asyncio
    async def test_mixed_chunks_with_and_without_namespace(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Mix of chunks with namespace and without — correct filtering applied."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-a-tagged", namespace="namespace-a"),
                _make_chunk("chunk-no-ns", namespace=None),
                _make_chunk("chunk-b-tagged", namespace="namespace-b"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=3,
        )

        request = RetrieveRequest(namespace="namespace-a", query="clinical data")
        response = await api.retrieve(request, credentials_alpha)

        # namespace-a chunk included, None chunk passes through, namespace-b excluded
        assert len(response.chunks) == 2
        chunk_ids = [c.chunk_id for c in response.chunks]
        assert "chunk-a-tagged" in chunk_ids
        assert "chunk-no-ns" in chunk_ids
        assert "chunk-b-tagged" not in chunk_ids

    @pytest.mark.asyncio
    async def test_empty_retrieval_results_handled(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Empty retrieval results handled correctly."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[],
            tamper_alerts=[],
            no_evidence_found=True,
            degraded_search=False,
            total_candidates=0,
        )

        request = RetrieveRequest(namespace="namespace-a", query="nothing")
        response = await api.retrieve(request, credentials_alpha)

        assert len(response.chunks) == 0
        assert response.no_evidence_found is True


# =============================================================================
# Test: Cross-Namespace Access
# =============================================================================


class TestCrossNamespaceAccess:
    """Tests that cross-namespace access requires explicit grant in caller token.

    Requirement 13.5: Cross-namespace access requires explicit grant in
    caller-provided access token or API key that includes the target namespace
    in its authorized scope.
    """

    @pytest.mark.asyncio
    async def test_cross_namespace_with_explicit_grant_in_token(
        self, api, credentials_gamma, mock_retrieval_service
    ):
        """Cross-namespace retrieval succeeds when target namespace is in token scope."""
        # gamma has authorized_namespaces=["namespace-c", "namespace-a"]
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-c1", namespace="namespace-c"),
                _make_chunk("chunk-a1", namespace="namespace-a"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=2,
        )

        # Request namespace-c with cross-namespace to namespace-a
        request = RetrieveRequest(
            namespace="namespace-c",
            query="cross-namespace query",
            cross_namespace_targets=["namespace-a"],
        )
        response = await api.retrieve(request, credentials_gamma)

        # Both namespace-c and namespace-a chunks should be returned
        assert len(response.chunks) == 2
        namespaces_returned = {c.namespace for c in response.chunks}
        assert "namespace-c" in namespaces_returned
        assert "namespace-a" in namespaces_returned

    @pytest.mark.asyncio
    async def test_cross_namespace_without_grant_is_denied(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Cross-namespace access is denied when target is not in token scope."""
        # alpha has authorized_namespaces=["namespace-a"] only
        request = RetrieveRequest(
            namespace="namespace-a",
            query="cross-namespace query",
            cross_namespace_targets=["namespace-b"],
        )

        with pytest.raises(UnauthorizedError):
            await api.retrieve(request, credentials_alpha)

    @pytest.mark.asyncio
    async def test_cross_namespace_partial_grant_denied(
        self, api, credentials_gamma, mock_retrieval_service
    ):
        """If any cross-namespace target is unauthorized, entire request is denied."""
        # gamma has ["namespace-c", "namespace-a"] — NOT namespace-b
        request = RetrieveRequest(
            namespace="namespace-c",
            query="cross-namespace query",
            cross_namespace_targets=["namespace-a", "namespace-b"],
        )

        with pytest.raises(UnauthorizedError):
            await api.retrieve(request, credentials_gamma)

    @pytest.mark.asyncio
    async def test_cross_namespace_filters_unauthorized_chunks(
        self, api, credentials_gamma, mock_retrieval_service
    ):
        """Even with cross-namespace grant, chunks from non-granted namespaces excluded."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-c1", namespace="namespace-c"),
                _make_chunk("chunk-a1", namespace="namespace-a"),
                _make_chunk("chunk-b1", namespace="namespace-b"),  # NOT authorized
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=3,
        )

        request = RetrieveRequest(
            namespace="namespace-c",
            query="clinical evidence",
            cross_namespace_targets=["namespace-a"],
        )
        response = await api.retrieve(request, credentials_gamma)

        # Only namespace-c and namespace-a chunks should be in response
        assert len(response.chunks) == 2
        chunk_ids = [c.chunk_id for c in response.chunks]
        assert "chunk-c1" in chunk_ids
        assert "chunk-a1" in chunk_ids
        assert "chunk-b1" not in chunk_ids

    @pytest.mark.asyncio
    async def test_cross_namespace_empty_targets_uses_primary_only(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Empty cross_namespace_targets means only primary namespace is authorized."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-a1", namespace="namespace-a"),
                _make_chunk("chunk-b1", namespace="namespace-b"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=2,
        )

        request = RetrieveRequest(
            namespace="namespace-a",
            query="clinical data",
            cross_namespace_targets=[],
        )
        response = await api.retrieve(request, credentials_alpha)

        assert len(response.chunks) == 1
        assert response.chunks[0].namespace == "namespace-a"

    @pytest.mark.asyncio
    async def test_cross_namespace_access_via_registry_grant(
        self, api, namespace_registry, mock_retrieval_service
    ):
        """Cross-namespace access via NamespaceRegistry grant (not just token scope)."""
        # Grant namespace-a cross-namespace access to namespace-b
        namespace_registry.grant_cross_namespace_access("namespace-a", "namespace-b")

        # Create credentials without namespace-b in authorized_namespaces,
        # but the tenant owns namespace-a which now has a cross-namespace grant
        credentials = APICredentials(
            api_key="alpha-key-123456789",
            tenant_id="tenant-alpha",
            authorized_namespaces=["namespace-a"],
        )

        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-a1", namespace="namespace-a"),
                _make_chunk("chunk-b1", namespace="namespace-b"),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=2,
        )

        # Request with cross-namespace target to namespace-b
        request = RetrieveRequest(
            namespace="namespace-a",
            query="clinical data",
            cross_namespace_targets=["namespace-b"],
        )
        response = await api.retrieve(request, credentials)

        # Both namespaces accessible via registry grant
        assert len(response.chunks) == 2


# =============================================================================
# Test: Shared Vector Index Partitioning
# =============================================================================


class TestSharedVectorIndexPartitioning:
    """Tests for shared vector index where multiple namespaces coexist.

    Requirement 13.7: When a shared vector index is configured across multiple
    use-case namespaces, partition retrieval results by namespace metadata and
    apply namespace-scoped access control before returning results.
    """

    @pytest.mark.asyncio
    async def test_shared_index_many_namespaces_strict_isolation(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """In a shared index with many namespaces, strict isolation is maintained."""
        # Simulate a shared index returning chunks from 5 different namespaces
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-a1", namespace="namespace-a", score=0.95),
                _make_chunk("chunk-b1", namespace="namespace-b", score=0.92),
                _make_chunk("chunk-c1", namespace="namespace-c", score=0.90),
                _make_chunk("chunk-a2", namespace="namespace-a", score=0.88),
                _make_chunk("chunk-x1", namespace="namespace-x", score=0.85),
                _make_chunk("chunk-y1", namespace="namespace-y", score=0.80),
                _make_chunk("chunk-a3", namespace="namespace-a", score=0.75),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=7,
        )

        request = RetrieveRequest(namespace="namespace-a", query="evidence query")
        response = await api.retrieve(request, credentials_alpha)

        # Only namespace-a chunks should be returned
        assert len(response.chunks) == 3
        assert all(c.namespace == "namespace-a" for c in response.chunks)
        chunk_ids = {c.chunk_id for c in response.chunks}
        assert chunk_ids == {"chunk-a1", "chunk-a2", "chunk-a3"}

    @pytest.mark.asyncio
    async def test_shared_index_high_score_unauthorized_excluded(
        self, api, credentials_beta, mock_retrieval_service
    ):
        """High-scoring chunks from unauthorized namespaces are still excluded."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                # Top-scoring chunk is from unauthorized namespace
                _make_chunk("chunk-a-top", namespace="namespace-a", score=0.99),
                _make_chunk("chunk-b-low", namespace="namespace-b", score=0.60),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=2,
        )

        request = RetrieveRequest(namespace="namespace-b", query="test query")
        response = await api.retrieve(request, credentials_beta)

        # Only the lower-scoring namespace-b chunk should be returned
        assert len(response.chunks) == 1
        assert response.chunks[0].chunk_id == "chunk-b-low"
        assert response.chunks[0].namespace == "namespace-b"

    @pytest.mark.asyncio
    async def test_shared_index_preserves_chunk_ordering(
        self, api, credentials_alpha, mock_retrieval_service
    ):
        """Namespace filtering preserves relative ordering of authorized chunks."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-a1", namespace="namespace-a", score=0.95),
                _make_chunk("chunk-b1", namespace="namespace-b", score=0.93),
                _make_chunk("chunk-a2", namespace="namespace-a", score=0.90),
                _make_chunk("chunk-b2", namespace="namespace-b", score=0.88),
                _make_chunk("chunk-a3", namespace="namespace-a", score=0.75),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=5,
        )

        request = RetrieveRequest(namespace="namespace-a", query="ordered query")
        response = await api.retrieve(request, credentials_alpha)

        assert len(response.chunks) == 3
        # Verify ordering is preserved
        assert response.chunks[0].chunk_id == "chunk-a1"
        assert response.chunks[1].chunk_id == "chunk-a2"
        assert response.chunks[2].chunk_id == "chunk-a3"
        # Verify scores maintain order
        assert response.chunks[0].score >= response.chunks[1].score
        assert response.chunks[1].score >= response.chunks[2].score

    @pytest.mark.asyncio
    async def test_shared_index_cross_namespace_returns_both(
        self, api, credentials_gamma, mock_retrieval_service
    ):
        """Shared index with cross-namespace grant returns chunks from both namespaces."""
        mock_retrieval_service.retrieve.return_value = RetrievalResult(
            verified_chunks=[
                _make_chunk("chunk-a1", namespace="namespace-a", score=0.9),
                _make_chunk("chunk-b1", namespace="namespace-b", score=0.85),
                _make_chunk("chunk-c1", namespace="namespace-c", score=0.8),
            ],
            tamper_alerts=[],
            no_evidence_found=False,
            degraded_search=False,
            total_candidates=3,
        )

        # gamma authorized for namespace-c and namespace-a
        request = RetrieveRequest(
            namespace="namespace-c",
            query="shared index query",
            cross_namespace_targets=["namespace-a"],
        )
        response = await api.retrieve(request, credentials_gamma)

        # namespace-a and namespace-c returned, namespace-b excluded
        assert len(response.chunks) == 2
        namespaces = {c.namespace for c in response.chunks}
        assert namespaces == {"namespace-a", "namespace-c"}


# =============================================================================
# Test: _filter_by_namespace Internal Method (Direct)
# =============================================================================


class TestFilterByNamespaceMethod:
    """Direct unit tests for the _filter_by_namespace method."""

    @pytest.fixture
    def api_instance(self, mock_ingestion_service, mock_retrieval_service,
                     auth_provider, namespace_registry):
        return AxisweaveServiceAPI(
            ingestion_service=mock_ingestion_service,
            retrieval_service=mock_retrieval_service,
            auth_provider=auth_provider,
            namespace_registry=namespace_registry,
        )

    def test_filter_empty_list(self, api_instance):
        """Filtering an empty list returns empty list."""
        result = api_instance._filter_by_namespace([], ["namespace-a"])
        assert result == []

    def test_filter_all_authorized(self, api_instance):
        """All chunks with authorized namespace pass through."""
        chunks = [
            _make_chunk("c1", namespace="ns-a"),
            _make_chunk("c2", namespace="ns-a"),
        ]
        result = api_instance._filter_by_namespace(chunks, ["ns-a"])
        assert len(result) == 2

    def test_filter_all_unauthorized(self, api_instance):
        """All chunks with unauthorized namespace are excluded."""
        chunks = [
            _make_chunk("c1", namespace="ns-b"),
            _make_chunk("c2", namespace="ns-c"),
        ]
        result = api_instance._filter_by_namespace(chunks, ["ns-a"])
        assert len(result) == 0

    def test_filter_mixed_namespaces(self, api_instance):
        """Mixed authorized/unauthorized chunks are correctly partitioned."""
        chunks = [
            _make_chunk("c1", namespace="ns-a"),
            _make_chunk("c2", namespace="ns-b"),
            _make_chunk("c3", namespace="ns-a"),
            _make_chunk("c4", namespace="ns-c"),
        ]
        result = api_instance._filter_by_namespace(chunks, ["ns-a"])
        assert len(result) == 2
        assert all(c.namespace == "ns-a" for c in result)

    def test_filter_multiple_authorized_namespaces(self, api_instance):
        """Multiple authorized namespaces are all allowed."""
        chunks = [
            _make_chunk("c1", namespace="ns-a"),
            _make_chunk("c2", namespace="ns-b"),
            _make_chunk("c3", namespace="ns-c"),
        ]
        result = api_instance._filter_by_namespace(chunks, ["ns-a", "ns-b"])
        assert len(result) == 2
        namespaces = {c.namespace for c in result}
        assert namespaces == {"ns-a", "ns-b"}

    def test_filter_none_namespace_passes_through(self, api_instance):
        """Chunks with None namespace pass through (pre-scoped at query level)."""
        chunks = [
            _make_chunk("c1", namespace=None),
            _make_chunk("c2", namespace="ns-b"),
        ]
        result = api_instance._filter_by_namespace(chunks, ["ns-a"])
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    def test_filter_preserves_order(self, api_instance):
        """Filtering preserves the original order of authorized chunks."""
        chunks = [
            _make_chunk("c1", namespace="ns-a", score=0.9),
            _make_chunk("c2", namespace="ns-b", score=0.85),
            _make_chunk("c3", namespace="ns-a", score=0.8),
            _make_chunk("c4", namespace="ns-b", score=0.75),
            _make_chunk("c5", namespace="ns-a", score=0.7),
        ]
        result = api_instance._filter_by_namespace(chunks, ["ns-a"])
        assert [c.chunk_id for c in result] == ["c1", "c3", "c5"]

    def test_filter_empty_authorized_namespaces(self, api_instance):
        """Empty authorized list excludes all namespaced chunks."""
        chunks = [
            _make_chunk("c1", namespace="ns-a"),
            _make_chunk("c2", namespace=None),  # Should still pass (no namespace)
        ]
        result = api_instance._filter_by_namespace(chunks, [])
        assert len(result) == 1
        assert result[0].chunk_id == "c2"

    def test_filter_logs_excluded_chunks(self, api_instance, caplog):
        """Excluded chunks generate warning log entries."""
        import logging

        chunks = [
            _make_chunk("c1", namespace="ns-unauthorized"),
        ]
        with caplog.at_level(logging.WARNING):
            api_instance._filter_by_namespace(chunks, ["ns-allowed"])

        assert "Namespace isolation" in caplog.text
        assert "ns-unauthorized" in caplog.text


# =============================================================================
# Test: Retrieval Request Validation for Namespace Access
# =============================================================================


class TestRetrievalNamespaceValidation:
    """Tests that retrieval enforces namespace access validation before search."""

    @pytest.mark.asyncio
    async def test_retrieve_denied_for_unauthorized_primary_namespace(
        self, api, credentials_beta
    ):
        """Retrieval is denied when caller doesn't have access to primary namespace."""
        request = RetrieveRequest(namespace="namespace-a", query="unauthorized query")

        with pytest.raises(UnauthorizedError):
            await api.retrieve(request, credentials_beta)

    @pytest.mark.asyncio
    async def test_retrieve_validates_namespace_format(
        self, api, credentials_alpha
    ):
        """Retrieval rejects invalid namespace format."""
        from clinical_reasoning_fabric.models.exceptions import InvalidNamespaceError

        request = RetrieveRequest(namespace="invalid ns!@#", query="test")

        with pytest.raises(InvalidNamespaceError):
            await api.retrieve(request, credentials_alpha)

    @pytest.mark.asyncio
    async def test_retrieve_validates_cross_namespace_target_format(
        self, api, credentials_gamma
    ):
        """Cross-namespace targets must have valid namespace format."""
        from clinical_reasoning_fabric.models.exceptions import InvalidNamespaceError

        request = RetrieveRequest(
            namespace="namespace-c",
            query="test",
            cross_namespace_targets=["invalid ns!"],
        )

        with pytest.raises(InvalidNamespaceError):
            await api.retrieve(request, credentials_gamma)

    @pytest.mark.asyncio
    async def test_retrieval_not_called_when_auth_fails(
        self, api, credentials_beta, mock_retrieval_service
    ):
        """Retrieval service is never called if authentication fails."""
        request = RetrieveRequest(namespace="namespace-a", query="unauthorized")

        with pytest.raises(UnauthorizedError):
            await api.retrieve(request, credentials_beta)

        # The retrieval service should never have been called
        mock_retrieval_service.retrieve.assert_not_called()
