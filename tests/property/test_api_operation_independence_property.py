"""Property-based tests for API Operation Independence.

**Validates: Requirements 13.1**

Property 24: API Operation Independence
- For any valid ingest, retrieve, or verify request with proper auth and valid
  namespace, the operation succeeds without requiring prior invocation of any
  other operation.
- Each of the three API operations (ingest, retrieve, verify) is independently
  callable and does not depend on the execution of the others.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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


# =============================================================================
# Constants
# =============================================================================

VALID_NAMESPACE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
VALID_CATEGORY_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_."
TEST_TENANT_ID = "tenant-independence-test"
TEST_API_KEY = "independence-test-key-001"
TEST_NAMESPACE = "independence-ns"


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Valid namespace identifiers (1-128 alphanumeric/hyphen/underscore)
valid_namespace_strategy = st.text(
    alphabet=VALID_NAMESPACE_ALPHABET,
    min_size=1,
    max_size=64,  # Keep shorter for practical test execution
)

# Valid document category (1-64 characters)
valid_category_strategy = st.text(
    alphabet=VALID_CATEGORY_ALPHABET,
    min_size=1,
    max_size=64,
).filter(lambda s: s.strip() != "")

# Valid document bytes (non-empty)
valid_document_bytes_strategy = st.binary(min_size=1, max_size=1024)

# Valid query strings (non-empty)
valid_query_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

# Valid document IDs (UUID-like strings)
valid_document_id_strategy = st.uuids().map(str)

# Valid top_k values
valid_top_k_strategy = st.integers(min_value=1, max_value=50)

# Valid min_score values
valid_min_score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# Helpers
# =============================================================================


def _make_kms_signature() -> KMSSignature:
    """Create a valid KMSSignature for mock responses."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789:key/test-key-id",
        signature="dGVzdC1zaWduYXR1cmUtYmFzZTY0",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_chunk_provenance(document_id: str = "doc-001") -> ChunkProvenance:
    """Create a valid ChunkProvenance for mock responses."""
    return ChunkProvenance(
        document_id=document_id,
        content_hash="a" * 64,
        kms_signature=_make_kms_signature(),
        chunk_index=0,
        ingestion_timestamp=datetime.now(timezone.utc),
    )


def _make_scored_chunk(chunk_id: str = "chunk-001") -> ScoredChunk:
    """Create a valid ScoredChunk for mock retrieval responses."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text="Sample clinical text for testing independence.",
        score=0.85,
        provenance=_make_chunk_provenance(),
    )


def _make_ingestion_result(document_id: str = "doc-001") -> IngestionResult:
    """Create a valid IngestionResult for mock ingestion responses."""
    return IngestionResult(
        document_id=document_id,
        content_hash="b" * 64,
        signature=_make_kms_signature(),
        chunk_count=3,
        ingestion_timestamp=datetime.now(timezone.utc),
    )


def _make_retrieval_result() -> RetrievalResult:
    """Create a valid RetrievalResult for mock retrieval responses."""
    return RetrievalResult(
        verified_chunks=[_make_scored_chunk()],
        tamper_alerts=[],
        no_evidence_found=False,
        degraded_search=False,
        total_candidates=1,
    )


def _create_fresh_api(namespace: str) -> tuple[AxisweaveServiceAPI, APICredentials]:
    """Create a fresh AxisweaveServiceAPI instance with proper auth for the given namespace.

    Each call creates a completely new instance with no prior operation history,
    demonstrating that operations are independent.
    """
    registry = NamespaceRegistry()
    registry.register_namespace(namespace, TEST_TENANT_ID)

    auth_provider = APIAuthProvider(namespace_registry=registry)
    auth_provider.register_api_key(
        api_key=TEST_API_KEY,
        tenant_id=TEST_TENANT_ID,
        authorized_namespaces=[namespace],
    )

    credentials = APICredentials(
        api_key=TEST_API_KEY,
        tenant_id=TEST_TENANT_ID,
        authorized_namespaces=[namespace],
    )

    # Mock service dependencies
    mock_ingestion = AsyncMock()
    mock_ingestion.ingest_document = AsyncMock(return_value=_make_ingestion_result())

    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_result())

    mock_kms = AsyncMock()
    mock_kms.verify_signature = AsyncMock(return_value=True)

    api = AxisweaveServiceAPI(
        ingestion_service=mock_ingestion,
        retrieval_service=mock_retrieval,
        auth_provider=auth_provider,
        namespace_registry=registry,
        kms_client=mock_kms,
    )

    return api, credentials


# =============================================================================
# Property 24: API Operation Independence
# =============================================================================


@pytest.mark.property
class TestAPIOperationIndependence:
    """Property 24: API Operation Independence.

    **Validates: Requirements 13.1**

    For any valid ingest, retrieve, or verify request with proper auth and
    valid namespace, the operation succeeds without requiring prior invocation
    of any other operation.
    """

    @given(
        namespace=valid_namespace_strategy,
        category=valid_category_strategy,
        doc_bytes=valid_document_bytes_strategy,
    )
    @settings(max_examples=100)
    def test_ingest_succeeds_without_prior_operations(
        self, namespace: str, category: str, doc_bytes: bytes
    ):
        """Ingest operation succeeds on a fresh API instance with no prior calls.

        **Validates: Requirements 13.1**

        A valid ingest request with proper auth and valid namespace should
        succeed without requiring prior retrieve or verify calls.
        """
        api, credentials = _create_fresh_api(namespace)

        request = IngestRequest(
            namespace=namespace,
            document_category=category,
            document_bytes=doc_bytes,
        )

        # Execute ingest in isolation — no prior retrieve or verify
        response = asyncio.get_event_loop().run_until_complete(
            api.ingest(request, credentials)
        )

        # Verify successful response
        assert isinstance(response, IngestResponse)
        assert response.document_id is not None
        assert response.content_hash is not None
        assert response.signature is not None
        assert response.chunk_count >= 1
        assert response.api_version == AxisweaveServiceAPI.API_VERSION

    @given(
        namespace=valid_namespace_strategy,
        query=valid_query_strategy,
        top_k=valid_top_k_strategy,
        min_score=valid_min_score_strategy,
    )
    @settings(max_examples=100)
    def test_retrieve_succeeds_without_prior_operations(
        self, namespace: str, query: str, top_k: int, min_score: float
    ):
        """Retrieve operation succeeds on a fresh API instance with no prior calls.

        **Validates: Requirements 13.1**

        A valid retrieve request with proper auth and valid namespace should
        succeed without requiring prior ingest or verify calls.
        """
        api, credentials = _create_fresh_api(namespace)

        request = RetrieveRequest(
            namespace=namespace,
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        # Execute retrieve in isolation — no prior ingest or verify
        response = asyncio.get_event_loop().run_until_complete(
            api.retrieve(request, credentials)
        )

        # Verify successful response
        assert isinstance(response, RetrieveResponse)
        assert response.chunks is not None
        assert isinstance(response.chunks, list)
        assert response.api_version == AxisweaveServiceAPI.API_VERSION

    @given(
        namespace=valid_namespace_strategy,
        document_id=valid_document_id_strategy,
    )
    @settings(max_examples=100)
    def test_verify_succeeds_without_prior_operations(
        self, namespace: str, document_id: str
    ):
        """Verify operation succeeds on a fresh API instance with no prior calls.

        **Validates: Requirements 13.1**

        A valid verify request with proper auth and valid namespace should
        succeed without requiring prior ingest or retrieve calls.
        """
        api, credentials = _create_fresh_api(namespace)

        request = VerifyRequest(
            namespace=namespace,
            document_id=document_id,
        )

        # Execute verify in isolation — no prior ingest or retrieve
        response = asyncio.get_event_loop().run_until_complete(
            api.verify(request, credentials)
        )

        # Verify successful response
        assert isinstance(response, VerifyResponse)
        assert response.document_id == document_id
        assert response.chunk_results is not None
        assert isinstance(response.chunk_results, list)
        assert response.api_version == AxisweaveServiceAPI.API_VERSION

    @given(
        namespace=valid_namespace_strategy,
        category=valid_category_strategy,
        doc_bytes=valid_document_bytes_strategy,
        query=valid_query_strategy,
        document_id=valid_document_id_strategy,
    )
    @settings(max_examples=50)
    def test_all_operations_succeed_independently_same_namespace(
        self, namespace: str, category: str, doc_bytes: bytes,
        query: str, document_id: str
    ):
        """All three operations succeed independently on separate fresh API instances.

        **Validates: Requirements 13.1**

        Each operation gets its own fresh API instance, proving no shared state
        or ordering dependency between ingest, retrieve, and verify.
        """
        # Create three completely independent API instances
        api_ingest, creds_ingest = _create_fresh_api(namespace)
        api_retrieve, creds_retrieve = _create_fresh_api(namespace)
        api_verify, creds_verify = _create_fresh_api(namespace)

        loop = asyncio.get_event_loop()

        # Ingest on its own instance (no prior ops)
        ingest_request = IngestRequest(
            namespace=namespace,
            document_category=category,
            document_bytes=doc_bytes,
        )
        ingest_response = loop.run_until_complete(
            api_ingest.ingest(ingest_request, creds_ingest)
        )
        assert isinstance(ingest_response, IngestResponse)

        # Retrieve on its own instance (no prior ops)
        retrieve_request = RetrieveRequest(
            namespace=namespace,
            query=query,
        )
        retrieve_response = loop.run_until_complete(
            api_retrieve.retrieve(retrieve_request, creds_retrieve)
        )
        assert isinstance(retrieve_response, RetrieveResponse)

        # Verify on its own instance (no prior ops)
        verify_request = VerifyRequest(
            namespace=namespace,
            document_id=document_id,
        )
        verify_response = loop.run_until_complete(
            api_verify.verify(verify_request, creds_verify)
        )
        assert isinstance(verify_response, VerifyResponse)

    @given(
        namespace=valid_namespace_strategy,
        document_id=valid_document_id_strategy,
        chunk_ids=st.lists(
            st.uuids().map(str),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=50)
    def test_verify_with_chunk_ids_succeeds_without_prior_ingest(
        self, namespace: str, document_id: str, chunk_ids: list[str]
    ):
        """Verify with specific chunk_ids succeeds without any prior ingestion.

        **Validates: Requirements 13.1**

        Provenance verification is callable independently — any authenticated
        caller with namespace access can verify, without having ingested the
        document themselves.
        """
        api, credentials = _create_fresh_api(namespace)

        request = VerifyRequest(
            namespace=namespace,
            document_id=document_id,
            chunk_ids=chunk_ids,
        )

        # Execute verify with chunk_ids — no prior ingest
        response = asyncio.get_event_loop().run_until_complete(
            api.verify(request, credentials)
        )

        assert isinstance(response, VerifyResponse)
        assert response.document_id == document_id
        # Should have a result for each requested chunk_id
        assert len(response.chunk_results) == len(chunk_ids)

    @given(
        namespace=valid_namespace_strategy,
        query=valid_query_strategy,
    )
    @settings(max_examples=50)
    def test_retrieve_returns_valid_provenance_without_prior_ingest(
        self, namespace: str, query: str
    ):
        """Retrieve returns full provenance metadata without prior ingest by caller.

        **Validates: Requirements 13.1**

        The retrieve operation returns results with full provenance metadata
        (document_id, content_hash, KMS_signature, chunk_index, ingestion_timestamp)
        regardless of whether the calling client previously ingested documents.
        """
        api, credentials = _create_fresh_api(namespace)

        request = RetrieveRequest(
            namespace=namespace,
            query=query,
        )

        # Execute retrieve — no prior ingest by this caller
        response = asyncio.get_event_loop().run_until_complete(
            api.retrieve(request, credentials)
        )

        assert isinstance(response, RetrieveResponse)
        # If chunks are returned, they should carry full provenance
        for chunk in response.chunks:
            assert chunk.provenance is not None
            assert chunk.provenance.document_id is not None
            assert chunk.provenance.content_hash is not None
            assert chunk.provenance.kms_signature is not None
            assert chunk.provenance.chunk_index is not None
            assert chunk.provenance.ingestion_timestamp is not None
