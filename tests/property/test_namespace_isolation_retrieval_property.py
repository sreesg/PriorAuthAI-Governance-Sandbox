"""Property-based tests for Namespace Isolation in Retrieval.

**Validates: Requirements 13.3, 13.5, 13.7**

Property 23: Namespace Isolation in Retrieval
- For any retrieval query specifying namespace A, all returned chunks have
  namespace metadata equal to A; no chunks from namespace B appear unless
  explicit cross-namespace authorization exists.
- Chunks with None namespace pass through (already scoped at query level).
- Unauthorized namespace chunks are always excluded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.api.auth_provider import APIAuthProvider, APICredentials
from clinical_reasoning_fabric.api.axisweave_service_api import AxisweaveServiceAPI
from clinical_reasoning_fabric.api.namespace import NamespaceRegistry
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    ScoredChunk,
)


# =============================================================================
# Constants
# =============================================================================

VALID_NAMESPACE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"


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
) -> ScoredChunk:
    """Create a ScoredChunk with the specified namespace."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"Clinical content of chunk {chunk_id}",
        score=score,
        provenance=_make_provenance(f"doc-{chunk_id}"),
        namespace=namespace,
    )


def _build_api_instance() -> AxisweaveServiceAPI:
    """Build an AxisweaveServiceAPI instance with mocked dependencies for testing."""
    ingestion_service = MagicMock()
    retrieval_service = MagicMock()
    auth_provider = MagicMock(spec=APIAuthProvider)
    namespace_registry = NamespaceRegistry()
    namespace_registry.register_namespace("test-ns", "tenant-test")

    return AxisweaveServiceAPI(
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
        auth_provider=auth_provider,
        namespace_registry=namespace_registry,
    )


# =============================================================================
# Hypothesis Strategies
# =============================================================================


def namespace_strategy() -> st.SearchStrategy[str]:
    """Generate valid namespace identifiers (1-32 chars for practical testing)."""
    return st.text(
        alphabet=VALID_NAMESPACE_ALPHABET,
        min_size=1,
        max_size=32,
    )


def distinct_namespaces_strategy(min_size: int = 2, max_size: int = 5):
    """Generate a list of distinct namespace identifiers."""
    return st.lists(
        namespace_strategy(),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


@st.composite
def chunks_with_namespaces_strategy(draw, namespaces: list[str] | None = None):
    """Generate a list of ScoredChunks with various namespace assignments.

    Each chunk gets a namespace drawn from the provided list or None.
    """
    if namespaces is None:
        namespaces = draw(distinct_namespaces_strategy(min_size=2, max_size=5))

    num_chunks = draw(st.integers(min_value=1, max_value=20))
    # Include None as a possible namespace assignment
    namespace_options = namespaces + [None]

    chunks = []
    for i in range(num_chunks):
        ns = draw(st.sampled_from(namespace_options))
        score = draw(st.floats(min_value=0.01, max_value=1.0))
        chunk = _make_chunk(chunk_id=f"chunk-{i}", namespace=ns, score=score)
        chunks.append(chunk)

    return chunks, namespaces


@st.composite
def authorized_subset_strategy(draw, namespaces: list[str]):
    """Generate a non-empty subset of namespaces as 'authorized'."""
    assume(len(namespaces) > 0)
    subset = draw(
        st.lists(
            st.sampled_from(namespaces),
            min_size=1,
            max_size=len(namespaces),
            unique=True,
        )
    )
    return subset


# =============================================================================
# Property Tests
# =============================================================================


@pytest.mark.property
class TestNamespaceIsolationInRetrieval:
    """Property 23: Namespace Isolation in Retrieval.

    **Validates: Requirements 13.3, 13.5, 13.7**

    Tests that:
    - For any retrieval returning chunks with various namespaces, filtering by
      authorized namespace A returns only chunks with namespace == A or None.
    - Chunks from unauthorized namespaces never appear in filtered results.
    - Cross-namespace access works when explicitly authorized.
    - None namespace chunks always pass through the filter.
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_only_authorized_namespace_chunks_returned(self, data):
        """Chunks with authorized namespaces pass; unauthorized are excluded."""
        api = _build_api_instance()

        # Generate distinct namespaces
        all_namespaces = data.draw(distinct_namespaces_strategy(min_size=2, max_size=5))
        assume(len(all_namespaces) >= 2)

        # Split into authorized and unauthorized
        split_point = data.draw(st.integers(min_value=1, max_value=len(all_namespaces) - 1))
        authorized = all_namespaces[:split_point]
        unauthorized = all_namespaces[split_point:]

        # Generate chunks from all namespaces (no None)
        num_chunks = data.draw(st.integers(min_value=1, max_value=15))
        chunks = []
        for i in range(num_chunks):
            ns = data.draw(st.sampled_from(all_namespaces))
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"chunk-{i}", namespace=ns, score=score))

        # Apply namespace filter
        result = api._filter_by_namespace(chunks, authorized)

        # Property: every returned chunk has an authorized namespace
        for chunk in result:
            assert chunk.namespace in authorized, (
                f"Chunk {chunk.chunk_id} with namespace '{chunk.namespace}' "
                f"returned but only {authorized} are authorized"
            )

        # Property: no chunk from unauthorized namespace appears
        unauthorized_in_result = [c for c in result if c.namespace in unauthorized]
        assert len(unauthorized_in_result) == 0, (
            f"Found {len(unauthorized_in_result)} chunks from unauthorized "
            f"namespaces {unauthorized} in result"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_none_namespace_chunks_pass_through(self, data):
        """Chunks with namespace=None always pass through the filter."""
        api = _build_api_instance()

        authorized = data.draw(
            st.lists(namespace_strategy(), min_size=1, max_size=3, unique=True)
        )

        # Generate a mix of chunks: some with authorized ns, some unauthorized, some None
        num_none_chunks = data.draw(st.integers(min_value=1, max_value=5))
        num_ns_chunks = data.draw(st.integers(min_value=0, max_value=5))

        chunks = []
        # Add chunks with namespace=None
        for i in range(num_none_chunks):
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"none-{i}", namespace=None, score=score))

        # Add chunks with some namespace
        for i in range(num_ns_chunks):
            ns = data.draw(namespace_strategy())
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"ns-{i}", namespace=ns, score=score))

        result = api._filter_by_namespace(chunks, authorized)

        # Property: all None-namespace chunks must appear in result
        none_chunks_in_result = [c for c in result if c.namespace is None]
        assert len(none_chunks_in_result) == num_none_chunks, (
            f"Expected {num_none_chunks} None-namespace chunks in result, "
            f"got {len(none_chunks_in_result)}"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_cross_namespace_access_with_authorization(self, data):
        """Cross-namespace access works when multiple namespaces are authorized."""
        api = _build_api_instance()

        # Generate namespaces: all of them are authorized (simulating cross-ns grant)
        all_namespaces = data.draw(distinct_namespaces_strategy(min_size=2, max_size=4))
        authorized = all_namespaces  # All namespaces explicitly authorized

        # Generate chunks spanning all authorized namespaces
        num_chunks = data.draw(st.integers(min_value=2, max_value=10))
        chunks = []
        for i in range(num_chunks):
            ns = data.draw(st.sampled_from(all_namespaces))
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"cross-{i}", namespace=ns, score=score))

        result = api._filter_by_namespace(chunks, authorized)

        # Property: all chunks pass through since all namespaces are authorized
        assert len(result) == len(chunks), (
            f"Expected all {len(chunks)} chunks to pass with full cross-namespace "
            f"authorization, but only {len(result)} returned"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_no_chunks_from_namespace_b_without_authorization(self, data):
        """Chunks from namespace B never appear when only namespace A is authorized."""
        api = _build_api_instance()

        # Generate exactly two distinct namespaces
        ns_a = data.draw(namespace_strategy())
        ns_b = data.draw(namespace_strategy())
        assume(ns_a != ns_b)

        authorized = [ns_a]  # Only namespace A authorized

        # Generate chunks: some from A, some from B
        num_a_chunks = data.draw(st.integers(min_value=1, max_value=8))
        num_b_chunks = data.draw(st.integers(min_value=1, max_value=8))

        chunks = []
        for i in range(num_a_chunks):
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"a-{i}", namespace=ns_a, score=score))
        for i in range(num_b_chunks):
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"b-{i}", namespace=ns_b, score=score))

        result = api._filter_by_namespace(chunks, authorized)

        # Property: only namespace A chunks returned
        assert len(result) == num_a_chunks, (
            f"Expected exactly {num_a_chunks} chunks from namespace '{ns_a}', "
            f"got {len(result)}"
        )
        for chunk in result:
            assert chunk.namespace == ns_a, (
                f"Chunk {chunk.chunk_id} has namespace '{chunk.namespace}' "
                f"but only '{ns_a}' is authorized"
            )

        # Property: zero namespace B chunks
        b_in_result = [c for c in result if c.namespace == ns_b]
        assert len(b_in_result) == 0, (
            f"Found {len(b_in_result)} chunks from unauthorized namespace '{ns_b}'"
        )

    @given(data=st.data())
    @settings(max_examples=50)
    def test_empty_authorized_namespaces_only_returns_none(self, data):
        """With empty authorized list, only None-namespace chunks pass."""
        api = _build_api_instance()

        # Generate chunks: mix of namespaced and None
        num_none = data.draw(st.integers(min_value=0, max_value=5))
        num_namespaced = data.draw(st.integers(min_value=0, max_value=5))
        assume(num_none + num_namespaced > 0)

        chunks = []
        for i in range(num_none):
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"none-{i}", namespace=None, score=score))
        for i in range(num_namespaced):
            ns = data.draw(namespace_strategy())
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"ns-{i}", namespace=ns, score=score))

        result = api._filter_by_namespace(chunks, [])

        # Property: only None-namespace chunks returned
        assert len(result) == num_none, (
            f"Expected {num_none} None-namespace chunks with empty authorized list, "
            f"got {len(result)}"
        )
        for chunk in result:
            assert chunk.namespace is None, (
                f"Chunk {chunk.chunk_id} with namespace '{chunk.namespace}' "
                f"returned despite empty authorized namespaces list"
            )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_filter_preserves_chunk_ordering(self, data):
        """Filtering preserves the original order of chunks."""
        api = _build_api_instance()

        authorized = data.draw(
            st.lists(namespace_strategy(), min_size=1, max_size=3, unique=True)
        )
        all_ns = authorized + data.draw(
            st.lists(namespace_strategy(), min_size=1, max_size=3, unique=True)
        )

        # Generate ordered chunks
        num_chunks = data.draw(st.integers(min_value=2, max_value=15))
        chunks = []
        for i in range(num_chunks):
            ns = data.draw(st.sampled_from(all_ns + [None]))
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"ord-{i}", namespace=ns, score=score))

        result = api._filter_by_namespace(chunks, authorized)

        # Property: relative order is preserved
        result_ids = [c.chunk_id for c in result]
        original_order = [c.chunk_id for c in chunks if c.chunk_id in result_ids]
        assert result_ids == original_order, (
            f"Filter changed chunk ordering. Expected {original_order}, got {result_ids}"
        )

    @given(data=st.data())
    @settings(max_examples=50)
    def test_filter_is_complete_no_authorized_chunks_lost(self, data):
        """No chunks from authorized namespaces are lost during filtering."""
        api = _build_api_instance()

        authorized = data.draw(
            st.lists(namespace_strategy(), min_size=1, max_size=3, unique=True)
        )

        # Generate chunks only from authorized namespaces and None
        num_chunks = data.draw(st.integers(min_value=1, max_value=10))
        chunks = []
        for i in range(num_chunks):
            ns = data.draw(st.sampled_from(authorized + [None]))
            score = data.draw(st.floats(min_value=0.01, max_value=1.0))
            chunks.append(_make_chunk(chunk_id=f"keep-{i}", namespace=ns, score=score))

        result = api._filter_by_namespace(chunks, authorized)

        # Property: all chunks must be returned (no false exclusions)
        assert len(result) == len(chunks), (
            f"Expected all {len(chunks)} chunks to pass (all authorized or None), "
            f"but only {len(result)} returned"
        )
