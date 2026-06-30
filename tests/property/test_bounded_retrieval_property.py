"""Property-based tests for Bounded Retrieval with Lineage Tracking.

**Validates: Requirements 11.4, 11.5, 11.7**

Property 17: Bounded Retrieval with Lineage Tracking
- Total retrieval calls (call_count) never exceed configured max_retrieval_calls
- After max is reached, subsequent calls return empty
- All valid retrieved snippets appear in get_retrieved_snippets()
- get_lineage_entries() contains one LineageEntry per valid retrieved snippet
- Graph updates via record_graph_update() pass execution_id to the graph service
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.bounded_retrieval import (
    BoundedRetrievalService,
    DEFAULT_MAX_RETRIEVAL_CALLS,
    MIN_RETRIEVAL_CALLS,
    MAX_RETRIEVAL_CALLS,
)
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    LineageEntry,
    ScoredChunk,
    TamperAlert,
    TraceCategory,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

non_empty_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=50,
)

valid_datetime = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)

sha256_hex = st.text(
    alphabet="0123456789abcdef",
    min_size=64,
    max_size=64,
)

# Strategy for max_retrieval_calls: valid range 1-50
max_calls_strategy = st.integers(min_value=MIN_RETRIEVAL_CALLS, max_value=MAX_RETRIEVAL_CALLS)

# Strategy for number of retrieval attempts (1-60, some exceeding max)
num_attempts_strategy = st.integers(min_value=1, max_value=60)


@st.composite
def kms_signature_strategy(draw):
    """Generate a valid KMSSignature."""
    return KMSSignature(
        key_id=draw(non_empty_text),
        signature=draw(non_empty_text),
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=draw(valid_datetime),
    )


@st.composite
def chunk_provenance_strategy(draw):
    """Generate valid ChunkProvenance."""
    return ChunkProvenance(
        document_id=draw(non_empty_text),
        content_hash=draw(sha256_hex),
        kms_signature=draw(kms_signature_strategy()),
        chunk_index=draw(st.integers(min_value=0, max_value=100)),
        ingestion_timestamp=draw(valid_datetime),
    )


@st.composite
def scored_chunk_strategy(draw):
    """Generate a valid ScoredChunk."""
    return ScoredChunk(
        chunk_id=draw(non_empty_text),
        text=draw(non_empty_text),
        score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        provenance=draw(chunk_provenance_strategy()),
    )


@st.composite
def chunks_per_call_strategy(draw):
    """Generate a list of 0-5 ScoredChunks for a single retrieval call."""
    num_chunks = draw(st.integers(min_value=0, max_value=5))
    return [draw(scored_chunk_strategy()) for _ in range(num_chunks)]


# =============================================================================
# Mock Factories
# =============================================================================


def create_mock_mcp_gateway(chunks_per_call: list[list[ScoredChunk]]):
    """Create a mock MCP Gateway that returns predefined chunks for each call.

    Args:
        chunks_per_call: A list where each element is the list of ScoredChunks
                         that will be returned by the gateway for that call number.
    """
    call_index = {"current": 0}

    async def mock_invoke_tool(tool_name: str, parameters: dict, agent_identity: str):
        idx = call_index["current"]
        call_index["current"] += 1
        if idx < len(chunks_per_call):
            return chunks_per_call[idx]
        return []

    gateway = AsyncMock()
    gateway.invoke_tool = AsyncMock(side_effect=mock_invoke_tool)
    return gateway


def create_mock_signature_verifier(all_valid: bool = True):
    """Create a mock signature verifier.

    Args:
        all_valid: If True, all chunks pass verification. If False, all are rejected.
    """

    async def mock_verify(chunks: list[ScoredChunk]):
        if all_valid:
            return (chunks, [])
        else:
            alerts = [
                TamperAlert(
                    chunk_id=c.chunk_id,
                    document_id=c.provenance.document_id,
                    content_hash=c.provenance.content_hash,
                    reason="Invalid KMS signature",
                    detected_at=datetime.now(timezone.utc),
                )
                for c in chunks
            ]
            return ([], alerts)

    verifier = AsyncMock()
    verifier.verify_signatures = AsyncMock(side_effect=mock_verify)
    return verifier


def create_mock_graph_service():
    """Create a mock graph service that tracks calls with execution_id."""
    graph_service = AsyncMock()
    graph_service.upsert_node = AsyncMock()
    return graph_service


def create_mock_audit_trail():
    """Create a mock audit trail service."""
    audit_trail = AsyncMock()
    audit_trail.record_entry = AsyncMock()
    return audit_trail


# =============================================================================
# Property 17: Bounded Retrieval with Lineage Tracking
# =============================================================================


@pytest.mark.property
class TestBoundedRetrievalWithLineageTracking:
    """Property 17: Bounded Retrieval with Lineage Tracking.

    **Validates: Requirements 11.4, 11.5, 11.7**

    - Total successful retrieval calls never exceed max_retrieval_calls
    - After max is reached, subsequent calls return empty
    - All valid retrieved snippets appear in get_retrieved_snippets()
    - get_lineage_entries() contains one LineageEntry per valid retrieved snippet
    - Graph updates via record_graph_update() pass execution_id
    """

    @given(
        max_calls=max_calls_strategy,
        num_attempts=num_attempts_strategy,
        chunks_lists=st.lists(
            chunks_per_call_strategy(),
            min_size=1,
            max_size=60,
        ),
    )
    @settings(max_examples=100, deadline=10000)
    def test_call_count_never_exceeds_max(self, max_calls, num_attempts, chunks_lists):
        """Total retrieval calls (call_count) never exceed max_retrieval_calls.

        **Validates: Requirements 11.5**

        The CRF SHALL limit the total number of retrieval calls per PA request
        to a configurable maximum with a default of 10 calls, where the
        configurable range is 1 to 50.
        """
        # Ensure we have enough chunk lists for the attempts
        while len(chunks_lists) < num_attempts:
            chunks_lists.append([])

        gateway = create_mock_mcp_gateway(chunks_lists)
        verifier = create_mock_signature_verifier(all_valid=True)
        graph_service = create_mock_graph_service()
        audit_trail = create_mock_audit_trail()

        service = BoundedRetrievalService(
            mcp_gateway=gateway,
            signature_verifier=verifier,
            graph_service=graph_service,
            audit_trail=audit_trail,
            execution_id="exec-test-001",
            request_id="req-test-001",
            agent_identity="agent-test",
            max_retrieval_calls=max_calls,
        )

        async def run_attempts():
            for i in range(num_attempts):
                await service.retrieve_additional(f"query-{i}")

        asyncio.get_event_loop().run_until_complete(run_attempts())

        # Property: call_count never exceeds max_retrieval_calls
        assert service.call_count <= max_calls, (
            f"call_count ({service.call_count}) exceeds max ({max_calls}) "
            f"after {num_attempts} attempts"
        )

    @given(
        max_calls=max_calls_strategy,
        chunks_lists=st.lists(
            chunks_per_call_strategy(),
            min_size=1,
            max_size=60,
        ),
    )
    @settings(max_examples=100, deadline=10000)
    def test_returns_empty_after_limit_reached(self, max_calls, chunks_lists):
        """After max is reached, subsequent calls return empty list.

        **Validates: Requirements 11.5**

        IF the retrieval call limit is reached, THEN THE CRF SHALL proceed
        to decision with available evidence.
        """
        # Ensure we have enough chunks and make at least max_calls + 1 attempts
        num_attempts = max_calls + 5
        while len(chunks_lists) < num_attempts:
            chunks_lists.append([])

        gateway = create_mock_mcp_gateway(chunks_lists)
        verifier = create_mock_signature_verifier(all_valid=True)
        graph_service = create_mock_graph_service()
        audit_trail = create_mock_audit_trail()

        service = BoundedRetrievalService(
            mcp_gateway=gateway,
            signature_verifier=verifier,
            graph_service=graph_service,
            audit_trail=audit_trail,
            execution_id="exec-test-002",
            request_id="req-test-002",
            agent_identity="agent-test",
            max_retrieval_calls=max_calls,
        )

        results_after_limit = []

        async def run_attempts():
            # First, exhaust the limit
            for i in range(max_calls):
                await service.retrieve_additional(f"query-{i}")
            # Now make additional attempts beyond the limit
            for i in range(5):
                result = await service.retrieve_additional(f"extra-query-{i}")
                results_after_limit.append(result)

        asyncio.get_event_loop().run_until_complete(run_attempts())

        # Property: all calls after limit return empty
        for i, result in enumerate(results_after_limit):
            assert result == [], (
                f"Call {max_calls + i + 1} after limit should return empty, "
                f"got {len(result)} chunks"
            )

    @given(
        max_calls=max_calls_strategy,
        num_attempts=st.integers(min_value=1, max_value=30),
        chunks_lists=st.lists(
            chunks_per_call_strategy(),
            min_size=1,
            max_size=30,
        ),
    )
    @settings(max_examples=100, deadline=10000)
    def test_all_valid_snippets_in_retrieved_snippets(
        self, max_calls, num_attempts, chunks_lists
    ):
        """All valid retrieved snippets appear in get_retrieved_snippets().

        **Validates: Requirements 11.7**

        WHEN the agent completes reasoning that included additional retrieval
        calls, THE CRF SHALL include all additionally retrieved evidence
        snippets with their provenance metadata in the Evidence_Bundle
        lineage_trail.
        """
        # Ensure we have enough chunk lists
        while len(chunks_lists) < num_attempts:
            chunks_lists.append([])

        gateway = create_mock_mcp_gateway(chunks_lists)
        verifier = create_mock_signature_verifier(all_valid=True)
        graph_service = create_mock_graph_service()
        audit_trail = create_mock_audit_trail()

        service = BoundedRetrievalService(
            mcp_gateway=gateway,
            signature_verifier=verifier,
            graph_service=graph_service,
            audit_trail=audit_trail,
            execution_id="exec-test-003",
            request_id="req-test-003",
            agent_identity="agent-test",
            max_retrieval_calls=max_calls,
        )

        all_returned_chunks: list[ScoredChunk] = []

        async def run_attempts():
            for i in range(num_attempts):
                result = await service.retrieve_additional(f"query-{i}")
                all_returned_chunks.extend(result)

        asyncio.get_event_loop().run_until_complete(run_attempts())

        # Property: get_retrieved_snippets() contains exactly all returned chunks
        retrieved = service.get_retrieved_snippets()
        assert len(retrieved) == len(all_returned_chunks), (
            f"get_retrieved_snippets() has {len(retrieved)} items but "
            f"we collected {len(all_returned_chunks)} valid returned chunks"
        )

        # Verify each returned chunk is present
        retrieved_ids = {c.chunk_id for c in retrieved}
        for chunk in all_returned_chunks:
            assert chunk.chunk_id in retrieved_ids, (
                f"Chunk {chunk.chunk_id} was returned but not in get_retrieved_snippets()"
            )

    @given(
        max_calls=max_calls_strategy,
        num_attempts=st.integers(min_value=1, max_value=30),
        chunks_lists=st.lists(
            chunks_per_call_strategy(),
            min_size=1,
            max_size=30,
        ),
    )
    @settings(max_examples=100, deadline=10000)
    def test_lineage_entries_match_retrieved_snippets(
        self, max_calls, num_attempts, chunks_lists
    ):
        """get_lineage_entries() contains one LineageEntry per valid retrieved snippet.

        **Validates: Requirements 11.7**

        Each lineage entry should have a non-empty conclusion, the evidence_id
        matching the chunk_id, and a valid retrieval_timestamp from provenance.
        """
        # Ensure we have enough chunk lists
        while len(chunks_lists) < num_attempts:
            chunks_lists.append([])

        gateway = create_mock_mcp_gateway(chunks_lists)
        verifier = create_mock_signature_verifier(all_valid=True)
        graph_service = create_mock_graph_service()
        audit_trail = create_mock_audit_trail()

        service = BoundedRetrievalService(
            mcp_gateway=gateway,
            signature_verifier=verifier,
            graph_service=graph_service,
            audit_trail=audit_trail,
            execution_id="exec-test-004",
            request_id="req-test-004",
            agent_identity="agent-test",
            max_retrieval_calls=max_calls,
        )

        async def run_attempts():
            for i in range(num_attempts):
                await service.retrieve_additional(f"query-{i}")

        asyncio.get_event_loop().run_until_complete(run_attempts())

        retrieved = service.get_retrieved_snippets()
        lineage = service.get_lineage_entries()

        # Property: one LineageEntry per retrieved snippet
        assert len(lineage) == len(retrieved), (
            f"lineage has {len(lineage)} entries but retrieved has {len(retrieved)} snippets"
        )

        # Each lineage entry references the correct chunk
        for i, (entry, snippet) in enumerate(zip(lineage, retrieved)):
            assert isinstance(entry, LineageEntry), (
                f"lineage[{i}] must be a LineageEntry"
            )
            assert entry.evidence_id == snippet.chunk_id, (
                f"lineage[{i}].evidence_id ({entry.evidence_id}) "
                f"!= snippet.chunk_id ({snippet.chunk_id})"
            )
            assert entry.conclusion is not None and len(entry.conclusion) > 0, (
                f"lineage[{i}].conclusion must be non-empty"
            )
            assert entry.retrieval_timestamp is not None, (
                f"lineage[{i}].retrieval_timestamp must not be None"
            )
            assert isinstance(entry.retrieval_timestamp, datetime), (
                f"lineage[{i}].retrieval_timestamp must be a datetime"
            )

    @given(
        max_calls=max_calls_strategy,
        num_updates=st.integers(min_value=1, max_value=20),
        node_types=st.lists(
            st.sampled_from(["Member", "Event", "SDOH_Factor", "EvidenceSource"]),
            min_size=1,
            max_size=20,
        ),
        node_ids=st.lists(non_empty_text, min_size=1, max_size=20),
    )
    @settings(max_examples=100, deadline=10000)
    def test_graph_updates_pass_execution_id(
        self, max_calls, num_updates, node_types, node_ids
    ):
        """Graph updates via record_graph_update() pass execution_id to graph service.

        **Validates: Requirements 11.4**

        WHEN the agent updates the clinical graph during reasoning, THE
        Causal_Ontology_Graph SHALL record the update with the agent
        execution_id as provenance.
        """
        gateway = create_mock_mcp_gateway([])
        verifier = create_mock_signature_verifier(all_valid=True)
        graph_service = create_mock_graph_service()
        audit_trail = create_mock_audit_trail()

        execution_id = "exec-provenance-test"
        service = BoundedRetrievalService(
            mcp_gateway=gateway,
            signature_verifier=verifier,
            graph_service=graph_service,
            audit_trail=audit_trail,
            execution_id=execution_id,
            request_id="req-test-005",
            agent_identity="agent-test",
            max_retrieval_calls=max_calls,
        )

        # Align list lengths
        actual_updates = min(num_updates, len(node_types), len(node_ids))

        async def run_updates():
            for i in range(actual_updates):
                await service.record_graph_update(
                    node_type=node_types[i],
                    node_id=node_ids[i],
                    properties={"updated_by": "test", "index": i},
                )

        asyncio.get_event_loop().run_until_complete(run_updates())

        # Property: every graph update was called with execution_id
        assert graph_service.upsert_node.call_count == actual_updates, (
            f"Expected {actual_updates} graph updates, got {graph_service.upsert_node.call_count}"
        )

        for call in graph_service.upsert_node.call_args_list:
            # Check that execution_id was passed
            _, kwargs = call
            assert kwargs.get("execution_id") == execution_id, (
                f"Graph update missing execution_id. kwargs: {kwargs}"
            )
