"""Unit tests for BoundedRetrievalService.

Tests retrieval call limit enforcement, KMS signature verification,
tamper alert generation, graph provenance recording, and lineage tracking.

Requirements referenced: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical_reasoning_fabric.beacon.bounded_retrieval import (
    APPROVED_RETRIEVAL_TOOL,
    DEFAULT_MAX_RETRIEVAL_CALLS,
    MAX_RETRIEVAL_CALLS,
    MIN_RETRIEVAL_CALLS,
    BoundedRetrievalService,
    RetrievalCallRecord,
)
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    LineageEntry,
    ScoredChunk,
    TamperAlert,
    TraceCategory,
    ToolResult,
)


# =============================================================================
# Test Helpers
# =============================================================================


def _make_kms_signature(key_id: str = "key-1") -> KMSSignature:
    """Create a valid KMSSignature for testing."""
    return KMSSignature(
        key_id=key_id,
        signature="dGVzdHNpZ25hdHVyZQ==",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_provenance(doc_id: str = "doc-001", chunk_idx: int = 0) -> ChunkProvenance:
    """Create a valid ChunkProvenance for testing."""
    return ChunkProvenance(
        document_id=doc_id,
        content_hash="a" * 64,
        kms_signature=_make_kms_signature(),
        chunk_index=chunk_idx,
        ingestion_timestamp=datetime.now(timezone.utc),
    )


def _make_scored_chunk(
    chunk_id: str = "chunk-001",
    text: str = "Test clinical note content",
    score: float = 0.8,
    doc_id: str = "doc-001",
    chunk_idx: int = 0,
) -> ScoredChunk:
    """Create a valid ScoredChunk for testing."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        provenance=_make_provenance(doc_id, chunk_idx),
    )


def _make_tamper_alert(chunk_id: str = "chunk-bad") -> TamperAlert:
    """Create a TamperAlert for testing."""
    return TamperAlert(
        chunk_id=chunk_id,
        document_id="doc-bad",
        content_hash="b" * 64,
        expected_signature="invalid_sig",
        reason="KMS signature verification failed",
        detected_at=datetime.now(timezone.utc),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp_gateway():
    """Mock MCP Gateway that returns chunks via invoke_tool."""
    gateway = AsyncMock()
    # Default: return a ToolResult with chunks in output_result
    chunks = [_make_scored_chunk(f"chunk-{i}", f"Clinical text {i}") for i in range(3)]
    gateway.invoke_tool.return_value = MagicMock(
        output_result=chunks,
        success=True,
    )
    return gateway


@pytest.fixture
def mock_signature_verifier():
    """Mock signature verifier that passes all chunks by default."""
    verifier = AsyncMock()
    # Default: all signatures are valid, no tamper alerts
    async def verify_all_valid(chunks):
        return (chunks, [])

    verifier.verify_signatures.side_effect = verify_all_valid
    return verifier


@pytest.fixture
def mock_graph_service():
    """Mock graph service for node upserts."""
    service = AsyncMock()
    return service


@pytest.fixture
def mock_audit_trail():
    """Mock audit trail service for recording entries."""
    trail = AsyncMock()
    trail.record_entry.return_value = MagicMock()
    return trail


@pytest.fixture
def bounded_service(mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail):
    """A configured BoundedRetrievalService for testing."""
    return BoundedRetrievalService(
        mcp_gateway=mock_mcp_gateway,
        signature_verifier=mock_signature_verifier,
        graph_service=mock_graph_service,
        audit_trail=mock_audit_trail,
        execution_id="exec-001",
        request_id="req-001",
        agent_identity="agent-001",
        max_retrieval_calls=DEFAULT_MAX_RETRIEVAL_CALLS,
    )


# =============================================================================
# Tests: Configuration and Initialization
# =============================================================================


class TestBoundedRetrievalConfiguration:
    """Tests for configurable max_retrieval_calls parameter."""

    def test_default_max_retrieval_calls(self, bounded_service):
        """Default max should be 10."""
        assert bounded_service.max_retrieval_calls == 10

    def test_custom_max_retrieval_calls(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Custom max within range [1, 50] should be accepted."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=25,
        )
        assert service.max_retrieval_calls == 25

    def test_min_boundary(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Minimum allowed value is 1."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=1,
        )
        assert service.max_retrieval_calls == 1

    def test_max_boundary(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Maximum allowed value is 50."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=50,
        )
        assert service.max_retrieval_calls == 50

    def test_below_min_raises_value_error(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Value below 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be between"):
            BoundedRetrievalService(
                mcp_gateway=mock_mcp_gateway,
                signature_verifier=mock_signature_verifier,
                graph_service=mock_graph_service,
                audit_trail=mock_audit_trail,
                execution_id="exec-001",
                request_id="req-001",
                agent_identity="agent-001",
                max_retrieval_calls=0,
            )

    def test_above_max_raises_value_error(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Value above 50 should raise ValueError."""
        with pytest.raises(ValueError, match="must be between"):
            BoundedRetrievalService(
                mcp_gateway=mock_mcp_gateway,
                signature_verifier=mock_signature_verifier,
                graph_service=mock_graph_service,
                audit_trail=mock_audit_trail,
                execution_id="exec-001",
                request_id="req-001",
                agent_identity="agent-001",
                max_retrieval_calls=51,
            )


# =============================================================================
# Tests: Retrieval Limit Enforcement
# =============================================================================


class TestRetrievalLimitEnforcement:
    """Tests for retrieval call limit enforcement (Requirement 11.5, 11.6)."""

    async def test_initial_state_not_limited(self, bounded_service):
        """Service starts with zero calls and limit not reached."""
        assert bounded_service.call_count == 0
        assert bounded_service.is_limit_reached is False

    async def test_calls_increment_counter(self, bounded_service):
        """Each successful retrieval increments the counter."""
        await bounded_service.retrieve_additional("query 1")
        assert bounded_service.call_count == 1

        await bounded_service.retrieve_additional("query 2")
        assert bounded_service.call_count == 2

    async def test_limit_reached_after_max_calls(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Limit is reached after max_retrieval_calls calls."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=3,
        )

        # Make 3 calls (the max)
        for i in range(3):
            result = await service.retrieve_additional(f"query {i}")
            assert len(result) > 0

        assert service.is_limit_reached is True
        assert service.call_count == 3

    async def test_returns_empty_after_limit(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """After limit reached, retrieval returns empty list."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=2,
        )

        # Exhaust the limit
        await service.retrieve_additional("query 1")
        await service.retrieve_additional("query 2")

        # Next call returns empty
        result = await service.retrieve_additional("query 3")
        assert result == []

    async def test_limit_reached_event_recorded_in_trace(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """retrieval_limit_reached event is recorded in audit trail."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=1,
        )

        # Make the one allowed call
        await service.retrieve_additional("query 1")

        # Trigger the limit
        await service.retrieve_additional("query 2")

        # Find the retrieval_limit_reached record_entry call
        limit_calls = [
            call
            for call in mock_audit_trail.record_entry.call_args_list
            if call.kwargs.get("details", {}).get("event") == "retrieval_limit_reached"
            or (call.args and len(call.args) > 3 and isinstance(call.args[3], dict) and call.args[3].get("event") == "retrieval_limit_reached")
        ]

        # Check via kwargs
        found_limit_event = False
        for call in mock_audit_trail.record_entry.call_args_list:
            details = call.kwargs.get("details") or (call.args[3] if len(call.args) > 3 else None)
            if details and details.get("event") == "retrieval_limit_reached":
                found_limit_event = True
                assert details["calls_made"] == 1
                assert details["limit"] == 1
                break

        assert found_limit_event, "retrieval_limit_reached event not found in audit trail"

    async def test_limit_event_logged_only_once(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """retrieval_limit_reached event is only logged once even with multiple attempts."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=1,
        )

        await service.retrieve_additional("query 1")
        # Trigger limit multiple times
        await service.retrieve_additional("query 2")
        await service.retrieve_additional("query 3")
        await service.retrieve_additional("query 4")

        # Count retrieval_limit_reached events
        limit_event_count = 0
        for call in mock_audit_trail.record_entry.call_args_list:
            details = call.kwargs.get("details")
            if details and details.get("event") == "retrieval_limit_reached":
                limit_event_count += 1

        assert limit_event_count == 1

    async def test_no_mcp_call_after_limit(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """MCP Gateway is not called after limit is reached."""
        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
            max_retrieval_calls=1,
        )

        await service.retrieve_additional("query 1")
        initial_call_count = mock_mcp_gateway.invoke_tool.call_count

        # After limit
        await service.retrieve_additional("query 2")
        assert mock_mcp_gateway.invoke_tool.call_count == initial_call_count


# =============================================================================
# Tests: MCP Gateway Integration
# =============================================================================


class TestMCPGatewayIntegration:
    """Tests that all retrieval goes through MCP Gateway (Requirement 11.1)."""

    async def test_uses_approved_retrieval_tool(self, bounded_service, mock_mcp_gateway):
        """Retrieval uses the approved qdrant_retrieval tool."""
        await bounded_service.retrieve_additional("clinical query")

        mock_mcp_gateway.invoke_tool.assert_called_once()
        call_args = mock_mcp_gateway.invoke_tool.call_args
        assert call_args.kwargs["tool_name"] == APPROVED_RETRIEVAL_TOOL

    async def test_passes_query_parameters(self, bounded_service, mock_mcp_gateway):
        """Query and top_k parameters are passed to MCP Gateway."""
        await bounded_service.retrieve_additional("find diagnosis", top_k=10)

        call_args = mock_mcp_gateway.invoke_tool.call_args
        params = call_args.kwargs["parameters"]
        assert params["query"] == "find diagnosis"
        assert params["top_k"] == 10

    async def test_passes_namespace_when_provided(self, bounded_service, mock_mcp_gateway):
        """Namespace is included in parameters when provided."""
        await bounded_service.retrieve_additional("query", namespace="tenant-a")

        call_args = mock_mcp_gateway.invoke_tool.call_args
        params = call_args.kwargs["parameters"]
        assert params["namespace"] == "tenant-a"

    async def test_passes_agent_identity(self, bounded_service, mock_mcp_gateway):
        """Agent identity is passed for audit trail."""
        await bounded_service.retrieve_additional("query")

        call_args = mock_mcp_gateway.invoke_tool.call_args
        assert call_args.kwargs["agent_identity"] == "agent-001"


# =============================================================================
# Tests: KMS Signature Verification
# =============================================================================


class TestKMSSignatureVerification:
    """Tests for KMS signature verification (Requirements 11.2, 11.3)."""

    async def test_valid_signatures_pass_through(
        self, bounded_service, mock_signature_verifier
    ):
        """Chunks with valid KMS signatures are returned."""
        result = await bounded_service.retrieve_additional("query")
        assert len(result) == 3  # All 3 chunks are valid

    async def test_invalid_signatures_discarded(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Chunks with invalid signatures are discarded."""
        # Set up verifier to reject some chunks
        valid_chunk = _make_scored_chunk("chunk-valid", "Valid content")
        invalid_alert = _make_tamper_alert("chunk-invalid")

        async def verify_with_failures(chunks):
            return ([valid_chunk], [invalid_alert])

        mock_signature_verifier.verify_signatures.side_effect = verify_with_failures

        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
        )

        result = await service.retrieve_additional("query")
        assert len(result) == 1
        assert result[0].chunk_id == "chunk-valid"

    async def test_tamper_alerts_logged(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Tamper alerts are logged for invalid chunks."""
        alert = _make_tamper_alert("chunk-bad")

        async def verify_with_alert(chunks):
            return ([], [alert])

        mock_signature_verifier.verify_signatures.side_effect = verify_with_alert

        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
        )

        await service.retrieve_additional("query")

        # Tamper alert should be recorded in audit trail
        found_tamper = False
        for call in mock_audit_trail.record_entry.call_args_list:
            details = call.kwargs.get("details")
            if details and details.get("action") == "tamper_alert":
                found_tamper = True
                assert details["chunk_id"] == "chunk-bad"
                break

        assert found_tamper, "Tamper alert not found in audit trail"

    async def test_tamper_alerts_accumulated(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Tamper alerts from multiple calls are accumulated."""
        call_count = [0]

        async def verify_with_varying_alerts(chunks):
            call_count[0] += 1
            alert = _make_tamper_alert(f"chunk-bad-{call_count[0]}")
            valid = [_make_scored_chunk(f"chunk-ok-{call_count[0]}")]
            return (valid, [alert])

        mock_signature_verifier.verify_signatures.side_effect = verify_with_varying_alerts

        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
        )

        await service.retrieve_additional("query 1")
        await service.retrieve_additional("query 2")

        assert len(service.tamper_alerts) == 2


# =============================================================================
# Tests: Graph Provenance Recording
# =============================================================================


class TestGraphProvenanceRecording:
    """Tests for graph updates with execution_id provenance (Requirement 11.4)."""

    async def test_graph_update_includes_execution_id(
        self, bounded_service, mock_graph_service
    ):
        """Graph upsert receives execution_id as provenance."""
        await bounded_service.record_graph_update(
            node_type="EvidenceSource",
            node_id="ev-001",
            properties={"source": "clinical_note"},
        )

        mock_graph_service.upsert_node.assert_called_once_with(
            node_type="EvidenceSource",
            node_id="ev-001",
            properties={"source": "clinical_note"},
            execution_id="exec-001",
        )

    async def test_graph_update_recorded_in_audit_trail(
        self, bounded_service, mock_audit_trail
    ):
        """Graph updates are recorded in the audit trail."""
        await bounded_service.record_graph_update(
            node_type="Event",
            node_id="event-001",
            properties={"type": "diagnosis"},
        )

        found_graph_update = False
        for call in mock_audit_trail.record_entry.call_args_list:
            details = call.kwargs.get("details")
            if details and details.get("action") == "graph_update":
                found_graph_update = True
                assert details["node_type"] == "Event"
                assert details["node_id"] == "event-001"
                assert details["execution_id"] == "exec-001"
                break

        assert found_graph_update


# =============================================================================
# Tests: Lineage Tracking
# =============================================================================


class TestLineageTracking:
    """Tests for retrieved snippet lineage tracking (Requirement 11.7)."""

    async def test_retrieved_snippets_accumulated(self, bounded_service):
        """All valid retrieved snippets are accumulated for lineage."""
        await bounded_service.retrieve_additional("query 1")
        await bounded_service.retrieve_additional("query 2")

        snippets = bounded_service.get_retrieved_snippets()
        # 3 chunks per call, 2 calls = 6 total
        assert len(snippets) == 6

    async def test_get_lineage_entries_contains_all_snippets(self, bounded_service):
        """Lineage entries contain all retrieved snippet IDs."""
        await bounded_service.retrieve_additional("query")

        lineage = bounded_service.get_lineage_entries()
        assert len(lineage) == 3  # 3 chunks from single call
        for entry in lineage:
            assert entry.conclusion  # Non-empty conclusion
            assert entry.evidence_id  # Non-empty evidence_id
            assert entry.retrieval_timestamp  # Non-null timestamp

    async def test_lineage_entries_are_lineage_entry_type(self, bounded_service):
        """Lineage entries are of type LineageEntry."""
        await bounded_service.retrieve_additional("query")
        lineage = bounded_service.get_lineage_entries()
        for entry in lineage:
            assert isinstance(entry, LineageEntry)

    async def test_invalid_chunks_excluded_from_lineage(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Only valid (verified) chunks appear in lineage."""
        valid_chunk = _make_scored_chunk("chunk-valid", "Valid evidence")
        alert = _make_tamper_alert("chunk-invalid")

        async def partial_verify(chunks):
            return ([valid_chunk], [alert])

        mock_signature_verifier.verify_signatures.side_effect = partial_verify

        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
        )

        await service.retrieve_additional("query")
        snippets = service.get_retrieved_snippets()
        assert len(snippets) == 1
        assert snippets[0].chunk_id == "chunk-valid"

    async def test_empty_snippets_initially(self, bounded_service):
        """Before any retrieval, snippets list is empty."""
        assert bounded_service.get_retrieved_snippets() == []
        assert bounded_service.get_lineage_entries() == []


# =============================================================================
# Tests: Call Records
# =============================================================================


class TestCallRecords:
    """Tests for RetrievalCallRecord tracking."""

    async def test_call_records_created(self, bounded_service):
        """Each retrieval creates a RetrievalCallRecord."""
        await bounded_service.retrieve_additional("query")
        records = bounded_service.call_records
        assert len(records) == 1
        assert records[0].call_number == 1
        assert records[0].query == "query"
        assert records[0].execution_id == "exec-001"

    async def test_call_record_tracks_chunk_counts(
        self, mock_mcp_gateway, mock_signature_verifier, mock_graph_service, mock_audit_trail
    ):
        """Call records track total, valid, and discarded chunk counts."""
        valid_chunk = _make_scored_chunk("chunk-valid")
        alert = _make_tamper_alert("chunk-bad")

        async def verify_mixed(chunks):
            return ([valid_chunk], [alert])

        mock_signature_verifier.verify_signatures.side_effect = verify_mixed

        service = BoundedRetrievalService(
            mcp_gateway=mock_mcp_gateway,
            signature_verifier=mock_signature_verifier,
            graph_service=mock_graph_service,
            audit_trail=mock_audit_trail,
            execution_id="exec-001",
            request_id="req-001",
            agent_identity="agent-001",
        )

        await service.retrieve_additional("query")
        records = service.call_records
        assert records[0].chunks_valid == 1
        assert records[0].chunks_discarded == 1

    async def test_call_record_has_timestamp(self, bounded_service):
        """Call records have timestamps."""
        await bounded_service.retrieve_additional("query")
        records = bounded_service.call_records
        assert records[0].timestamp is not None
        assert isinstance(records[0].timestamp, datetime)
