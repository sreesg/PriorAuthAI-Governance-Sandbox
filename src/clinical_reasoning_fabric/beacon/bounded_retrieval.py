"""
BoundedRetrievalService — Bounded Agent Reasoning with Targeted Retrieval.

Implements retrieval call limiting during agent reasoning, enforcing a configurable
maximum number of retrieval calls per PA request (default 10, range 1-50). All
retrieval calls go through the MCP Gateway using the approved retrieval tool.
KMS signatures are verified on newly retrieved chunks, and tamper alerts are
logged for invalid ones.

Requirements referenced: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from clinical_reasoning_fabric.models.core import (
    LineageEntry,
    ScoredChunk,
    TamperAlert,
    TraceCategory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_RETRIEVAL_CALLS: int = 10
MIN_RETRIEVAL_CALLS: int = 1
MAX_RETRIEVAL_CALLS: int = 50

# The approved retrieval tool name in the MCP Gateway catalog
APPROVED_RETRIEVAL_TOOL: str = "qdrant_retrieval"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RetrievalCallRecord:
    """Records metadata for each retrieval call made during agent reasoning.

    Tracks query, execution time, results, and provenance for audit trail.
    """

    call_number: int
    query: str
    execution_id: str
    timestamp: datetime
    chunks_returned: int
    chunks_valid: int
    chunks_discarded: int
    tamper_alerts: list[TamperAlert] = field(default_factory=list)
    duration_ms: int = 0


# =============================================================================
# Protocols for Dependency Injection
# =============================================================================


@runtime_checkable
class MCPGatewayProtocol(Protocol):
    """Protocol for MCP Gateway tool invocation."""

    async def invoke_tool(
        self, tool_name: str, parameters: dict[str, Any], agent_identity: str
    ) -> Any:
        """Invoke an approved tool through the gateway."""
        ...


@runtime_checkable
class SignatureVerifierProtocol(Protocol):
    """Protocol for KMS signature verification on chunks."""

    async def verify_signatures(
        self, chunks: list[ScoredChunk]
    ) -> tuple[list[ScoredChunk], list[TamperAlert]]:
        """Verify KMS signatures on chunks, returning verified and tamper alerts."""
        ...


@runtime_checkable
class GraphServiceProtocol(Protocol):
    """Protocol for Causal Ontology Graph operations."""

    async def upsert_node(
        self,
        node_type: str,
        node_id: str,
        properties: dict[str, Any],
        execution_id: Optional[str] = None,
    ) -> None:
        """Upsert a graph node with optional execution provenance."""
        ...


@runtime_checkable
class AuditTrailProtocol(Protocol):
    """Protocol for audit trail recording."""

    async def record_entry(
        self,
        request_id: str,
        identity_id: str,
        category: TraceCategory,
        details: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Record a trace entry."""
        ...


# =============================================================================
# BoundedRetrievalService
# =============================================================================


class BoundedRetrievalService:
    """Service that enforces bounded retrieval during agent reasoning.

    Tracks the number of retrieval calls made per PA request and enforces
    a configurable maximum. All retrieval calls go through the MCP Gateway
    using the approved retrieval tool. KMS signatures are verified on newly
    retrieved chunks; invalid ones are discarded with tamper alerts logged.

    Requirement 11.1: Additional retrieval calls exclusively through MCP Gateway.
    Requirement 11.2: Verify KMS signatures on newly retrieved chunks.
    Requirement 11.3: Discard invalid chunks, log tamper alert.
    Requirement 11.4: Graph updates record execution_id as provenance.
    Requirement 11.5: Configurable maximum retrieval calls (default 10, range 1-50).
    Requirement 11.6: On limit reached, proceed with available evidence, record event.
    Requirement 11.7: Include all retrieved snippets with provenance in lineage_trail.
    """

    def __init__(
        self,
        mcp_gateway: MCPGatewayProtocol,
        signature_verifier: SignatureVerifierProtocol,
        graph_service: GraphServiceProtocol,
        audit_trail: AuditTrailProtocol,
        execution_id: str,
        request_id: str,
        agent_identity: str,
        max_retrieval_calls: int = DEFAULT_MAX_RETRIEVAL_CALLS,
    ) -> None:
        """Initialize BoundedRetrievalService for a single PA request.

        Args:
            mcp_gateway: MCP Gateway for tool invocations.
            signature_verifier: Service for KMS signature verification.
            graph_service: Causal Ontology Graph service for node upserts.
            audit_trail: Audit trail service for recording trace events.
            execution_id: Unique execution ID for this PA request.
            request_id: Request ID for audit trail correlation.
            agent_identity: Authenticated agent identity.
            max_retrieval_calls: Maximum retrieval calls allowed (default 10, range 1-50).

        Raises:
            ValueError: If max_retrieval_calls is outside range [1, 50].
        """
        # Validate max_retrieval_calls within allowed range
        if not (MIN_RETRIEVAL_CALLS <= max_retrieval_calls <= MAX_RETRIEVAL_CALLS):
            raise ValueError(
                f"max_retrieval_calls must be between {MIN_RETRIEVAL_CALLS} and "
                f"{MAX_RETRIEVAL_CALLS}, got {max_retrieval_calls}"
            )

        self._mcp_gateway = mcp_gateway
        self._signature_verifier = signature_verifier
        self._graph_service = graph_service
        self._audit_trail = audit_trail
        self._execution_id = execution_id
        self._request_id = request_id
        self._agent_identity = agent_identity
        self._max_retrieval_calls = max_retrieval_calls

        # State tracking
        self._call_count: int = 0
        self._call_records: list[RetrievalCallRecord] = []
        self._retrieved_snippets: list[ScoredChunk] = []
        self._all_tamper_alerts: list[TamperAlert] = []
        self._limit_reached_logged: bool = False

    @property
    def is_limit_reached(self) -> bool:
        """Whether the retrieval call limit has been reached.

        Requirement 11.5: Configurable maximum with default 10.
        """
        return self._call_count >= self._max_retrieval_calls

    @property
    def call_count(self) -> int:
        """Number of retrieval calls made so far."""
        return self._call_count

    @property
    def max_retrieval_calls(self) -> int:
        """Configured maximum retrieval calls."""
        return self._max_retrieval_calls

    @property
    def call_records(self) -> list[RetrievalCallRecord]:
        """All retrieval call records for this request."""
        return list(self._call_records)

    @property
    def tamper_alerts(self) -> list[TamperAlert]:
        """All tamper alerts generated during retrieval."""
        return list(self._all_tamper_alerts)

    async def retrieve_additional(
        self, query: str, top_k: int = 20, namespace: Optional[str] = None
    ) -> list[ScoredChunk]:
        """Retrieve additional evidence during agent reasoning.

        Checks if the retrieval limit has been reached. If so, logs a
        retrieval_limit_reached event and returns empty. Otherwise, calls
        the MCP Gateway with the approved retrieval tool, verifies KMS
        signatures, discards invalid chunks with tamper alerts, and returns
        valid chunks with provenance.

        Args:
            query: The retrieval query string.
            top_k: Maximum number of chunks to retrieve.
            namespace: Optional namespace scope for retrieval.

        Returns:
            List of valid ScoredChunks with verified signatures.
            Empty list if limit reached or no valid results.

        Requirement 11.1: Retrieval through MCP Gateway using approved tool.
        Requirement 11.2: Verify KMS signatures on new chunks.
        Requirement 11.3: Discard invalid chunks, log tamper alert.
        Requirement 11.5: Limit enforcement.
        Requirement 11.6: On limit reached, proceed with available evidence.
        """
        # Check if limit has been reached
        if self.is_limit_reached:
            await self._handle_limit_reached()
            return []

        import time

        start_time = time.monotonic()
        timestamp = datetime.now(timezone.utc)

        # Build parameters for the approved retrieval tool
        parameters: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
        }
        if namespace:
            parameters["namespace"] = namespace

        # Invoke retrieval through MCP Gateway (Requirement 11.1)
        tool_result = await self._mcp_gateway.invoke_tool(
            tool_name=APPROVED_RETRIEVAL_TOOL,
            parameters=parameters,
            agent_identity=self._agent_identity,
        )

        # Extract chunks from tool result
        raw_chunks = self._extract_chunks_from_result(tool_result)

        # Verify KMS signatures on returned chunks (Requirement 11.2)
        verified_chunks, tamper_alerts = await self._signature_verifier.verify_signatures(
            raw_chunks
        )

        # Log tamper alerts for discarded chunks (Requirement 11.3)
        if tamper_alerts:
            self._all_tamper_alerts.extend(tamper_alerts)
            for alert in tamper_alerts:
                logger.warning(
                    "Tamper alert during bounded retrieval | "
                    f"execution_id={self._execution_id} | "
                    f"chunk_id={alert.chunk_id} | "
                    f"reason={alert.reason}"
                )
                # Record tamper alert in audit trail
                await self._audit_trail.record_entry(
                    request_id=self._request_id,
                    identity_id=self._agent_identity,
                    category=TraceCategory.AGENT_ACTION,
                    details={
                        "action": "tamper_alert",
                        "chunk_id": alert.chunk_id,
                        "document_id": alert.document_id,
                        "reason": alert.reason,
                        "execution_id": self._execution_id,
                    },
                )

        # Increment call counter
        self._call_count += 1

        # Store valid snippets for lineage trail (Requirement 11.7)
        self._retrieved_snippets.extend(verified_chunks)

        # Calculate duration
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Record this retrieval call
        record = RetrievalCallRecord(
            call_number=self._call_count,
            query=query,
            execution_id=self._execution_id,
            timestamp=timestamp,
            chunks_returned=len(raw_chunks),
            chunks_valid=len(verified_chunks),
            chunks_discarded=len(tamper_alerts),
            tamper_alerts=tamper_alerts,
            duration_ms=duration_ms,
        )
        self._call_records.append(record)

        # Record retrieval in audit trail
        await self._audit_trail.record_entry(
            request_id=self._request_id,
            identity_id=self._agent_identity,
            category=TraceCategory.CONTEXT_RETRIEVAL,
            details={
                "action": "bounded_retrieval",
                "call_number": self._call_count,
                "query": query,
                "chunks_returned": len(raw_chunks),
                "chunks_valid": len(verified_chunks),
                "chunks_discarded": len(tamper_alerts),
                "execution_id": self._execution_id,
                "limit": self._max_retrieval_calls,
            },
        )

        return verified_chunks

    async def record_graph_update(
        self,
        node_type: str,
        node_id: str,
        properties: dict[str, Any],
    ) -> None:
        """Record a graph update with execution_id as provenance.

        Requirement 11.4: When the agent updates the clinical graph during
        reasoning, the Causal_Ontology_Graph SHALL record the update with
        the agent execution_id as provenance.

        Args:
            node_type: Type of the graph node to upsert.
            node_id: Unique identifier for the node.
            properties: Properties to set on the node.
        """
        await self._graph_service.upsert_node(
            node_type=node_type,
            node_id=node_id,
            properties=properties,
            execution_id=self._execution_id,
        )

        # Record in audit trail
        await self._audit_trail.record_entry(
            request_id=self._request_id,
            identity_id=self._agent_identity,
            category=TraceCategory.AGENT_ACTION,
            details={
                "action": "graph_update",
                "node_type": node_type,
                "node_id": node_id,
                "execution_id": self._execution_id,
            },
        )

    def get_retrieved_snippets(self) -> list[ScoredChunk]:
        """Return all valid snippets retrieved during reasoning for lineage_trail.

        Requirement 11.7: All additionally retrieved evidence snippets with
        provenance metadata included in Evidence_Bundle lineage_trail.

        Returns:
            List of all valid ScoredChunks retrieved across all calls.
        """
        return list(self._retrieved_snippets)

    def get_lineage_entries(self) -> list[LineageEntry]:
        """Convert retrieved snippets to LineageEntry objects for Evidence_Bundle.

        Requirement 11.7: Include all additionally retrieved snippets with
        provenance in Evidence_Bundle lineage_trail.

        Returns:
            List of LineageEntry objects with conclusion, evidence_id,
            and retrieval_timestamp for each retrieved snippet.
        """
        lineage_entries: list[LineageEntry] = []
        for snippet in self._retrieved_snippets:
            entry = LineageEntry(
                conclusion=f"Additional evidence retrieved: {snippet.text[:200]}",
                evidence_id=snippet.chunk_id,
                retrieval_timestamp=snippet.provenance.ingestion_timestamp,
            )
            lineage_entries.append(entry)
        return lineage_entries

    async def _handle_limit_reached(self) -> None:
        """Handle the case where retrieval limit is reached.

        Requirement 11.6: On limit reached, proceed to decision with available
        evidence and record retrieval_limit_reached event in trace with call
        count and limit value.
        """
        if not self._limit_reached_logged:
            self._limit_reached_logged = True

            logger.info(
                "Retrieval limit reached | "
                f"execution_id={self._execution_id} | "
                f"calls_made={self._call_count} | "
                f"limit={self._max_retrieval_calls}"
            )

            # Record retrieval_limit_reached event in execution trace
            await self._audit_trail.record_entry(
                request_id=self._request_id,
                identity_id=self._agent_identity,
                category=TraceCategory.AGENT_ACTION,
                details={
                    "event": "retrieval_limit_reached",
                    "calls_made": self._call_count,
                    "limit": self._max_retrieval_calls,
                    "execution_id": self._execution_id,
                },
            )

    def _extract_chunks_from_result(self, tool_result: Any) -> list[ScoredChunk]:
        """Extract ScoredChunk objects from the MCP Gateway tool result.

        Handles both direct list returns and ToolResult wrapper objects.

        Args:
            tool_result: The result from MCP Gateway invoke_tool.

        Returns:
            List of ScoredChunk objects extracted from the result.
        """
        # Handle ToolResult wrapper (has output_result attribute)
        if hasattr(tool_result, "output_result"):
            result_data = tool_result.output_result
        elif hasattr(tool_result, "success") and hasattr(tool_result, "output_result"):
            result_data = tool_result.output_result
        else:
            result_data = tool_result

        # If the result is None (tool failed), return empty
        if result_data is None:
            return []

        # If it's already a list of ScoredChunks, return directly
        if isinstance(result_data, list):
            scored_chunks: list[ScoredChunk] = []
            for item in result_data:
                if isinstance(item, ScoredChunk):
                    scored_chunks.append(item)
                elif isinstance(item, dict):
                    # Try to construct ScoredChunk from dict
                    try:
                        scored_chunks.append(ScoredChunk(**item))
                    except Exception:
                        logger.warning(
                            f"Could not parse chunk from tool result: {item}"
                        )
            return scored_chunks

        # If it's a RetrievalResult-like object with verified_chunks
        if hasattr(result_data, "verified_chunks"):
            return list(result_data.verified_chunks)

        return []
