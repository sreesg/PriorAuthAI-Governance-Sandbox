"""CDC Pipeline Service — Manages Change Data Capture from Snowflake/Iceberg to Neo4j.

Uses dbt model definitions for transformation and Dagster for orchestration.
Projects current clinical state into the Causal Ontology Graph via upsert
operations that preserve unrelated relationships.

Requirements referenced: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

from clinical_reasoning_fabric.models.core import CDCEvent, EventCheckpoint
from clinical_reasoning_fabric.models.exceptions import UnmappableRecordError

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Valid node types that can be mapped from CDC events
VALID_NODE_TYPES = frozenset({
    "Member",
    "Event",
    "PolicyRule",
    "SDOH_Factor",
    "EvidenceSource",
})

# Valid relationship types that can be mapped from CDC events
VALID_RELATIONSHIP_TYPES = frozenset({
    "HAS_CONDITION",
    "IS_PRESCRIBED",
    "TRIGGERED_BY",
    "GOVERNED_BY",
    "EVIDENCED_BY",
})

# Mapping from source table prefixes to graph entity types (dbt model definitions)
# This simulates the dbt model mapping layer
SOURCE_TABLE_TO_NODE_TYPE: dict[str, str] = {
    "members": "Member",
    "member": "Member",
    "events": "Event",
    "event": "Event",
    "clinical_events": "Event",
    "policy_rules": "PolicyRule",
    "policy_rule": "PolicyRule",
    "policies": "PolicyRule",
    "sdoh_factors": "SDOH_Factor",
    "sdoh_factor": "SDOH_Factor",
    "sdoh": "SDOH_Factor",
    "evidence_sources": "EvidenceSource",
    "evidence_source": "EvidenceSource",
    "evidence": "EvidenceSource",
}

# Mapping from source table prefixes to relationship types
SOURCE_TABLE_TO_RELATIONSHIP_TYPE: dict[str, str] = {
    "member_conditions": "HAS_CONDITION",
    "has_condition": "HAS_CONDITION",
    "member_prescriptions": "IS_PRESCRIBED",
    "is_prescribed": "IS_PRESCRIBED",
    "event_triggers": "TRIGGERED_BY",
    "triggered_by": "TRIGGERED_BY",
    "event_policies": "GOVERNED_BY",
    "governed_by": "GOVERNED_BY",
    "event_evidence": "EVIDENCED_BY",
    "evidenced_by": "EVIDENCED_BY",
}

# Retry configuration
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5  # 5s, 10s, 20s exponential backoff


# =============================================================================
# Graph Entity Types
# =============================================================================


class GraphEntityKind(str, Enum):
    """Whether the graph entity is a node or a relationship."""

    NODE = "node"
    RELATIONSHIP = "relationship"


@dataclass
class GraphEntity:
    """Represents a transformed CDC event ready for graph upsert.

    This is the output of dbt model transformation — a source record
    mapped to its corresponding graph node or relationship type.
    """

    kind: GraphEntityKind
    entity_type: str  # Node type (Member, Event, etc.) or relationship type (HAS_CONDITION, etc.)
    entity_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    # For relationships only
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    # Metadata
    source_event_id: str = ""
    source_commit_timestamp: Optional[datetime] = None
    operation: str = "UPSERT"  # INSERT/UPDATE map to UPSERT for idempotency


@dataclass
class ProcessingResult:
    """Result of processing a single CDC event."""

    event_id: str
    success: bool
    entity: Optional[GraphEntity] = None
    error: Optional[str] = None
    retries_attempted: int = 0
    processed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class BatchProcessingResult:
    """Result of processing a batch of CDC events.

    Requirements: 12.5, 12.6, 12.7
    """

    total_events: int
    successful: int = 0
    failed: int = 0
    skipped_unmappable: int = 0
    skipped_already_processed: int = 0
    results: list[ProcessingResult] = field(default_factory=list)
    checkpoint_after: Optional[EventCheckpoint] = None


# =============================================================================
# Protocol for graph service dependency (CausalOntologyGraphService)
# =============================================================================


class GraphServiceProtocol(Protocol):
    """Protocol defining the graph service interface needed by CDC pipeline."""

    async def upsert_node(
        self, node_type: str, node_id: str, properties: dict, execution_id: str | None = None
    ) -> None: ...

    async def upsert_relationship(
        self, source_id: str, target_id: str, rel_type: str, properties: dict
    ) -> None: ...


# =============================================================================
# Protocol for Dagster client dependency
# =============================================================================


class DagsterClientProtocol(Protocol):
    """Protocol defining the Dagster client interface for orchestration."""

    async def trigger_run(self, job_name: str, run_config: dict) -> str: ...
    async def get_run_status(self, run_id: str) -> str: ...


# =============================================================================
# Checkpoint Store Protocol and Implementations
# =============================================================================


class CheckpointStoreProtocol(Protocol):
    """Protocol for persistent checkpoint storage backends."""

    def load(self) -> Optional[EventCheckpoint]: ...
    def save(self, checkpoint: EventCheckpoint) -> None: ...


class InMemoryCheckpointStore:
    """In-memory checkpoint store for testing purposes."""

    def __init__(self) -> None:
        self._checkpoint: Optional[EventCheckpoint] = None

    def load(self) -> Optional[EventCheckpoint]:
        return self._checkpoint

    def save(self, checkpoint: EventCheckpoint) -> None:
        self._checkpoint = checkpoint


class FileCheckpointStore:
    """JSON file-based persistent checkpoint store.

    Persists the last successfully processed event checkpoint to a JSON file,
    enabling pipeline restart from the last checkpoint without duplicating
    graph mutations.

    Requirements: 12.5
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._path = Path(checkpoint_path)
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[EventCheckpoint]:
        """Load checkpoint from disk. Returns None if file doesn't exist."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return EventCheckpoint(
                last_event_id=data["last_event_id"],
                last_source_commit_timestamp=datetime.fromisoformat(
                    data["last_source_commit_timestamp"]
                ),
                last_processed_at=datetime.fromisoformat(data["last_processed_at"]),
                total_events_processed=data.get("total_events_processed", 0),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "Failed to load checkpoint from %s: %s. Starting fresh.",
                self._path,
                str(e),
            )
            return None

    def save(self, checkpoint: EventCheckpoint) -> None:
        """Persist checkpoint to disk atomically using write-then-rename."""
        data = {
            "last_event_id": checkpoint.last_event_id,
            "last_source_commit_timestamp": checkpoint.last_source_commit_timestamp.isoformat(),
            "last_processed_at": checkpoint.last_processed_at.isoformat(),
            "total_events_processed": checkpoint.total_events_processed,
        }
        # Write to a temp file then rename for atomicity
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(self._path)


# =============================================================================
# CDC Pipeline Service
# =============================================================================


class CDCPipelineService:
    """Manages CDC from Snowflake/Iceberg to Neo4j via dbt models and Dagster orchestration.

    Transforms source records into graph node/relationship types using dbt model
    definitions, then applies changes as upserts that update target properties
    without removing unrelated relationships.

    Requirements:
        12.1: Detect changes within 5 minutes of source commit timestamp
        12.2: Transform records using dbt models into corresponding graph types
        12.3: Apply as upserts without removing unrelated relationships
        12.4: Retry up to 3 times with exponential backoff starting at 5 seconds
        12.5: Maintain event checkpoint for resume from last successfully processed
        12.6: Skip unmappable records with logging, continue processing
        12.7: Apply events for same entity in source-commit-timestamp order
    """

    def __init__(
        self,
        dagster_client: DagsterClientProtocol,
        neo4j_service: GraphServiceProtocol,
        checkpoint_store: Optional[CheckpointStoreProtocol] = None,
    ) -> None:
        self.dagster = dagster_client
        self.graph = neo4j_service
        self._checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self._checkpoint: Optional[EventCheckpoint] = self._checkpoint_store.load()

    async def process_batch(self, events: list[CDCEvent]) -> BatchProcessingResult:
        """Process a batch of CDC events in source-commit-timestamp order.

        Sorts events by source_commit_timestamp before processing to ensure
        that multiple events for the same entity are applied in the correct order.
        Skips unmappable records (logs and continues), and resumes from the
        last checkpoint to avoid duplicating graph mutations.

        Args:
            events: List of CDC events to process.

        Returns:
            BatchProcessingResult with per-event results and final checkpoint.

        Requirements: 12.5, 12.6, 12.7
        """
        if not events:
            return BatchProcessingResult(total_events=0)

        # Step 1: Sort events by source_commit_timestamp (Requirement 12.7)
        sorted_events = sorted(events, key=lambda e: e.source_commit_timestamp)

        # Step 2: Filter out already-processed events based on checkpoint (Req 12.5)
        events_to_process = self.resume_from_checkpoint(sorted_events)

        batch_result = BatchProcessingResult(
            total_events=len(events),
            skipped_already_processed=len(sorted_events) - len(events_to_process),
        )

        # Step 3: Process each event in order
        for event in events_to_process:
            result = await self.process_change_event(event)
            batch_result.results.append(result)

            if result.success:
                batch_result.successful += 1
            elif result.error and "UnmappableRecordError" in result.error:
                batch_result.skipped_unmappable += 1
                batch_result.failed += 1
                # Continue processing — unmappable records are skipped (Req 12.6)
                logger.info(
                    "Skipped unmappable record event_id=%s, continuing batch.",
                    event.event_id,
                )
            else:
                batch_result.failed += 1

        batch_result.checkpoint_after = self.get_checkpoint()
        return batch_result

    def resume_from_checkpoint(self, events: list[CDCEvent]) -> list[CDCEvent]:
        """Filter a batch of events to exclude those already processed.

        Uses the checkpoint's last_source_commit_timestamp and last_event_id
        to determine which events have already been successfully processed.
        Events with a timestamp earlier than or equal to the checkpoint timestamp
        AND the same event_id as the checkpoint's last_event_id are excluded.
        Events with timestamps strictly greater than the checkpoint are included.
        Events with the same timestamp but different event_ids are included
        (they may be concurrent events that weren't processed yet).

        This ensures idempotency: replaying from checkpoint does not duplicate
        graph mutations.

        Args:
            events: Sorted list of CDC events (by source_commit_timestamp).

        Returns:
            List of events that still need to be processed.

        Requirements: 12.5
        """
        checkpoint = self.get_checkpoint()
        if checkpoint is None:
            # No checkpoint — all events need processing
            return events

        filtered: list[CDCEvent] = []
        for event in events:
            # Skip events that are strictly before the checkpoint timestamp
            if event.source_commit_timestamp < checkpoint.last_source_commit_timestamp:
                continue
            # For events at exactly the checkpoint timestamp, skip only if it's
            # the exact event we already processed (by event_id)
            if (
                event.source_commit_timestamp == checkpoint.last_source_commit_timestamp
                and event.event_id == checkpoint.last_event_id
            ):
                continue
            # Include everything else
            filtered.append(event)

        return filtered

    async def process_change_event(self, event: CDCEvent) -> ProcessingResult:
        """Transform and apply a single CDC event to the graph.

        Transforms the CDC event into a graph entity using dbt model definitions,
        then applies the upsert to Neo4j. Retries up to 3 times with exponential
        backoff (5s, 10s, 20s) on apply failure.

        Args:
            event: The CDC event to process.

        Returns:
            ProcessingResult indicating success or failure with details.

        Requirements: 12.1, 12.2, 12.3, 12.4
        """
        # Step 1: Transform CDC event to graph entity
        try:
            entity = await self.transform_to_graph_entity(event)
        except UnmappableRecordError as e:
            logger.warning(
                "Skipping unmappable record: event_id=%s, entity_type=%s, reason=%s",
                event.event_id,
                event.entity_type,
                e.reason,
            )
            return ProcessingResult(
                event_id=event.event_id,
                success=False,
                error=f"UnmappableRecordError: {e.reason}",
            )

        # Step 2: Apply upsert with retry logic (Requirement 12.4)
        retries = 0
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                await self.apply_upsert(entity)
                # Success — update checkpoint
                self.update_checkpoint(
                    EventCheckpoint(
                        last_event_id=event.event_id,
                        last_source_commit_timestamp=event.source_commit_timestamp,
                        last_processed_at=datetime.now(timezone.utc),
                        total_events_processed=(
                            self._checkpoint.total_events_processed + 1
                            if self._checkpoint
                            else 1
                        ),
                    )
                )
                return ProcessingResult(
                    event_id=event.event_id,
                    success=True,
                    entity=entity,
                    retries_attempted=retries,
                )
            except Exception as e:
                last_error = e
                retries = attempt + 1

                if attempt < MAX_RETRIES:
                    # Exponential backoff: 5s, 10s, 20s
                    backoff_seconds = BASE_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Apply upsert failed (attempt %d/%d), retrying in %ds: "
                        "event_id=%s, error=%s",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        backoff_seconds,
                        event.event_id,
                        str(e),
                    )
                    await asyncio.sleep(backoff_seconds)

        # All retries exhausted — alert operations team
        error_msg = (
            f"Failed to apply upsert after {MAX_RETRIES} retries: {str(last_error)}"
        )
        logger.error(
            "CDC event processing failed permanently: event_id=%s, error=%s",
            event.event_id,
            error_msg,
        )
        return ProcessingResult(
            event_id=event.event_id,
            success=False,
            entity=entity,
            error=error_msg,
            retries_attempted=retries,
        )

    async def transform_to_graph_entity(self, event: CDCEvent) -> GraphEntity:
        """Map source record to graph node/relationship type using dbt model definitions.

        Uses the event's entity_type and source_table to determine the correct
        graph node or relationship type. The mapping follows dbt model definitions
        that define how source records project into the graph schema.

        Args:
            event: The CDC event containing the source record data.

        Returns:
            GraphEntity ready for upsert into the graph.

        Raises:
            UnmappableRecordError: If the source record cannot be mapped to a valid
                graph node or relationship type.

        Requirements: 12.2, 12.6
        """
        # Determine if this is a node or relationship based on entity_type and properties
        graph_type: Optional[str] = None
        kind: GraphEntityKind = GraphEntityKind.NODE

        # First, check if entity_type directly maps to a valid node type
        if event.entity_type in VALID_NODE_TYPES:
            graph_type = event.entity_type
            kind = GraphEntityKind.NODE
        else:
            # Check if entity_type maps to a relationship type
            # (for CDC events that represent relationship changes)
            if event.entity_type in VALID_RELATIONSHIP_TYPES:
                graph_type = event.entity_type
                kind = GraphEntityKind.RELATIONSHIP
            else:
                # Try source table mapping as fallback (dbt model mapping)
                source_table_lower = event.source_table.lower()

                # Check node type mapping
                for table_key, node_type in SOURCE_TABLE_TO_NODE_TYPE.items():
                    if table_key in source_table_lower:
                        graph_type = node_type
                        kind = GraphEntityKind.NODE
                        break

                # If not found in nodes, check relationship mapping
                if graph_type is None:
                    for table_key, rel_type in SOURCE_TABLE_TO_RELATIONSHIP_TYPE.items():
                        if table_key in source_table_lower:
                            graph_type = rel_type
                            kind = GraphEntityKind.RELATIONSHIP
                            break

        # If we still can't map it, raise UnmappableRecordError
        if graph_type is None:
            raise UnmappableRecordError(
                reason=(
                    f"Cannot map entity_type='{event.entity_type}' from "
                    f"source_table='{event.source_table}' to a valid graph "
                    f"node or relationship type"
                ),
                source_record_id=event.event_id,
                entity_type=event.entity_type,
            )

        # Build the GraphEntity
        properties = dict(event.properties)

        # For relationship entities, extract source/target from properties
        source_node_id: Optional[str] = None
        target_node_id: Optional[str] = None

        if kind == GraphEntityKind.RELATIONSHIP:
            source_node_id = properties.pop("source_id", None) or properties.pop(
                "source_node_id", None
            )
            target_node_id = properties.pop("target_id", None) or properties.pop(
                "target_node_id", None
            )

            if not source_node_id or not target_node_id:
                raise UnmappableRecordError(
                    reason=(
                        f"Relationship type '{graph_type}' requires source_id and "
                        f"target_id in properties, but one or both are missing"
                    ),
                    source_record_id=event.event_id,
                    entity_type=event.entity_type,
                )

        return GraphEntity(
            kind=kind,
            entity_type=graph_type,
            entity_id=event.entity_id,
            properties=properties,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_event_id=event.event_id,
            source_commit_timestamp=event.source_commit_timestamp,
            operation=event.operation,
        )

    async def apply_upsert(self, entity: GraphEntity) -> None:
        """Apply upsert to Neo4j without removing unrelated relationships.

        For nodes: updates only the properties specified in the entity,
        leaving all existing relationships and unspecified properties intact.

        For relationships: updates only the relationship properties between
        the specified source and target nodes, without affecting other
        relationships on those nodes.

        Args:
            entity: The GraphEntity to upsert.

        Raises:
            Exception: If the graph service operation fails.

        Requirements: 12.3
        """
        if entity.kind == GraphEntityKind.NODE:
            # Upsert node — this updates properties without removing relationships
            await self.graph.upsert_node(
                node_type=entity.entity_type,
                node_id=entity.entity_id,
                properties=entity.properties,
                execution_id=entity.source_event_id,
            )
        elif entity.kind == GraphEntityKind.RELATIONSHIP:
            # Upsert relationship — updates only this relationship's properties
            # without removing or modifying other relationships on the nodes
            if entity.source_node_id and entity.target_node_id:
                await self.graph.upsert_relationship(
                    source_id=entity.source_node_id,
                    target_id=entity.target_node_id,
                    rel_type=entity.entity_type,
                    properties=entity.properties,
                )
            else:
                raise ValueError(
                    f"Relationship entity '{entity.entity_id}' missing "
                    f"source_node_id or target_node_id"
                )

    def get_checkpoint(self) -> Optional[EventCheckpoint]:
        """Return the last successfully processed event checkpoint.

        Returns None if no events have been processed yet.

        Requirements: 12.5
        """
        return self._checkpoint

    def update_checkpoint(self, checkpoint: EventCheckpoint) -> None:
        """Update the event checkpoint after successful processing.

        Persists the checkpoint to the backing store to enable pipeline
        restart without duplicating mutations.

        Requirements: 12.5
        """
        self._checkpoint = checkpoint
        self._checkpoint_store.save(checkpoint)
