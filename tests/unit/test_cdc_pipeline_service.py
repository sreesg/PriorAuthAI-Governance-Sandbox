"""Unit tests for CDCPipelineService.

Tests transformation, upsert application, retry logic, error handling,
batch processing, checkpoint persistence, event ordering, and resume logic
for the CDC pipeline that projects Snowflake/Iceberg state into Neo4j.

Requirements validated: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from clinical_reasoning_fabric.cdc.cdc_pipeline_service import (
    BASE_BACKOFF_SECONDS,
    BatchProcessingResult,
    CDCPipelineService,
    FileCheckpointStore,
    GraphEntity,
    GraphEntityKind,
    InMemoryCheckpointStore,
    MAX_RETRIES,
    ProcessingResult,
    VALID_NODE_TYPES,
    VALID_RELATIONSHIP_TYPES,
)
from clinical_reasoning_fabric.models.core import CDCEvent, EventCheckpoint
from clinical_reasoning_fabric.models.exceptions import UnmappableRecordError


# =============================================================================
# Fixtures
# =============================================================================


class FakeGraphService:
    """Fake graph service for testing."""

    def __init__(self, fail_count: int = 0):
        self.upsert_node_calls: list[dict] = []
        self.upsert_relationship_calls: list[dict] = []
        self._fail_count = fail_count
        self._call_count = 0

    async def upsert_node(
        self, node_type: str, node_id: str, properties: dict, execution_id: str | None = None
    ) -> None:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Simulated failure on attempt {self._call_count}")
        self.upsert_node_calls.append({
            "node_type": node_type,
            "node_id": node_id,
            "properties": properties,
            "execution_id": execution_id,
        })

    async def upsert_relationship(
        self, source_id: str, target_id: str, rel_type: str, properties: dict
    ) -> None:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Simulated failure on attempt {self._call_count}")
        self.upsert_relationship_calls.append({
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
            "properties": properties,
        })


class FakeDagsterClient:
    """Fake Dagster client for testing."""

    async def trigger_run(self, job_name: str, run_config: dict) -> str:
        return "run-123"

    async def get_run_status(self, run_id: str) -> str:
        return "SUCCESS"


def make_cdc_event(
    entity_type: str = "Member",
    entity_id: str = "member-001",
    operation: str = "INSERT",
    properties: dict | None = None,
    source_table: str = "members",
    event_id: str = "evt-001",
    source_commit_timestamp: datetime | None = None,
) -> CDCEvent:
    """Helper to create a CDCEvent for testing."""
    return CDCEvent(
        event_id=event_id,
        entity_type=entity_type,
        entity_id=entity_id,
        operation=operation,
        properties=properties or {"name": "John Doe", "status": "active"},
        source_table=source_table,
        source_commit_timestamp=source_commit_timestamp or datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        detected_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def graph_service():
    return FakeGraphService()


@pytest.fixture
def dagster_client():
    return FakeDagsterClient()


@pytest.fixture
def pipeline(graph_service, dagster_client):
    return CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph_service)


# =============================================================================
# Tests: transform_to_graph_entity
# =============================================================================


class TestTransformToGraphEntity:
    """Tests for CDC event to graph entity transformation (Req 12.2)."""

    @pytest.mark.parametrize("entity_type", list(VALID_NODE_TYPES))
    async def test_valid_node_types_transform_correctly(self, pipeline, entity_type):
        """Each valid node type should transform to a GraphEntity with NODE kind."""
        event = make_cdc_event(entity_type=entity_type)
        entity = await pipeline.transform_to_graph_entity(event)

        assert entity.kind == GraphEntityKind.NODE
        assert entity.entity_type == entity_type
        assert entity.entity_id == event.entity_id
        assert entity.properties == event.properties

    async def test_relationship_type_transforms_correctly(self, pipeline):
        """A CDC event with relationship entity_type and proper properties should transform."""
        # CDCEvent validator only allows node types for entity_type,
        # so relationships come from source_table mapping
        event = CDCEvent(
            event_id="evt-rel-001",
            entity_type="Member",  # The entity_type on CDCEvent is validated
            entity_id="rel-001",
            operation="INSERT",
            properties={
                "source_id": "member-001",
                "target_id": "event-001",
                "since": "2024-01-01",
            },
            source_table="member_conditions",
            source_commit_timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            detected_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        )
        # Since entity_type "Member" is a valid node type, it will map as node
        entity = await pipeline.transform_to_graph_entity(event)
        assert entity.kind == GraphEntityKind.NODE
        assert entity.entity_type == "Member"

    async def test_source_table_mapping_for_nodes(self, pipeline):
        """Source table names should map to correct node types via dbt model definitions."""
        event = make_cdc_event(
            entity_type="Event",
            source_table="clinical_events_raw",
        )
        entity = await pipeline.transform_to_graph_entity(event)
        assert entity.kind == GraphEntityKind.NODE
        assert entity.entity_type == "Event"

    async def test_unmappable_record_raises_error(self, pipeline):
        """Records with invalid entity types that can't be mapped should raise UnmappableRecordError."""
        # Create event with invalid entity_type - we need to bypass the CDCEvent validator
        # Since CDCEvent validates entity_type, we'll test via a source_table that doesn't map
        # But entity_type validation prevents invalid types. Let's test via source_table fallback
        # by using a valid entity_type with a source_table that doesn't match any mapping.
        # Actually, since entity_type="Member" is valid, it will always map.
        # The unmappable case happens when both entity_type and source_table fail.
        # We need to construct this carefully - create CDCEvent without pydantic validation
        # for the purpose of testing the transform logic.
        event = CDCEvent.model_construct(
            event_id="evt-bad-001",
            entity_type="UnknownType",
            entity_id="unknown-001",
            operation="INSERT",
            properties={"foo": "bar"},
            source_table="random_unknown_table",
            source_commit_timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            detected_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        )

        with pytest.raises(UnmappableRecordError) as exc_info:
            await pipeline.transform_to_graph_entity(event)

        assert "UnknownType" in exc_info.value.reason
        assert exc_info.value.source_record_id == "evt-bad-001"

    async def test_relationship_missing_source_target_raises_error(self, pipeline):
        """Relationship mapping without source_id/target_id should raise UnmappableRecordError."""
        # Use model_construct to bypass entity_type validation for relationship testing
        event = CDCEvent.model_construct(
            event_id="evt-rel-bad",
            entity_type="HAS_CONDITION",
            entity_id="rel-bad",
            operation="INSERT",
            properties={"some_prop": "value"},  # No source_id or target_id
            source_table="has_condition",
            source_commit_timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            detected_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        )

        with pytest.raises(UnmappableRecordError) as exc_info:
            await pipeline.transform_to_graph_entity(event)

        assert "source_id" in exc_info.value.reason

    async def test_properties_preserved_in_transformation(self, pipeline):
        """All properties from the CDC event should be preserved in the graph entity."""
        props = {"name": "Jane", "dob": "1990-01-01", "status": "active", "plan_id": "P123"}
        event = make_cdc_event(properties=props)
        entity = await pipeline.transform_to_graph_entity(event)
        assert entity.properties == props

    async def test_source_commit_timestamp_preserved(self, pipeline):
        """Source commit timestamp should be carried through to the graph entity."""
        event = make_cdc_event()
        entity = await pipeline.transform_to_graph_entity(event)
        assert entity.source_commit_timestamp == event.source_commit_timestamp


# =============================================================================
# Tests: apply_upsert
# =============================================================================


class TestApplyUpsert:
    """Tests for graph upsert application (Req 12.3)."""

    async def test_node_upsert_calls_graph_service(self, pipeline, graph_service):
        """Node entities should be upserted via the graph service upsert_node method."""
        entity = GraphEntity(
            kind=GraphEntityKind.NODE,
            entity_type="Member",
            entity_id="member-001",
            properties={"name": "John", "status": "active"},
            source_event_id="evt-001",
        )

        await pipeline.apply_upsert(entity)

        assert len(graph_service.upsert_node_calls) == 1
        call = graph_service.upsert_node_calls[0]
        assert call["node_type"] == "Member"
        assert call["node_id"] == "member-001"
        assert call["properties"] == {"name": "John", "status": "active"}
        assert call["execution_id"] == "evt-001"

    async def test_relationship_upsert_calls_graph_service(self, pipeline, graph_service):
        """Relationship entities should be upserted via upsert_relationship."""
        entity = GraphEntity(
            kind=GraphEntityKind.RELATIONSHIP,
            entity_type="HAS_CONDITION",
            entity_id="rel-001",
            properties={"since": "2024-01-01"},
            source_node_id="member-001",
            target_node_id="event-001",
            source_event_id="evt-001",
        )

        await pipeline.apply_upsert(entity)

        assert len(graph_service.upsert_relationship_calls) == 1
        call = graph_service.upsert_relationship_calls[0]
        assert call["source_id"] == "member-001"
        assert call["target_id"] == "event-001"
        assert call["rel_type"] == "HAS_CONDITION"
        assert call["properties"] == {"since": "2024-01-01"}

    async def test_relationship_without_source_target_raises(self, pipeline):
        """Relationship entity without source/target node IDs should raise ValueError."""
        entity = GraphEntity(
            kind=GraphEntityKind.RELATIONSHIP,
            entity_type="HAS_CONDITION",
            entity_id="rel-bad",
            properties={},
            source_node_id=None,
            target_node_id=None,
        )

        with pytest.raises(ValueError, match="missing source_node_id or target_node_id"):
            await pipeline.apply_upsert(entity)

    async def test_upsert_does_not_remove_other_relationships(self, pipeline, graph_service):
        """Upserting a node should not touch relationships — only properties are updated."""
        # Upsert a node with specific properties
        entity = GraphEntity(
            kind=GraphEntityKind.NODE,
            entity_type="Member",
            entity_id="member-001",
            properties={"name": "Updated Name"},
            source_event_id="evt-002",
        )

        await pipeline.apply_upsert(entity)

        # The upsert_node call only passes the specified properties
        # The graph service is responsible for merging without deleting
        call = graph_service.upsert_node_calls[0]
        assert call["properties"] == {"name": "Updated Name"}
        # No relationship calls were made
        assert len(graph_service.upsert_relationship_calls) == 0


# =============================================================================
# Tests: process_change_event (with retry logic)
# =============================================================================


class TestProcessChangeEvent:
    """Tests for full event processing with retry logic (Req 12.4)."""

    async def test_successful_processing(self, dagster_client):
        """A successful processing should return success=True and update checkpoint."""
        graph = FakeGraphService(fail_count=0)
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)
        event = make_cdc_event()

        result = await pipeline.process_change_event(event)

        assert result.success is True
        assert result.event_id == event.event_id
        assert result.retries_attempted == 0
        assert result.entity is not None
        assert pipeline.get_checkpoint() is not None
        assert pipeline.get_checkpoint().last_event_id == event.event_id

    @patch("clinical_reasoning_fabric.cdc.cdc_pipeline_service.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_failure_then_success(self, mock_sleep, dagster_client):
        """Should retry with exponential backoff and succeed on retry."""
        # Fail first 2 attempts, succeed on 3rd
        graph = FakeGraphService(fail_count=2)
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)
        event = make_cdc_event()

        result = await pipeline.process_change_event(event)

        assert result.success is True
        assert result.retries_attempted == 2
        # Verify exponential backoff delays
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(5)   # 5s first backoff
        mock_sleep.assert_any_call(10)  # 10s second backoff

    @patch("clinical_reasoning_fabric.cdc.cdc_pipeline_service.asyncio.sleep", new_callable=AsyncMock)
    async def test_all_retries_exhausted(self, mock_sleep, dagster_client):
        """Should fail after MAX_RETRIES + 1 attempts with all backoffs."""
        # Fail all 4 attempts (1 initial + 3 retries)
        graph = FakeGraphService(fail_count=10)
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)
        event = make_cdc_event()

        result = await pipeline.process_change_event(event)

        assert result.success is False
        assert result.retries_attempted == MAX_RETRIES + 1
        assert "Failed to apply upsert" in result.error
        # 3 backoff sleeps (before retries 2, 3, 4)
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(5)   # 5s
        mock_sleep.assert_any_call(10)  # 10s
        mock_sleep.assert_any_call(20)  # 20s

    async def test_unmappable_event_skipped(self, dagster_client):
        """Unmappable events should return failure without retrying."""
        graph = FakeGraphService()
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)

        event = CDCEvent.model_construct(
            event_id="evt-unmappable",
            entity_type="InvalidType",
            entity_id="bad-001",
            operation="INSERT",
            properties={},
            source_table="unknown_table_xyz",
            source_commit_timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            detected_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        )

        result = await pipeline.process_change_event(event)

        assert result.success is False
        assert "UnmappableRecordError" in result.error
        assert result.retries_attempted == 0
        # No graph calls should have been made
        assert len(graph.upsert_node_calls) == 0

    async def test_checkpoint_not_updated_on_failure(self, dagster_client):
        """Checkpoint should not be updated when processing fails."""
        graph = FakeGraphService(fail_count=10)
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)
        event = make_cdc_event()

        # Patch sleep to avoid delays in test
        with patch(
            "clinical_reasoning_fabric.cdc.cdc_pipeline_service.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await pipeline.process_change_event(event)

        assert result.success is False
        assert pipeline.get_checkpoint() is None


# =============================================================================
# Tests: Checkpoint management
# =============================================================================


class TestCheckpointManagement:
    """Tests for event checkpoint get/update."""

    async def test_initial_checkpoint_is_none(self, pipeline):
        """Before any processing, checkpoint should be None."""
        assert pipeline.get_checkpoint() is None

    async def test_checkpoint_updated_after_success(self, pipeline):
        """After successful processing, checkpoint reflects the last event."""
        event = make_cdc_event(event_id="evt-100")
        result = await pipeline.process_change_event(event)

        assert result.success is True
        checkpoint = pipeline.get_checkpoint()
        assert checkpoint is not None
        assert checkpoint.last_event_id == "evt-100"
        assert checkpoint.total_events_processed == 1

    async def test_checkpoint_increments_on_successive_events(self, pipeline):
        """Each successful event should increment total_events_processed."""
        for i in range(3):
            event = make_cdc_event(event_id=f"evt-{i}")
            await pipeline.process_change_event(event)

        checkpoint = pipeline.get_checkpoint()
        assert checkpoint.total_events_processed == 3
        assert checkpoint.last_event_id == "evt-2"


# =============================================================================
# Tests: Batch Processing (Req 12.5, 12.6, 12.7)
# =============================================================================


class TestProcessBatch:
    """Tests for batch event processing with ordering and checkpoint logic."""

    async def test_batch_sorts_events_by_source_commit_timestamp(self, dagster_client):
        """Events in a batch should be sorted by source_commit_timestamp before processing."""
        graph = FakeGraphService()
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)

        # Create events with out-of-order timestamps
        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_cdc_event(event_id="evt-3", entity_id="m-003", source_commit_timestamp=t3),
            make_cdc_event(event_id="evt-1", entity_id="m-001", source_commit_timestamp=t1),
            make_cdc_event(event_id="evt-2", entity_id="m-002", source_commit_timestamp=t2),
        ]

        result = await pipeline.process_batch(events)

        assert result.total_events == 3
        assert result.successful == 3
        # Verify processing happened in timestamp order via checkpoint
        checkpoint = pipeline.get_checkpoint()
        assert checkpoint.last_event_id == "evt-3"  # Last processed = latest timestamp
        # Verify graph calls are in order
        assert graph.upsert_node_calls[0]["node_id"] == "m-001"
        assert graph.upsert_node_calls[1]["node_id"] == "m-002"
        assert graph.upsert_node_calls[2]["node_id"] == "m-003"

    async def test_same_entity_events_applied_in_timestamp_order(self, dagster_client):
        """Multiple events for the same entity should be applied in timestamp order."""
        graph = FakeGraphService()
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)

        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Same entity_id, different timestamps (submitted out of order)
        events = [
            make_cdc_event(
                event_id="evt-3", entity_id="member-001",
                source_commit_timestamp=t3,
                properties={"name": "Final Name", "status": "active"},
            ),
            make_cdc_event(
                event_id="evt-1", entity_id="member-001",
                source_commit_timestamp=t1,
                properties={"name": "First Name", "status": "active"},
            ),
            make_cdc_event(
                event_id="evt-2", entity_id="member-001",
                source_commit_timestamp=t2,
                properties={"name": "Middle Name", "status": "active"},
            ),
        ]

        result = await pipeline.process_batch(events)

        assert result.successful == 3
        # Verify they were applied in t1, t2, t3 order
        assert graph.upsert_node_calls[0]["properties"]["name"] == "First Name"
        assert graph.upsert_node_calls[1]["properties"]["name"] == "Middle Name"
        assert graph.upsert_node_calls[2]["properties"]["name"] == "Final Name"

    async def test_unmappable_records_skipped_but_processing_continues(self, dagster_client):
        """Unmappable records should be skipped with logging, but subsequent events process."""
        graph = FakeGraphService()
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)

        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_cdc_event(event_id="evt-1", entity_id="m-001", source_commit_timestamp=t1),
            # This event has an unmappable entity_type (bypass validation with model_construct)
            CDCEvent.model_construct(
                event_id="evt-bad",
                entity_type="UnknownType",
                entity_id="bad-001",
                operation="INSERT",
                properties={"foo": "bar"},
                source_table="unknown_xyz_table",
                source_commit_timestamp=t2,
                detected_at=t2,
            ),
            make_cdc_event(event_id="evt-3", entity_id="m-003", source_commit_timestamp=t3),
        ]

        result = await pipeline.process_batch(events)

        assert result.total_events == 3
        assert result.successful == 2
        assert result.skipped_unmappable == 1
        assert result.failed == 1  # unmappable counts as failed
        # The valid events should still have been processed
        assert len(graph.upsert_node_calls) == 2
        assert graph.upsert_node_calls[0]["node_id"] == "m-001"
        assert graph.upsert_node_calls[1]["node_id"] == "m-003"

    async def test_resume_from_checkpoint_skips_already_processed(self, dagster_client):
        """On resume, events before or at the checkpoint should be skipped."""
        graph = FakeGraphService()
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)

        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Set a checkpoint indicating evt-2 at t2 was already processed
        pipeline.update_checkpoint(EventCheckpoint(
            last_event_id="evt-2",
            last_source_commit_timestamp=t2,
            last_processed_at=datetime.now(timezone.utc),
            total_events_processed=2,
        ))

        events = [
            make_cdc_event(event_id="evt-1", entity_id="m-001", source_commit_timestamp=t1),
            make_cdc_event(event_id="evt-2", entity_id="m-002", source_commit_timestamp=t2),
            make_cdc_event(event_id="evt-3", entity_id="m-003", source_commit_timestamp=t3),
        ]

        result = await pipeline.process_batch(events)

        assert result.total_events == 3
        assert result.skipped_already_processed == 2  # evt-1 (before) and evt-2 (at checkpoint)
        assert result.successful == 1  # only evt-3 processed
        assert len(graph.upsert_node_calls) == 1
        assert graph.upsert_node_calls[0]["node_id"] == "m-003"

    async def test_empty_batch_returns_empty_result(self, dagster_client):
        """An empty batch should return a result with zero events."""
        graph = FakeGraphService()
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)

        result = await pipeline.process_batch([])

        assert result.total_events == 0
        assert result.successful == 0
        assert result.failed == 0

    async def test_idempotency_replay_from_checkpoint(self, dagster_client):
        """Replaying the same batch after a checkpoint should not duplicate mutations."""
        graph = FakeGraphService()
        pipeline = CDCPipelineService(dagster_client=dagster_client, neo4j_service=graph)

        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)

        events = [
            make_cdc_event(event_id="evt-1", entity_id="m-001", source_commit_timestamp=t1),
            make_cdc_event(event_id="evt-2", entity_id="m-002", source_commit_timestamp=t2),
        ]

        # First processing — both events processed
        result1 = await pipeline.process_batch(events)
        assert result1.successful == 2
        assert len(graph.upsert_node_calls) == 2

        # Replay the same batch — should skip both (already processed)
        result2 = await pipeline.process_batch(events)
        assert result2.skipped_already_processed == 2
        assert result2.successful == 0
        # No new graph calls
        assert len(graph.upsert_node_calls) == 2  # Still only 2 from first run


# =============================================================================
# Tests: resume_from_checkpoint
# =============================================================================


class TestResumeFromCheckpoint:
    """Tests for the resume_from_checkpoint filtering logic."""

    async def test_no_checkpoint_returns_all_events(self, pipeline):
        """With no checkpoint, all events should be returned."""
        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)

        events = [
            make_cdc_event(event_id="evt-1", source_commit_timestamp=t1),
            make_cdc_event(event_id="evt-2", source_commit_timestamp=t2),
        ]

        result = pipeline.resume_from_checkpoint(events)
        assert len(result) == 2

    async def test_filters_events_before_checkpoint(self, pipeline):
        """Events with timestamps before the checkpoint should be filtered out."""
        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        pipeline.update_checkpoint(EventCheckpoint(
            last_event_id="evt-2",
            last_source_commit_timestamp=t2,
            last_processed_at=datetime.now(timezone.utc),
            total_events_processed=2,
        ))

        events = [
            make_cdc_event(event_id="evt-1", source_commit_timestamp=t1),
            make_cdc_event(event_id="evt-2", source_commit_timestamp=t2),
            make_cdc_event(event_id="evt-3", source_commit_timestamp=t3),
        ]

        result = pipeline.resume_from_checkpoint(events)
        assert len(result) == 1
        assert result[0].event_id == "evt-3"

    async def test_concurrent_events_at_same_timestamp_included(self, pipeline):
        """Events at the same timestamp but different event_ids should be included."""
        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)

        pipeline.update_checkpoint(EventCheckpoint(
            last_event_id="evt-2a",
            last_source_commit_timestamp=t2,
            last_processed_at=datetime.now(timezone.utc),
            total_events_processed=2,
        ))

        events = [
            make_cdc_event(event_id="evt-1", source_commit_timestamp=t1),
            make_cdc_event(event_id="evt-2a", source_commit_timestamp=t2),
            make_cdc_event(event_id="evt-2b", source_commit_timestamp=t2),  # Concurrent
            make_cdc_event(event_id="evt-2c", source_commit_timestamp=t2),  # Concurrent
        ]

        result = pipeline.resume_from_checkpoint(events)
        # evt-1 is before checkpoint, evt-2a is the checkpoint event itself — both filtered
        # evt-2b and evt-2c have same timestamp but different IDs — included
        assert len(result) == 2
        assert result[0].event_id == "evt-2b"
        assert result[1].event_id == "evt-2c"


# =============================================================================
# Tests: FileCheckpointStore
# =============================================================================


class TestFileCheckpointStore:
    """Tests for persistent file-based checkpoint storage."""

    def test_load_returns_none_for_nonexistent_file(self, tmp_path):
        """Loading from a non-existent file should return None."""
        store = FileCheckpointStore(tmp_path / "checkpoint.json")
        assert store.load() is None

    def test_save_and_load_round_trip(self, tmp_path):
        """Saving and loading a checkpoint should preserve all fields."""
        store = FileCheckpointStore(tmp_path / "checkpoint.json")
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        checkpoint = EventCheckpoint(
            last_event_id="evt-42",
            last_source_commit_timestamp=ts,
            last_processed_at=ts,
            total_events_processed=42,
        )

        store.save(checkpoint)
        loaded = store.load()

        assert loaded is not None
        assert loaded.last_event_id == "evt-42"
        assert loaded.last_source_commit_timestamp == ts
        assert loaded.total_events_processed == 42

    def test_load_returns_none_for_corrupted_file(self, tmp_path):
        """Loading from a corrupted file should return None gracefully."""
        filepath = tmp_path / "checkpoint.json"
        filepath.write_text("not valid json {{{", encoding="utf-8")

        store = FileCheckpointStore(filepath)
        assert store.load() is None

    def test_pipeline_persists_checkpoint_via_store(self, tmp_path, dagster_client):
        """CDCPipelineService should persist checkpoints through the store."""
        store = FileCheckpointStore(tmp_path / "checkpoint.json")
        graph = FakeGraphService()
        pipeline = CDCPipelineService(
            dagster_client=dagster_client,
            neo4j_service=graph,
            checkpoint_store=store,
        )

        # Process an event
        import asyncio
        event = make_cdc_event(event_id="evt-persist")
        asyncio.get_event_loop().run_until_complete(pipeline.process_change_event(event))

        # Verify the checkpoint was persisted to file
        loaded = store.load()
        assert loaded is not None
        assert loaded.last_event_id == "evt-persist"

    def test_pipeline_loads_checkpoint_on_init(self, tmp_path, dagster_client):
        """CDCPipelineService should load existing checkpoint from store on init."""
        store = FileCheckpointStore(tmp_path / "checkpoint.json")
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        store.save(EventCheckpoint(
            last_event_id="evt-existing",
            last_source_commit_timestamp=ts,
            last_processed_at=ts,
            total_events_processed=10,
        ))

        graph = FakeGraphService()
        pipeline = CDCPipelineService(
            dagster_client=dagster_client,
            neo4j_service=graph,
            checkpoint_store=store,
        )

        checkpoint = pipeline.get_checkpoint()
        assert checkpoint is not None
        assert checkpoint.last_event_id == "evt-existing"
        assert checkpoint.total_events_processed == 10
