"""
Property tests for the CDC Pipeline Service.

**Validates: Requirements 12.2, 12.3, 12.5, 12.7**

Property 18: CDC Transformation Type Correctness — For any valid source record with
recognized entity_type, transformation produces correct node type with required fields.

Property 19: CDC Upsert Relationship Preservation — Upsert updates only referenced
properties/relationships, leaving others unchanged.

Property 20: CDC Checkpoint Idempotency — Replay from checkpoint does not duplicate
nodes/relationships.

Property 21: CDC Temporal Ordering — Events applied in source-commit-timestamp order
yield latest state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from clinical_reasoning_fabric.cdc.cdc_pipeline_service import (
    CDCPipelineService,
    GraphEntity,
    GraphEntityKind,
    GraphServiceProtocol,
    InMemoryCheckpointStore,
    VALID_NODE_TYPES,
    VALID_RELATIONSHIP_TYPES,
)
from clinical_reasoning_fabric.models.core import CDCEvent, EventCheckpoint


# =============================================================================
# Test Helpers — Fake Graph Service for tracking calls
# =============================================================================


class TrackingGraphService:
    """A graph service that records all upsert calls for verification."""

    def __init__(self) -> None:
        self.node_calls: list[dict[str, Any]] = []
        self.relationship_calls: list[dict[str, Any]] = []

    async def upsert_node(
        self, node_type: str, node_id: str, properties: dict, execution_id: str | None = None
    ) -> None:
        self.node_calls.append({
            "node_type": node_type,
            "node_id": node_id,
            "properties": properties,
            "execution_id": execution_id,
        })

    async def upsert_relationship(
        self, source_id: str, target_id: str, rel_type: str, properties: dict
    ) -> None:
        self.relationship_calls.append({
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
            "properties": properties,
        })

    @property
    def total_calls(self) -> int:
        return len(self.node_calls) + len(self.relationship_calls)


class FakeDagsterClient:
    """Minimal Dagster client stub for testing."""

    async def trigger_run(self, job_name: str, run_config: dict) -> str:
        return "run-001"

    async def get_run_status(self, run_id: str) -> str:
        return "SUCCESS"


# =============================================================================
# Strategies
# =============================================================================

VALID_NODE_TYPE_LIST = list(VALID_NODE_TYPES)
VALID_OPERATIONS = ["INSERT", "UPDATE"]

# Strategy for a valid entity type that maps to a node
node_type_strategy = st.sampled_from(VALID_NODE_TYPE_LIST)

# Strategy for valid operations
operation_strategy = st.sampled_from(VALID_OPERATIONS)

# Strategy for timestamp within a reasonable range
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)

# Strategy for generating random properties (simple key-value pairs)
property_key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)
property_value_strategy = st.one_of(
    st.text(min_size=0, max_size=50),
    st.integers(min_value=-10000, max_value=10000),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False),
)
properties_strategy = st.dictionaries(
    keys=property_key_strategy,
    values=property_value_strategy,
    min_size=0,
    max_size=5,
)


@st.composite
def valid_cdc_node_event(draw, entity_type=None, entity_id=None, timestamp=None):
    """Generate a valid CDC event that maps to a graph node."""
    etype = entity_type or draw(node_type_strategy)
    eid = entity_id or draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
        min_size=1,
        max_size=30,
    ))
    ts = timestamp or draw(timestamp_strategy)
    props = draw(properties_strategy)
    event_id = draw(st.uuids().map(str))
    op = draw(operation_strategy)

    return CDCEvent(
        event_id=event_id,
        entity_type=etype,
        entity_id=eid,
        operation=op,
        properties=props,
        source_table=f"{etype.lower()}_table",
        source_commit_timestamp=ts,
        detected_at=ts + timedelta(seconds=draw(st.integers(min_value=1, max_value=300))),
    )


@st.composite
def valid_cdc_events_batch(draw, min_size=2, max_size=10):
    """Generate a batch of valid CDC node events with unique timestamps.

    Uses unique timestamps so checkpoint filtering works deterministically—
    the checkpoint only stores the last event_id/timestamp, so events at the
    same timestamp with different event_ids would pass through (by design).
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    base_ts = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2029, 12, 31),
        timezones=st.just(timezone.utc),
    ))
    # Generate distinct offsets for unique timestamps
    offsets = draw(
        st.lists(
            st.integers(min_value=1, max_value=86400 * 365),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    events = []
    for offset in offsets:
        ts = base_ts + timedelta(seconds=offset)
        event = draw(valid_cdc_node_event(timestamp=ts))
        events.append(event)
    return events


@st.composite
def same_entity_events_with_different_timestamps(draw, min_events=2, max_events=5):
    """Generate multiple CDC events for the same entity at different timestamps."""
    entity_type = draw(node_type_strategy)
    entity_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
        min_size=1,
        max_size=20,
    ))

    n = draw(st.integers(min_value=min_events, max_value=max_events))

    # Generate distinct timestamps
    base_ts = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2029, 12, 31),
        timezones=st.just(timezone.utc),
    ))
    offsets = draw(
        st.lists(
            st.integers(min_value=1, max_value=86400),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    timestamps = [base_ts + timedelta(seconds=offset) for offset in offsets]

    events = []
    for i, ts in enumerate(timestamps):
        props = draw(properties_strategy)
        event_id = draw(st.uuids().map(str))
        events.append(CDCEvent(
            event_id=event_id,
            entity_type=entity_type,
            entity_id=entity_id,
            operation="UPDATE",
            properties=props,
            source_table=f"{entity_type.lower()}_table",
            source_commit_timestamp=ts,
            detected_at=ts + timedelta(seconds=5),
        ))

    return events


# =============================================================================
# Property Tests
# =============================================================================


@pytest.mark.property
class TestCDCTransformationTypeCorrectness:
    """Property 18: CDC Transformation Type Correctness.

    **Validates: Requirements 12.2**

    For any valid source record with recognized entity_type, transformation
    produces correct node type with required fields.
    """

    @given(event=valid_cdc_node_event())
    def test_transformation_produces_correct_node_type(self, event: CDCEvent):
        """For any valid CDC event with entity_type in VALID_NODE_TYPES,
        transform_to_graph_entity produces a GraphEntity with kind=NODE
        and matching entity_type."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        entity = asyncio.get_event_loop().run_until_complete(
            pipeline.transform_to_graph_entity(event)
        )

        # The entity_type should match the input
        assert entity.entity_type == event.entity_type, (
            f"Expected entity_type='{event.entity_type}', got '{entity.entity_type}'"
        )
        # Kind should be NODE for valid node types
        assert entity.kind == GraphEntityKind.NODE, (
            f"Expected kind=NODE for entity_type='{event.entity_type}', got '{entity.kind}'"
        )
        # Entity ID should be preserved
        assert entity.entity_id == event.entity_id, (
            f"Expected entity_id='{event.entity_id}', got '{entity.entity_id}'"
        )
        # Source event ID should be preserved
        assert entity.source_event_id == event.event_id, (
            f"Source event_id not preserved in GraphEntity"
        )
        # Source commit timestamp should be preserved
        assert entity.source_commit_timestamp == event.source_commit_timestamp, (
            f"Source commit timestamp not preserved in GraphEntity"
        )

    @given(event=valid_cdc_node_event())
    def test_transformation_preserves_all_properties(self, event: CDCEvent):
        """All properties from the source event appear in the transformed entity."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        entity = asyncio.get_event_loop().run_until_complete(
            pipeline.transform_to_graph_entity(event)
        )

        # All event properties should be in entity properties
        for key, value in event.properties.items():
            assert key in entity.properties, (
                f"Property '{key}' from event not found in transformed entity"
            )
            assert entity.properties[key] == value, (
                f"Property '{key}' value mismatch: expected '{value}', got '{entity.properties[key]}'"
            )


@pytest.mark.property
class TestCDCUpsertRelationshipPreservation:
    """Property 19: CDC Upsert Relationship Preservation.

    **Validates: Requirements 12.3**

    Upsert updates only referenced properties/relationships, leaving others unchanged.
    """

    @given(event=valid_cdc_node_event())
    def test_upsert_only_sends_event_properties(self, event: CDCEvent):
        """Verify that apply_upsert only sends the properties from the event
        to the graph service (no extra deletions or additions)."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        entity = asyncio.get_event_loop().run_until_complete(
            pipeline.transform_to_graph_entity(event)
        )
        asyncio.get_event_loop().run_until_complete(
            pipeline.apply_upsert(entity)
        )

        # Exactly one node call should have been made
        assert len(graph_service.node_calls) == 1, (
            f"Expected exactly 1 node upsert call, got {len(graph_service.node_calls)}"
        )
        # No relationship calls should have been made for a node entity
        assert len(graph_service.relationship_calls) == 0, (
            f"Expected 0 relationship calls for a node upsert, got {len(graph_service.relationship_calls)}"
        )

        call = graph_service.node_calls[0]
        # The properties sent to the graph service should match exactly
        assert call["properties"] == entity.properties, (
            f"Properties sent to graph service don't match entity properties. "
            f"Sent: {call['properties']}, Expected: {entity.properties}"
        )
        # Node type should match
        assert call["node_type"] == entity.entity_type
        # Node ID should match
        assert call["node_id"] == entity.entity_id

    @given(
        events=st.lists(valid_cdc_node_event(), min_size=2, max_size=5)
    )
    def test_multiple_upserts_do_not_interfere(self, events: list[CDCEvent]):
        """Multiple upserts to different entities each only affect their own properties."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        for event in events:
            entity = asyncio.get_event_loop().run_until_complete(
                pipeline.transform_to_graph_entity(event)
            )
            asyncio.get_event_loop().run_until_complete(
                pipeline.apply_upsert(entity)
            )

        # Each event should have produced exactly one graph call
        assert graph_service.total_calls == len(events), (
            f"Expected {len(events)} total graph calls, got {graph_service.total_calls}"
        )

        # Each call's properties should match only its own event's properties
        for i, (event, call) in enumerate(zip(events, graph_service.node_calls)):
            assert call["properties"] == event.properties, (
                f"Call {i}: properties sent to graph don't match event properties"
            )


@pytest.mark.property
class TestCDCCheckpointIdempotency:
    """Property 20: CDC Checkpoint Idempotency.

    **Validates: Requirements 12.5**

    Replay from checkpoint does not duplicate nodes/relationships.
    """

    @given(events=valid_cdc_events_batch(min_size=2, max_size=8))
    def test_replay_from_checkpoint_produces_zero_new_calls(self, events: list[CDCEvent]):
        """Process a batch, then replay the same batch — second run should produce
        0 new graph calls because checkpoint filtering skips already-processed events."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        # First run: process all events
        result1 = asyncio.get_event_loop().run_until_complete(
            pipeline.process_batch(events)
        )
        calls_after_first_run = graph_service.total_calls

        # Record how many events were actually processed (some may be unmappable)
        assert result1.successful > 0, "Expected at least some events to succeed"

        # Second run: replay the exact same batch
        result2 = asyncio.get_event_loop().run_until_complete(
            pipeline.process_batch(events)
        )
        calls_after_second_run = graph_service.total_calls

        # No new graph calls should have been made
        new_calls = calls_after_second_run - calls_after_first_run
        assert new_calls == 0, (
            f"Replay produced {new_calls} new graph calls, expected 0. "
            f"Checkpoint should have filtered all events. "
            f"First run: {result1.successful} successful, "
            f"Second run: {result2.successful} successful, "
            f"skipped_already_processed: {result2.skipped_already_processed}"
        )

        # The second batch should report all events as already processed
        assert result2.skipped_already_processed == len(events), (
            f"Expected {len(events)} events skipped as already processed, "
            f"got {result2.skipped_already_processed}"
        )

    @given(events=valid_cdc_events_batch(min_size=3, max_size=8))
    def test_partial_replay_does_not_duplicate(self, events: list[CDCEvent]):
        """Process a batch partially (simulate restart after some events),
        then replay the full batch — events processed before checkpoint are skipped."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        # Sort events as process_batch would
        sorted_events = sorted(events, key=lambda e: e.source_commit_timestamp)

        # Process first half manually
        half = len(sorted_events) // 2
        assume(half >= 1)
        first_half = sorted_events[:half]

        for event in first_half:
            asyncio.get_event_loop().run_until_complete(
                pipeline.process_change_event(event)
            )

        calls_after_partial = graph_service.total_calls

        # Now process the full batch — already processed events should be skipped
        result = asyncio.get_event_loop().run_until_complete(
            pipeline.process_batch(events)
        )

        # The events from the first half should have been skipped
        # (they have timestamps <= the checkpoint's last_source_commit_timestamp)
        # Only events after the checkpoint should produce new calls
        remaining_events = len(sorted_events) - half
        new_calls = graph_service.total_calls - calls_after_partial
        assert new_calls <= remaining_events, (
            f"Replay after partial processing produced {new_calls} new calls, "
            f"expected at most {remaining_events} (total {len(sorted_events)} - {half} processed)"
        )
        # At minimum, the second half should NOT include events from before checkpoint
        assert result.skipped_already_processed >= half, (
            f"Expected at least {half} events skipped as already processed, "
            f"got {result.skipped_already_processed}"
        )


@pytest.mark.property
class TestCDCTemporalOrdering:
    """Property 21: CDC Temporal Ordering.

    **Validates: Requirements 12.7**

    Events applied in source-commit-timestamp order yield latest state.
    """

    @given(events=same_entity_events_with_different_timestamps(min_events=2, max_events=5))
    def test_events_applied_in_timestamp_order(self, events: list[CDCEvent]):
        """Verify that process_batch applies events for the same entity in
        source-commit-timestamp order regardless of their input order."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        # Shuffle events to ensure input order differs from timestamp order
        # (Hypothesis generates them in varying orders already, but let's verify)
        import random
        shuffled = list(events)
        random.shuffle(shuffled)

        # Process the shuffled batch
        asyncio.get_event_loop().run_until_complete(
            pipeline.process_batch(shuffled)
        )

        # Verify the node calls were made in timestamp order
        assert len(graph_service.node_calls) == len(events), (
            f"Expected {len(events)} node calls, got {len(graph_service.node_calls)}"
        )

        # The events should have been sorted and applied in timestamp order
        expected_order = sorted(events, key=lambda e: e.source_commit_timestamp)

        for i, (call, expected_event) in enumerate(zip(graph_service.node_calls, expected_order)):
            assert call["properties"] == expected_event.properties, (
                f"Call {i}: properties don't match timestamp-sorted event. "
                f"Expected properties from event at {expected_event.source_commit_timestamp}, "
                f"got properties from a different event."
            )

    @given(events=same_entity_events_with_different_timestamps(min_events=2, max_events=5))
    def test_latest_state_reflects_most_recent_event(self, events: list[CDCEvent]):
        """The last graph call for an entity should correspond to the event with
        the latest source_commit_timestamp, ensuring the graph reflects latest state."""
        graph_service = TrackingGraphService()
        dagster_client = FakeDagsterClient()
        pipeline = CDCPipelineService(dagster_client, graph_service)

        asyncio.get_event_loop().run_until_complete(
            pipeline.process_batch(events)
        )

        # The latest event by timestamp
        latest_event = max(events, key=lambda e: e.source_commit_timestamp)

        # The last node call should have the latest event's properties
        last_call = graph_service.node_calls[-1]
        assert last_call["properties"] == latest_event.properties, (
            f"Last graph call doesn't reflect the latest event. "
            f"Expected properties from event at {latest_event.source_commit_timestamp}"
        )
        assert last_call["node_id"] == latest_event.entity_id, (
            f"Last graph call node_id doesn't match the latest event's entity_id"
        )
