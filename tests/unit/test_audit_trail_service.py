"""Unit tests for AuditTrailService (BEACON Layer 6).

Tests immutable append-only trace recording, monotonically increasing sequence
numbers, UTC ISO-8601 timestamps, retrieval by request_id, immutability enforcement,
and error handling.

Validates:
    - record_entry() assigns monotonically increasing sequence numbers
    - record_entry() generates UTC ISO-8601 timestamps with ms precision
    - record_entry() includes request_id, identity_id, and valid category
    - Append-only storage prohibits modification and deletion
    - get_trace() retrieves complete trace by request_id
    - get_trace() returns entries ordered by sequence_number
    - On recording failure: raises TraceRecordingError (halts PA processing)
    - 7-year retention policy is configured

Requirements referenced: 8.1, 8.2, 8.4, 8.5, 8.6
"""

import asyncio
import re
from unittest.mock import AsyncMock, patch

import pytest

from clinical_reasoning_fabric.beacon.audit_trail_service import (
    RETENTION_POLICY_DAYS,
    RETENTION_POLICY_YEARS,
    GET_TRACE_TIMEOUT_SECONDS,
    AppendOnlyStorage,
    AuditTrailService,
    ImmutableStorageViolationError,
    InMemoryAppendOnlyStorage,
)
from clinical_reasoning_fabric.models.core import TraceCategory, TraceEntry
from clinical_reasoning_fabric.models.exceptions import TraceRecordingError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage():
    """Fresh in-memory append-only storage instance."""
    return InMemoryAppendOnlyStorage()


@pytest.fixture
def audit_service(storage):
    """AuditTrailService configured with in-memory storage."""
    return AuditTrailService(storage_backend=storage)


# =============================================================================
# Retention Policy Tests
# =============================================================================


class TestRetentionPolicy:
    """Tests for the 7-year retention policy configuration (Requirement 8.4)."""

    def test_retention_policy_is_seven_years(self):
        """Retention policy constant is set to 7 years."""
        assert RETENTION_POLICY_YEARS == 7

    def test_retention_policy_days_calculation(self):
        """Retention policy days is 7 * 365 = 2555."""
        assert RETENTION_POLICY_DAYS == 7 * 365

    def test_storage_exposes_retention_policy(self, storage):
        """Storage backend exposes the configured retention policy."""
        assert storage.retention_policy_days == RETENTION_POLICY_DAYS


# =============================================================================
# Record Entry Tests — Sequence Numbers
# =============================================================================


class TestRecordEntrySequenceNumbers:
    """Tests for monotonically increasing sequence number assignment (Requirement 8.1)."""

    async def test_first_entry_gets_sequence_1(self, audit_service):
        """First recorded entry is assigned sequence_number 1."""
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        assert entry.sequence_number == 1

    async def test_sequence_numbers_increase_monotonically(self, audit_service):
        """Successive entries get strictly increasing sequence numbers."""
        entries = []
        for i in range(5):
            entry = await audit_service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=TraceCategory.AGENT_ACTION,
            )
            entries.append(entry)

        for i in range(1, len(entries)):
            assert entries[i].sequence_number > entries[i - 1].sequence_number

    async def test_sequence_numbers_are_globally_unique(self, audit_service):
        """Sequence numbers are unique across different request_ids."""
        entry1 = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        entry2 = await audit_service.record_entry(
            request_id="req-002",
            identity_id="agent-002",
            category=TraceCategory.TOOL_INVOCATION,
        )
        assert entry1.sequence_number != entry2.sequence_number
        assert entry2.sequence_number > entry1.sequence_number

    async def test_sequence_numbers_never_repeat(self, audit_service):
        """No two entries share the same sequence number."""
        entries = []
        for i in range(20):
            entry = await audit_service.record_entry(
                request_id=f"req-{i % 3}",
                identity_id="agent-001",
                category=TraceCategory.DECISION_STEP,
            )
            entries.append(entry)

        seq_numbers = [e.sequence_number for e in entries]
        assert len(seq_numbers) == len(set(seq_numbers))


# =============================================================================
# Record Entry Tests — Timestamps
# =============================================================================


class TestRecordEntryTimestamps:
    """Tests for UTC ISO-8601 timestamps with ms precision (Requirement 8.2)."""

    async def test_timestamp_is_utc_iso8601_with_ms(self, audit_service):
        """Timestamp follows format: YYYY-MM-DDTHH:MM:SS.mmmZ."""
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        # Validate format: 2024-01-15T10:30:45.123Z
        iso8601_ms_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
        assert re.match(iso8601_ms_pattern, entry.timestamp), (
            f"Timestamp '{entry.timestamp}' does not match UTC ISO-8601 ms format"
        )

    async def test_timestamp_ends_with_z_for_utc(self, audit_service):
        """Timestamp ends with 'Z' indicating UTC."""
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.CONTEXT_RETRIEVAL,
        )
        assert entry.timestamp.endswith("Z")

    async def test_timestamp_has_exactly_three_ms_digits(self, audit_service):
        """Timestamp milliseconds portion has exactly 3 digits."""
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.TOOL_INVOCATION,
        )
        # Extract ms portion: everything between last '.' and 'Z'
        ms_part = entry.timestamp.split(".")[-1].rstrip("Z")
        assert len(ms_part) == 3
        assert ms_part.isdigit()

    async def test_timestamps_are_non_decreasing(self, audit_service):
        """Subsequent entry timestamps are >= prior timestamps."""
        entries = []
        for _ in range(5):
            entry = await audit_service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=TraceCategory.AGENT_ACTION,
            )
            entries.append(entry)

        for i in range(1, len(entries)):
            assert entries[i].timestamp >= entries[i - 1].timestamp


# =============================================================================
# Record Entry Tests — Required Fields
# =============================================================================


class TestRecordEntryFields:
    """Tests for required fields in trace entries (Requirements 8.1, 8.2)."""

    async def test_entry_contains_request_id(self, audit_service):
        """Recorded entry preserves the request_id."""
        entry = await audit_service.record_entry(
            request_id="pa-req-12345",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        assert entry.request_id == "pa-req-12345"

    async def test_entry_contains_identity_id(self, audit_service):
        """Recorded entry preserves the identity_id."""
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="clinical-agent-007",
            category=TraceCategory.TOOL_INVOCATION,
        )
        assert entry.identity_id == "clinical-agent-007"

    async def test_entry_contains_valid_category(self, audit_service):
        """Recorded entry has a valid TraceCategory."""
        for category in TraceCategory:
            entry = await audit_service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=category,
            )
            assert entry.category == category

    async def test_all_trace_categories_accepted(self, audit_service):
        """All four trace categories can be recorded."""
        categories = [
            TraceCategory.AGENT_ACTION,
            TraceCategory.TOOL_INVOCATION,
            TraceCategory.CONTEXT_RETRIEVAL,
            TraceCategory.DECISION_STEP,
        ]
        for cat in categories:
            entry = await audit_service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=cat,
            )
            assert entry.category == cat

    async def test_entry_preserves_details(self, audit_service):
        """Optional details dict is preserved in the entry."""
        details = {"tool_name": "retrieve_evidence", "duration_ms": 150}
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.TOOL_INVOCATION,
            details=details,
        )
        assert entry.details == details

    async def test_entry_details_defaults_to_none(self, audit_service):
        """Details is None when not provided."""
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        assert entry.details is None


# =============================================================================
# Immutability Tests
# =============================================================================


class TestImmutability:
    """Tests for append-only immutability enforcement (Requirement 8.4)."""

    async def test_delete_raises_violation_error(self, storage):
        """Attempting to delete an entry raises ImmutableStorageViolationError."""
        with pytest.raises(ImmutableStorageViolationError) as exc_info:
            await storage.delete(sequence_number=1)

        assert "delete" in str(exc_info.value)
        assert exc_info.value.operation == "delete"

    async def test_update_raises_violation_error(self, storage):
        """Attempting to update an entry raises ImmutableStorageViolationError."""
        fake_entry = TraceEntry(
            sequence_number=1,
            timestamp="2024-01-15T10:30:45.123Z",
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        with pytest.raises(ImmutableStorageViolationError) as exc_info:
            await storage.update(sequence_number=1, entry=fake_entry)

        assert "update" in str(exc_info.value)
        assert exc_info.value.operation == "update"

    async def test_append_is_allowed(self, storage):
        """Appending new entries is the only permitted write operation."""
        entry = TraceEntry(
            sequence_number=1,
            timestamp="2024-01-15T10:30:45.123Z",
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        # Should not raise
        await storage.append(entry)
        assert storage.entry_count == 1

    async def test_historical_entries_unchanged_after_new_append(self, audit_service, storage):
        """Appending new entries does not modify previously stored entries."""
        entry1 = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        original_seq = entry1.sequence_number
        original_timestamp = entry1.timestamp

        # Record more entries
        for _ in range(5):
            await audit_service.record_entry(
                request_id="req-002",
                identity_id="agent-002",
                category=TraceCategory.DECISION_STEP,
            )

        # Retrieve original entry and verify unchanged
        trace = await audit_service.get_trace("req-001")
        assert len(trace) == 1
        assert trace[0].sequence_number == original_seq
        assert trace[0].timestamp == original_timestamp
        assert trace[0].request_id == "req-001"


# =============================================================================
# Get Trace Tests
# =============================================================================


class TestGetTrace:
    """Tests for trace retrieval by request_id (Requirement 8.6)."""

    async def test_get_trace_returns_all_entries_for_request(self, audit_service):
        """get_trace returns all entries matching the request_id."""
        for i in range(5):
            await audit_service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=TraceCategory.AGENT_ACTION,
            )

        trace = await audit_service.get_trace("req-001")
        assert len(trace) == 5

    async def test_get_trace_returns_only_matching_request(self, audit_service):
        """get_trace filters to only the specified request_id."""
        await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        await audit_service.record_entry(
            request_id="req-002",
            identity_id="agent-002",
            category=TraceCategory.TOOL_INVOCATION,
        )
        await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.DECISION_STEP,
        )

        trace = await audit_service.get_trace("req-001")
        assert len(trace) == 2
        assert all(e.request_id == "req-001" for e in trace)

    async def test_get_trace_returns_entries_ordered_by_sequence(self, audit_service):
        """Entries are returned in sequence_number order (ascending)."""
        for _ in range(10):
            await audit_service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=TraceCategory.AGENT_ACTION,
            )

        trace = await audit_service.get_trace("req-001")
        for i in range(1, len(trace)):
            assert trace[i].sequence_number > trace[i - 1].sequence_number

    async def test_get_trace_returns_empty_for_unknown_request(self, audit_service):
        """get_trace returns empty list for non-existent request_id."""
        trace = await audit_service.get_trace("non-existent-request")
        assert trace == []

    async def test_get_trace_timeout_raises_error(self, audit_service):
        """get_trace raises TraceRecordingError on timeout."""

        async def slow_retrieval(request_id: str) -> list[TraceEntry]:
            await asyncio.sleep(35)
            return []

        audit_service._storage.get_entries_by_request_id = slow_retrieval

        with pytest.raises(TraceRecordingError) as exc_info:
            await audit_service.get_trace("req-001")

        assert "timeout" in exc_info.value.reason.lower()
        assert exc_info.value.request_id == "req-001"

    async def test_get_trace_timeout_configured_at_30_seconds(self):
        """Timeout constant is 30 seconds per Requirement 8.6."""
        assert GET_TRACE_TIMEOUT_SECONDS == 30.0


# =============================================================================
# Error Handling Tests — HALT PA Processing
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and PA processing halt (Requirement 8.5)."""

    async def test_recording_failure_raises_trace_recording_error(self):
        """If storage.append() fails, TraceRecordingError is raised."""
        storage = InMemoryAppendOnlyStorage()

        async def failing_append(entry: TraceEntry) -> None:
            raise RuntimeError("Disk full")

        storage.append = failing_append
        service = AuditTrailService(storage_backend=storage)

        with pytest.raises(TraceRecordingError) as exc_info:
            await service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=TraceCategory.AGENT_ACTION,
            )

        assert "Trace recording failed" in exc_info.value.reason
        assert exc_info.value.request_id == "req-001"

    async def test_trace_recording_error_contains_request_id(self):
        """TraceRecordingError includes the request_id for context."""
        storage = InMemoryAppendOnlyStorage()

        async def failing_append(entry: TraceEntry) -> None:
            raise IOError("Network partition")

        storage.append = failing_append
        service = AuditTrailService(storage_backend=storage)

        with pytest.raises(TraceRecordingError) as exc_info:
            await service.record_entry(
                request_id="pa-req-critical",
                identity_id="agent-001",
                category=TraceCategory.DECISION_STEP,
            )

        assert exc_info.value.request_id == "pa-req-critical"

    async def test_retrieval_failure_raises_trace_recording_error(self):
        """If storage retrieval fails, TraceRecordingError is raised."""
        storage = InMemoryAppendOnlyStorage()

        async def failing_get(request_id: str) -> list[TraceEntry]:
            raise RuntimeError("Storage unavailable")

        storage.get_entries_by_request_id = failing_get
        service = AuditTrailService(storage_backend=storage)

        with pytest.raises(TraceRecordingError) as exc_info:
            await service.get_trace("req-001")

        assert "retrieval failed" in exc_info.value.reason.lower()

    async def test_no_unaudited_decisions_on_failure(self):
        """Recording failure prevents continuation — no unaudited decisions.

        This verifies the contract: if record_entry fails, the caller
        cannot proceed (exception propagates, halting PA processing).
        """
        storage = InMemoryAppendOnlyStorage()

        async def failing_append(entry: TraceEntry) -> None:
            raise RuntimeError("Cannot write")

        storage.append = failing_append
        service = AuditTrailService(storage_backend=storage)

        processing_continued = False
        try:
            await service.record_entry(
                request_id="req-001",
                identity_id="agent-001",
                category=TraceCategory.DECISION_STEP,
            )
            # This line should never be reached
            processing_continued = True
        except TraceRecordingError:
            pass

        assert not processing_continued, "PA processing must HALT on trace failure"


# =============================================================================
# Storage Protocol Compliance Tests
# =============================================================================


class TestStorageProtocol:
    """Tests verifying InMemoryAppendOnlyStorage complies with AppendOnlyStorage protocol."""

    def test_storage_implements_protocol(self, storage):
        """InMemoryAppendOnlyStorage satisfies AppendOnlyStorage protocol."""
        assert isinstance(storage, AppendOnlyStorage)

    async def test_storage_append_increases_count(self, storage):
        """Each append increases the entry count."""
        assert storage.entry_count == 0

        entry = TraceEntry(
            sequence_number=1,
            timestamp="2024-01-15T10:30:45.123Z",
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
        )
        await storage.append(entry)
        assert storage.entry_count == 1

    async def test_storage_get_entries_returns_correct_subset(self, storage):
        """get_entries_by_request_id returns only matching entries."""
        entries = [
            TraceEntry(
                sequence_number=i + 1,
                timestamp="2024-01-15T10:30:45.123Z",
                request_id=f"req-{i % 2}",
                identity_id="agent-001",
                category=TraceCategory.AGENT_ACTION,
            )
            for i in range(6)
        ]
        for e in entries:
            await storage.append(e)

        req0_entries = await storage.get_entries_by_request_id("req-0")
        req1_entries = await storage.get_entries_by_request_id("req-1")

        assert len(req0_entries) == 3
        assert len(req1_entries) == 3
        assert all(e.request_id == "req-0" for e in req0_entries)
        assert all(e.request_id == "req-1" for e in req1_entries)


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestFullTraceWorkflow:
    """End-to-end trace recording and retrieval workflow tests."""

    async def test_record_multiple_categories_and_retrieve(self, audit_service):
        """Record entries of different categories and retrieve the full trace."""
        categories = [
            TraceCategory.CONTEXT_RETRIEVAL,
            TraceCategory.AGENT_ACTION,
            TraceCategory.TOOL_INVOCATION,
            TraceCategory.DECISION_STEP,
        ]
        for cat in categories:
            await audit_service.record_entry(
                request_id="pa-req-full",
                identity_id="clinical-agent-001",
                category=cat,
            )

        trace = await audit_service.get_trace("pa-req-full")
        assert len(trace) == 4

        # Verify ordering
        for i in range(1, len(trace)):
            assert trace[i].sequence_number > trace[i - 1].sequence_number

        # Verify all categories represented
        recorded_categories = {e.category for e in trace}
        assert recorded_categories == set(categories)

    async def test_concurrent_recording_maintains_ordering(self, audit_service):
        """Concurrent record_entry calls still produce valid ordering."""

        async def record_batch(request_id: str, count: int):
            for _ in range(count):
                await audit_service.record_entry(
                    request_id=request_id,
                    identity_id="agent-001",
                    category=TraceCategory.AGENT_ACTION,
                )

        # Simulate concurrent recording
        await asyncio.gather(
            record_batch("req-A", 10),
            record_batch("req-B", 10),
        )

        # All sequence numbers should be unique across both
        all_entries = audit_service._storage.get_all_entries()
        all_seq = [e.sequence_number for e in all_entries]
        assert len(all_seq) == 20
        assert len(set(all_seq)) == 20  # All unique

    async def test_trace_entry_is_valid_pydantic_model(self, audit_service):
        """Recorded entries are valid TraceEntry pydantic models."""
        entry = await audit_service.record_entry(
            request_id="req-001",
            identity_id="agent-001",
            category=TraceCategory.AGENT_ACTION,
            details={"action": "start_reasoning"},
        )

        assert isinstance(entry, TraceEntry)
        # Validate through pydantic model validation
        validated = TraceEntry.model_validate(entry.model_dump())
        assert validated.sequence_number == entry.sequence_number
        assert validated.timestamp == entry.timestamp
