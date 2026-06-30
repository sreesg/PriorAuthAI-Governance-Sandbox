"""
AuditTrailService — BEACON Layer 6: Observability and Immutable Audit Trail.

Implements immutable, append-only execution trace recording for every PA request.
Every agent action, tool invocation, context retrieval, and decision step is
recorded with monotonically increasing sequence numbers.

Requirements referenced: 8.1, 8.2, 8.4, 8.5, 8.6

Key behaviors:
- record_entry(): assigns monotonically increasing sequence numbers globally,
  includes UTC ISO-8601 timestamps with ms precision, request_id, identity_id,
  and category.
- Append-only storage: prohibits modification or deletion of historical entries.
- 7-year retention policy.
- get_trace(): retrieves complete trace by request_id within 30 seconds.
- On recording failure: HALT PA processing, raise TraceRecordingError.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from clinical_reasoning_fabric.models.core import TraceCategory, TraceEntry
from clinical_reasoning_fabric.models.exceptions import TraceRecordingError


# =============================================================================
# Constants
# =============================================================================

# 7-year retention policy in days (Requirement 8.4)
RETENTION_POLICY_YEARS: int = 7
RETENTION_POLICY_DAYS: int = RETENTION_POLICY_YEARS * 365

# Maximum time allowed for trace retrieval (Requirement 8.6)
GET_TRACE_TIMEOUT_SECONDS: float = 30.0


# =============================================================================
# Storage Protocol
# =============================================================================


@runtime_checkable
class AppendOnlyStorage(Protocol):
    """Protocol for an append-only storage backend.

    Implementations MUST:
    - Only allow appending new entries (no update/delete of historical entries)
    - Support retrieval of all entries for a given request_id
    - Enforce the 7-year retention policy

    Requirement 8.4: Store execution traces in append-only storage that does
    not permit modification or deletion of historical entries and retains
    traces for a minimum of 7 years.
    """

    async def append(self, entry: TraceEntry) -> None:
        """Append a trace entry to storage. Raises on failure."""
        ...

    async def get_entries_by_request_id(self, request_id: str) -> list[TraceEntry]:
        """Retrieve all trace entries for a request_id, ordered by sequence_number."""
        ...

    async def delete(self, sequence_number: int) -> None:
        """Attempting to delete MUST raise an error — storage is immutable."""
        ...

    async def update(self, sequence_number: int, entry: TraceEntry) -> None:
        """Attempting to update MUST raise an error — storage is immutable."""
        ...


# =============================================================================
# In-Memory Append-Only Storage Implementation
# =============================================================================


class ImmutableStorageViolationError(Exception):
    """Raised when an attempt is made to modify or delete historical entries."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Immutable storage violation: '{operation}' is prohibited. "
            f"Historical entries cannot be modified or deleted."
        )


class InMemoryAppendOnlyStorage:
    """In-memory implementation of AppendOnlyStorage.

    Provides an append-only store that prohibits modification or deletion.
    Thread-safe for concurrent access.

    Requirement 8.4: Append-only storage that does not permit modification
    or deletion of historical entries, with 7-year retention.
    """

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []
        self._lock = threading.Lock()
        self._retention_days: int = RETENTION_POLICY_DAYS

    @property
    def retention_policy_days(self) -> int:
        """Retention policy duration in days (7 years)."""
        return self._retention_days

    @property
    def entry_count(self) -> int:
        """Total number of stored entries."""
        with self._lock:
            return len(self._entries)

    async def append(self, entry: TraceEntry) -> None:
        """Append a trace entry. Thread-safe.

        Raises:
            TraceRecordingError: If the append operation fails.
        """
        with self._lock:
            self._entries.append(entry)

    async def get_entries_by_request_id(self, request_id: str) -> list[TraceEntry]:
        """Retrieve all entries for a request_id, ordered by sequence_number.

        Returns entries sorted by sequence_number (ascending).
        """
        with self._lock:
            matching = [e for e in self._entries if e.request_id == request_id]
        return sorted(matching, key=lambda e: e.sequence_number)

    async def delete(self, sequence_number: int) -> None:
        """PROHIBITED: Raises ImmutableStorageViolationError.

        Requirement 8.4: Storage does not permit deletion of historical entries.
        """
        raise ImmutableStorageViolationError("delete")

    async def update(self, sequence_number: int, entry: TraceEntry) -> None:
        """PROHIBITED: Raises ImmutableStorageViolationError.

        Requirement 8.4: Storage does not permit modification of historical entries.
        """
        raise ImmutableStorageViolationError("update")

    def get_all_entries(self) -> list[TraceEntry]:
        """Get all stored entries (for testing/debugging only)."""
        with self._lock:
            return list(self._entries)


# =============================================================================
# AuditTrailService
# =============================================================================


class AuditTrailService:
    """Immutable, append-only execution trace recording service.

    Records every agent action, tool invocation, context retrieval, and decision
    step with monotonically increasing sequence numbers and UTC ISO-8601 timestamps.

    Requirement 8.1: Record immutable execution trace with entries categorized as
    agent_action, tool_invocation, context_retrieval, or decision_step.

    Requirement 8.2: Include UTC ISO-8601 timestamp with ms precision, request_id,
    authenticated identity, and entry category in every trace entry.

    Requirement 8.5: If trace recording fails, HALT PA processing and return error.

    Requirement 8.6: Support retrieval of complete execution trace by request_id
    within 30 seconds.
    """

    def __init__(self, storage_backend: AppendOnlyStorage) -> None:
        """Initialize AuditTrailService with an append-only storage backend.

        Args:
            storage_backend: An implementation of AppendOnlyStorage protocol.
        """
        self._storage = storage_backend
        self._sequence_counter: int = 0
        self._sequence_lock = threading.Lock()

    @property
    def storage(self) -> AppendOnlyStorage:
        """Access the underlying storage backend."""
        return self._storage

    @property
    def current_sequence_number(self) -> int:
        """The current (last assigned) sequence number."""
        with self._sequence_lock:
            return self._sequence_counter

    def _next_sequence_number(self) -> int:
        """Generate the next monotonically increasing sequence number.

        Thread-safe. Each call returns a strictly higher value than the previous.
        """
        with self._sequence_lock:
            self._sequence_counter += 1
            return self._sequence_counter

    @staticmethod
    def _generate_timestamp() -> str:
        """Generate a UTC ISO-8601 timestamp with millisecond precision.

        Format: YYYY-MM-DDTHH:MM:SS.mmmZ
        Example: 2024-01-15T10:30:45.123Z

        Requirement 8.2: UTC ISO-8601 timestamp with millisecond precision.
        """
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    async def record_entry(
        self,
        request_id: str,
        identity_id: str,
        category: TraceCategory,
        details: dict | None = None,
    ) -> TraceEntry:
        """Record a trace entry with monotonically increasing sequence number.

        Assigns a globally unique, strictly increasing sequence number and a
        UTC ISO-8601 timestamp with millisecond precision.

        Args:
            request_id: Correlation ID for the PA request.
            identity_id: Authenticated identity performing the action.
            category: The type of trace entry (agent_action, tool_invocation,
                     context_retrieval, or decision_step).
            details: Optional structured metadata for this entry.

        Returns:
            The recorded TraceEntry with assigned sequence_number and timestamp.

        Raises:
            TraceRecordingError: On any recording failure. This HALTS PA processing
                (Requirement 8.5).
        """
        try:
            seq_num = self._next_sequence_number()
            timestamp = self._generate_timestamp()

            entry = TraceEntry(
                sequence_number=seq_num,
                timestamp=timestamp,
                request_id=request_id,
                identity_id=identity_id,
                category=category,
                details=details,
            )

            await self._storage.append(entry)
            return entry

        except Exception as exc:
            # Requirement 8.5: On failure, HALT PA processing
            raise TraceRecordingError(
                reason=f"Trace recording failed: {str(exc)}",
                request_id=request_id,
            ) from exc

    async def get_trace(self, request_id: str) -> list[TraceEntry]:
        """Retrieve the complete execution trace for a request_id.

        Retrieves all trace entries for the given request_id within a
        30-second timeout. Returns entries ordered by sequence_number.

        Args:
            request_id: The request_id to retrieve the trace for.

        Returns:
            List of TraceEntry objects ordered by sequence_number.

        Raises:
            TraceRecordingError: If retrieval fails or exceeds 30 seconds.

        Requirement 8.6: Retrieval within 30 seconds of query submission.
        """
        try:
            entries = await asyncio.wait_for(
                self._storage.get_entries_by_request_id(request_id),
                timeout=GET_TRACE_TIMEOUT_SECONDS,
            )
            return entries

        except asyncio.TimeoutError:
            raise TraceRecordingError(
                reason=(
                    f"Trace retrieval for request_id '{request_id}' "
                    f"exceeded {GET_TRACE_TIMEOUT_SECONDS}s timeout"
                ),
                request_id=request_id,
            )
        except TraceRecordingError:
            raise
        except Exception as exc:
            raise TraceRecordingError(
                reason=f"Trace retrieval failed: {str(exc)}",
                request_id=request_id,
            ) from exc
