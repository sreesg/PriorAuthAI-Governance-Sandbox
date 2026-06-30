"""Property-based tests for Execution Trace Ordering Invariant.

**Validates: Requirements 8.1, 8.2**

Property 13: Execution Trace Ordering Invariant
- For any sequence of trace entries recorded and retrieved:
  - sequence_numbers are strictly increasing
  - timestamps are valid UTC ISO-8601 with millisecond precision
  - request_id is non-empty and matches the input request_id
  - identity_id is non-empty
  - category is a valid TraceCategory enum member
"""

import asyncio
import re

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from clinical_reasoning_fabric.models.core import TraceCategory, TraceEntry


# =============================================================================
# Constants
# =============================================================================

# UTC ISO-8601 with millisecond precision: YYYY-MM-DDTHH:MM:SS.mmmZ
ISO8601_MS_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(Z|[+-]\d{2}:\d{2})$"
)

# All valid TraceCategory values
VALID_CATEGORIES = list(TraceCategory)


# =============================================================================
# Hypothesis Strategies
# =============================================================================


# Strategy for non-empty identity IDs
identity_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_@."),
    min_size=1,
    max_size=50,
)

# Strategy for picking a random TraceCategory
category_strategy = st.sampled_from(VALID_CATEGORIES)

# Strategy for generating a fixed request_id per test run (non-empty)
request_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=50,
)

# Strategy for number of entries to record (1-50)
num_entries_strategy = st.integers(min_value=1, max_value=50)


@st.composite
def trace_entries_input_strategy(draw):
    """Generate a list of (identity_id, category) pairs to record,
    along with a fixed request_id for the test run.
    """
    request_id = draw(request_id_strategy)
    num_entries = draw(num_entries_strategy)

    entries = []
    for _ in range(num_entries):
        identity_id = draw(identity_id_strategy)
        category = draw(category_strategy)
        entries.append((identity_id, category))

    return request_id, entries


# =============================================================================
# Helpers
# =============================================================================


def _make_service() -> AuditTrailService:
    """Create a fresh AuditTrailService with in-memory storage."""
    storage = InMemoryAppendOnlyStorage()
    return AuditTrailService(storage_backend=storage)


# =============================================================================
# Property 13: Execution Trace Ordering Invariant
# =============================================================================


@pytest.mark.property
class TestExecutionTraceOrderingInvariant:
    """Property 13: Execution Trace Ordering Invariant.

    **Validates: Requirements 8.1, 8.2**

    For any sequence of trace entries recorded and retrieved:
    1. sequence_numbers are strictly increasing
    2. timestamps are valid UTC ISO-8601 with millisecond precision
    3. request_id is non-empty and matches the input request_id
    4. identity_id is non-empty
    5. category is a valid TraceCategory enum member
    """

    @given(data=trace_entries_input_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_sequence_numbers_strictly_increasing(self, data):
        """Sequence numbers in the retrieved trace are strictly increasing.

        **Validates: Requirements 8.1**

        After recording N entries for a request_id, retrieving the trace
        yields entries whose sequence_numbers form a strictly increasing
        sequence (each > previous).
        """
        request_id, entries_input = data
        service = _make_service()

        # Record all entries
        for identity_id, category in entries_input:
            await service.record_entry(
                request_id=request_id,
                identity_id=identity_id,
                category=category,
            )

        # Retrieve the trace
        trace = await service.get_trace(request_id)

        assert len(trace) == len(entries_input), (
            f"Expected {len(entries_input)} entries, got {len(trace)}"
        )

        # Verify strictly increasing sequence numbers
        for i in range(1, len(trace)):
            assert trace[i].sequence_number > trace[i - 1].sequence_number, (
                f"Sequence numbers not strictly increasing at index {i}: "
                f"{trace[i - 1].sequence_number} >= {trace[i].sequence_number}"
            )

    @given(data=trace_entries_input_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_timestamps_valid_utc_iso8601_ms(self, data):
        """Every timestamp in the trace is valid UTC ISO-8601 with ms precision.

        **Validates: Requirements 8.2**

        Every recorded trace entry must have a timestamp matching the
        pattern YYYY-MM-DDTHH:MM:SS.mmmZ (UTC ISO-8601 with millisecond precision).
        """
        request_id, entries_input = data
        service = _make_service()

        # Record all entries
        for identity_id, category in entries_input:
            await service.record_entry(
                request_id=request_id,
                identity_id=identity_id,
                category=category,
            )

        # Retrieve the trace
        trace = await service.get_trace(request_id)

        for entry in trace:
            assert ISO8601_MS_PATTERN.match(entry.timestamp), (
                f"Timestamp '{entry.timestamp}' does not match UTC ISO-8601 "
                f"with millisecond precision (expected pattern: "
                f"YYYY-MM-DDTHH:MM:SS.mmmZ)"
            )

    @given(data=trace_entries_input_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_request_id_non_empty_and_matching(self, data):
        """Every entry's request_id is non-empty and matches the input request_id.

        **Validates: Requirements 8.2**

        The request_id correlation field in every trace entry must be
        non-empty and must match the request_id used when recording.
        """
        request_id, entries_input = data
        service = _make_service()

        # Record all entries
        for identity_id, category in entries_input:
            await service.record_entry(
                request_id=request_id,
                identity_id=identity_id,
                category=category,
            )

        # Retrieve the trace
        trace = await service.get_trace(request_id)

        for entry in trace:
            assert entry.request_id is not None and len(entry.request_id) > 0, (
                f"request_id must be non-empty, got: '{entry.request_id}'"
            )
            assert entry.request_id == request_id, (
                f"request_id mismatch: expected '{request_id}', "
                f"got '{entry.request_id}'"
            )

    @given(data=trace_entries_input_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_identity_id_non_empty(self, data):
        """Every entry's identity_id is non-empty.

        **Validates: Requirements 8.2**

        The identity_id (authenticated identity) field in every trace entry
        must be non-empty for audit attribution purposes.
        """
        request_id, entries_input = data
        service = _make_service()

        # Record all entries
        for identity_id, category in entries_input:
            await service.record_entry(
                request_id=request_id,
                identity_id=identity_id,
                category=category,
            )

        # Retrieve the trace
        trace = await service.get_trace(request_id)

        for i, entry in enumerate(trace):
            assert entry.identity_id is not None and len(entry.identity_id) > 0, (
                f"identity_id must be non-empty at index {i}, "
                f"got: '{entry.identity_id}'"
            )

    @given(data=trace_entries_input_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_category_valid_trace_category_enum(self, data):
        """Every entry's category is a valid TraceCategory enum member.

        **Validates: Requirements 8.1**

        The category field in every trace entry must be one of:
        AGENT_ACTION, TOOL_INVOCATION, CONTEXT_RETRIEVAL, or DECISION_STEP.
        """
        request_id, entries_input = data
        service = _make_service()

        # Record all entries
        for identity_id, category in entries_input:
            await service.record_entry(
                request_id=request_id,
                identity_id=identity_id,
                category=category,
            )

        # Retrieve the trace
        trace = await service.get_trace(request_id)

        for entry in trace:
            assert isinstance(entry.category, TraceCategory), (
                f"category must be a TraceCategory enum member, "
                f"got: {type(entry.category)} = {entry.category}"
            )
            assert entry.category in VALID_CATEGORIES, (
                f"category '{entry.category}' is not a valid TraceCategory. "
                f"Valid values: {[c.value for c in VALID_CATEGORIES]}"
            )

    @given(data=trace_entries_input_strategy())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_all_invariants_hold_together(self, data):
        """All trace ordering invariants hold simultaneously for any sequence.

        **Validates: Requirements 8.1, 8.2**

        Combined property: for any sequence of trace entries, ALL of the
        following hold at once:
        - sequence_numbers strictly increasing
        - timestamps valid UTC ISO-8601 with ms precision
        - request_id non-empty and matching
        - identity_id non-empty
        - category is valid TraceCategory
        """
        request_id, entries_input = data
        service = _make_service()

        # Record all entries
        for identity_id, category in entries_input:
            await service.record_entry(
                request_id=request_id,
                identity_id=identity_id,
                category=category,
            )

        # Retrieve the trace
        trace = await service.get_trace(request_id)

        assert len(trace) == len(entries_input), (
            f"Expected {len(entries_input)} entries, got {len(trace)}"
        )

        for i, entry in enumerate(trace):
            # Strictly increasing sequence numbers
            if i > 0:
                assert entry.sequence_number > trace[i - 1].sequence_number, (
                    f"Sequence numbers not strictly increasing at index {i}: "
                    f"{trace[i - 1].sequence_number} >= {entry.sequence_number}"
                )

            # Valid timestamp
            assert ISO8601_MS_PATTERN.match(entry.timestamp), (
                f"Invalid timestamp at index {i}: '{entry.timestamp}'"
            )

            # Non-empty, matching request_id
            assert entry.request_id == request_id and len(entry.request_id) > 0, (
                f"request_id mismatch or empty at index {i}: '{entry.request_id}'"
            )

            # Non-empty identity_id
            assert entry.identity_id is not None and len(entry.identity_id) > 0, (
                f"identity_id empty at index {i}: '{entry.identity_id}'"
            )

            # Valid category
            assert isinstance(entry.category, TraceCategory), (
                f"Invalid category at index {i}: {entry.category}"
            )
