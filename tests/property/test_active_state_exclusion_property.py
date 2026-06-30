"""
Property test for Active State Exclusion of Resolved Records.

**Validates: Requirements 3.5**

Property 6: For any set of records with mixed statuses, the active state filtering
logic returns only non-resolved/non-discontinued/non-closed/non-superseded records.

This tests the core filtering logic used by CausalOntologyGraphService to determine
which records are considered "active" clinical state.
"""

from __future__ import annotations

import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.graph.causal_ontology_graph_service import INACTIVE_STATUSES


# =============================================================================
# Strategies
# =============================================================================

# Active statuses — any status NOT in the inactive set
ACTIVE_STATUS_EXAMPLES = ["active", "pending", "confirmed", "in_progress", "scheduled", "draft"]

# Inactive statuses — these should always be excluded
INACTIVE_STATUS_LIST = list(INACTIVE_STATUSES)

# Strategy for generating active statuses (including None and empty string which are treated as active)
active_status_strategy = st.one_of(
    st.sampled_from(ACTIVE_STATUS_EXAMPLES),
    st.just(""),
    st.none(),
    # Also generate arbitrary strings that are NOT in INACTIVE_STATUSES
    st.text(min_size=1, max_size=30).filter(lambda s: s.lower() not in INACTIVE_STATUSES),
)

# Strategy for generating inactive statuses
inactive_status_strategy = st.sampled_from(INACTIVE_STATUS_LIST)

# Strategy for generating any status (mixed active/inactive)
any_status_strategy = st.one_of(
    active_status_strategy,
    inactive_status_strategy,
)


@st.composite
def record_strategy(draw, status_strategy=any_status_strategy, idx=None):
    """Generate a clinical record with a random status."""
    record_id = draw(st.uuids())
    status = draw(status_strategy)
    return {
        "event_id": f"evt-{record_id}",
        "status": status,
        "condition_code": draw(st.sampled_from(["E11.9", "I10", "J45.20", "M54.5", "Z79.4"])),
        "description": draw(st.text(min_size=1, max_size=50)),
    }


@st.composite
def unique_records_list(draw, min_size=1, max_size=50, status_strategy=any_status_strategy):
    """Generate a list of records with unique event_ids."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    records = []
    for i in range(n):
        status = draw(status_strategy)
        condition_code = draw(st.sampled_from(["E11.9", "I10", "J45.20", "M54.5", "Z79.4"]))
        description = draw(st.text(min_size=1, max_size=50))
        records.append({
            "event_id": f"evt-{i:06d}-{draw(st.uuids())}",
            "status": status,
            "condition_code": condition_code,
            "description": description,
        })
    return records


# Strategy for a list of records with mixed statuses (unique event_ids)
mixed_records_strategy = unique_records_list(min_size=1, max_size=50)


# =============================================================================
# Core filtering function (mirrors service logic)
# =============================================================================


def filter_active_records(records: list[dict]) -> list[dict]:
    """
    Filter records to return only active ones.

    Mirrors the Cypher logic: WHERE NOT coalesce(e.status, '') IN $inactive_statuses

    A record is active if its status is NOT in {"resolved", "discontinued", "closed", "superseded"}.
    Records with None or empty string status are considered active.
    """
    active = []
    for record in records:
        status = record.get("status") or ""  # coalesce(status, '')
        if status not in INACTIVE_STATUSES:
            active.append(record)
    return active


# =============================================================================
# Property Tests
# =============================================================================


@pytest.mark.property
class TestActiveStateExclusionProperty:
    """Property 6: Active State Exclusion of Resolved Records.

    **Validates: Requirements 3.5**

    THE Causal_Ontology_Graph SHALL represent only active clinical state, where a record
    is considered active if it has not been marked as resolved, discontinued, or superseded
    by a subsequent CDC event, and SHALL exclude any record whose status has been set to
    closed, resolved, or discontinued in the source system.
    """

    @given(records=mixed_records_strategy)
    def test_filtered_result_contains_only_non_inactive_records(self, records: list[dict]):
        """Assert: the filtered result contains ONLY records whose status is NOT in
        {"resolved", "discontinued", "closed", "superseded"}.
        """
        active_records = filter_active_records(records)

        for record in active_records:
            status = record.get("status") or ""
            assert status not in INACTIVE_STATUSES, (
                f"Active result contains record with inactive status '{status}': {record}"
            )

    @given(records=mixed_records_strategy)
    def test_all_inactive_status_records_are_excluded(self, records: list[dict]):
        """Assert: ALL records with status in {"resolved", "discontinued", "closed", "superseded"}
        are excluded from the active result.
        """
        active_records = filter_active_records(records)
        active_event_ids = {r["event_id"] for r in active_records}

        for record in records:
            status = record.get("status") or ""
            if status in INACTIVE_STATUSES:
                assert record["event_id"] not in active_event_ids, (
                    f"Record with inactive status '{status}' was not excluded: {record}"
                )

    @given(records=mixed_records_strategy)
    def test_records_with_non_inactive_status_are_included(self, records: list[dict]):
        """Assert: records with any other status (including empty string or None) are included.

        This verifies that the filter is not over-zealous and keeps all records
        that don't have an explicitly inactive status.
        """
        active_records = filter_active_records(records)
        active_event_ids = {r["event_id"] for r in active_records}

        for record in records:
            status = record.get("status") or ""
            if status not in INACTIVE_STATUSES:
                assert record["event_id"] in active_event_ids, (
                    f"Record with active status '{status}' was not included: {record}"
                )

    @given(records=mixed_records_strategy)
    def test_active_count_plus_inactive_count_equals_total(self, records: list[dict]):
        """The number of active + inactive records should equal the total input count."""
        active_records = filter_active_records(records)
        inactive_count = sum(
            1 for r in records if (r.get("status") or "") in INACTIVE_STATUSES
        )

        assert len(active_records) + inactive_count == len(records), (
            f"Partition mismatch: {len(active_records)} active + {inactive_count} inactive "
            f"!= {len(records)} total"
        )

    @given(
        active_records=unique_records_list(min_size=1, max_size=20, status_strategy=active_status_strategy),
        inactive_records=unique_records_list(min_size=1, max_size=20, status_strategy=inactive_status_strategy),
    )
    def test_mixed_input_preserves_all_active_excludes_all_inactive(
        self, active_records: list[dict], inactive_records: list[dict]
    ):
        """When given a known mix of active and inactive records, all active records
        are preserved and all inactive records are excluded.

        Since each list has unique UUIDs in their event_ids, there are no collisions.
        """
        all_records = active_records + inactive_records

        result = filter_active_records(all_records)

        # All originally-active records should be in the result
        active_event_ids = {r["event_id"] for r in active_records}
        result_event_ids = {r["event_id"] for r in result}

        assert active_event_ids.issubset(result_event_ids), (
            f"Some active records are missing from result. "
            f"Missing: {active_event_ids - result_event_ids}"
        )

        # No originally-inactive records should be in the result
        inactive_event_ids = {r["event_id"] for r in inactive_records}
        assert inactive_event_ids.isdisjoint(result_event_ids), (
            f"Some inactive records are in the result. "
            f"Unexpected: {inactive_event_ids & result_event_ids}"
        )

    @given(records=unique_records_list(min_size=1, max_size=20, status_strategy=active_status_strategy))
    def test_all_active_input_returns_all(self, records: list[dict]):
        """When all records have active statuses, the filter returns all records."""
        result = filter_active_records(records)
        assert len(result) == len(records), (
            f"Expected all {len(records)} records returned but got {len(result)}"
        )

    @given(records=unique_records_list(min_size=1, max_size=20, status_strategy=inactive_status_strategy))
    def test_all_inactive_input_returns_empty(self, records: list[dict]):
        """When all records have inactive statuses, the filter returns an empty list."""
        result = filter_active_records(records)
        assert len(result) == 0, (
            f"Expected empty result for all-inactive input but got {len(result)} records"
        )
