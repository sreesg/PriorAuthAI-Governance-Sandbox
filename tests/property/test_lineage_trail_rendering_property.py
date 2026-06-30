"""Property-based tests for Lineage Trail Rendering Completeness.

**Validates: Requirements 15.4**

Property 32: Lineage Trail Rendering Completeness
- For any Evidence Bundle with N lineage entries, all N entries are returned
  in the API response with conclusion, evidence_id, and timestamp fields.
- No lineage entry is lost or missing required fields in the response.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from hypothesis import given, settings
from hypothesis import strategies as st

from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    create_evidence_graph_router,
    store_evidence_bundle,
    get_evidence_bundle_store,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Non-empty text for string fields
non_empty_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.",
    min_size=1,
    max_size=80,
)

# Conclusion strings (clinical statements)
conclusion_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.,;:()",
    min_size=5,
    max_size=200,
)

# Evidence ID identifiers
evidence_id_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=40,
)

# ISO-8601 timestamp strings
timestamp_strategy = st.datetimes(
    min_value=__import__("datetime").datetime(2020, 1, 1),
    max_value=__import__("datetime").datetime(2030, 12, 31),
).map(lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")

# Confidence score (optional, 0.0-1.0)
confidence_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)

# Decision type
decision_strategy = st.sampled_from(["approve", "escalate"])

# KMS signature data
signature_strategy = st.fixed_dictionaries({
    "key_id": st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_:/",
        min_size=5,
        max_size=60,
    ),
    "signature": st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=",
        min_size=10,
        max_size=60,
    ),
    "algorithm": st.sampled_from([
        "RSASSA_PKCS1_V1_5_SHA_256",
        "RSASSA_PSS_SHA_256",
        "ECDSA_SHA_256",
    ]),
})


@st.composite
def lineage_entry_strategy(draw):
    """Generate a single valid lineage entry dictionary."""
    entry = {
        "conclusion": draw(conclusion_strategy),
        "evidence_id": draw(evidence_id_strategy),
        "timestamp": draw(timestamp_strategy),
    }
    confidence = draw(confidence_strategy)
    if confidence is not None:
        entry["confidence"] = confidence
    return entry


@st.composite
def lineage_trail_strategy(draw):
    """Generate a list of 1-15 lineage entries with unique evidence_ids."""
    num_entries = draw(st.integers(min_value=1, max_value=15))
    entries = []
    used_ids = set()
    for _ in range(num_entries):
        entry = draw(lineage_entry_strategy())
        # Ensure unique evidence_id
        while entry["evidence_id"] in used_ids:
            entry["evidence_id"] = draw(evidence_id_strategy)
        used_ids.add(entry["evidence_id"])
        entries.append(entry)
    return entries


@st.composite
def evidence_bundle_strategy(draw):
    """Generate a full evidence bundle with lineage trail and signatures."""
    lineage_trail = draw(lineage_trail_strategy())
    num_sigs = draw(st.integers(min_value=1, max_value=3))
    signatures = [draw(signature_strategy) for _ in range(num_sigs)]
    return {
        "decision": draw(decision_strategy),
        "reason": draw(non_empty_text),
        "lineage_trail": lineage_trail,
        "signatures": signatures,
    }


# =============================================================================
# Helpers
# =============================================================================


def _create_app():
    """Create a fresh FastAPI app with the evidence/graph router."""
    app = FastAPI()
    router = create_evidence_graph_router()
    app.include_router(router)
    return app


async def _store_bundle_and_query(execution_id: str, bundle_data: dict) -> dict:
    """Store an evidence bundle and query the API endpoint.

    Returns the API response as a parsed dictionary.
    """
    # Clear the store for isolation
    get_evidence_bundle_store().clear()

    app = _create_app()

    # Store the bundle
    store_evidence_bundle(execution_id, bundle_data)

    # Query the API endpoint
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"/api/evidence-bundle/{execution_id}")
        assert response.status_code == 200
        return response.json()


# =============================================================================
# Property 32: Lineage Trail Rendering Completeness
# =============================================================================


@pytest.mark.property
class TestLineageTrailRenderingCompleteness:
    """Property 32: Lineage Trail Rendering Completeness.

    **Validates: Requirements 15.4**

    For any Evidence Bundle with N lineage entries, all N entries are returned
    in the API response with conclusion, evidence_id, and timestamp fields.
    """

    @given(bundle_data=evidence_bundle_strategy())
    @settings(max_examples=100)
    def test_all_lineage_entries_present_in_response(self, bundle_data: dict):
        """All N lineage entries from the stored bundle appear in the API response.

        **Validates: Requirements 15.4**

        For an Evidence Bundle with N lineage entries, the API response
        must contain exactly N lineage entries.
        """
        execution_id = "exec-prop32-count"
        result = asyncio.get_event_loop().run_until_complete(
            _store_bundle_and_query(execution_id, bundle_data)
        )

        expected_count = len(bundle_data["lineage_trail"])
        actual_count = len(result["lineage_trail"])
        assert actual_count == expected_count, (
            f"Expected {expected_count} lineage entries, got {actual_count}"
        )

    @given(bundle_data=evidence_bundle_strategy())
    @settings(max_examples=100)
    def test_conclusion_present_in_all_lineage_entries(self, bundle_data: dict):
        """Every lineage entry in the response has a non-empty conclusion field.

        **Validates: Requirements 15.4**

        Each lineage entry must contain a conclusion statement linking
        the decision step to its reasoning.
        """
        execution_id = "exec-prop32-conclusion"
        result = asyncio.get_event_loop().run_until_complete(
            _store_bundle_and_query(execution_id, bundle_data)
        )

        for i, entry in enumerate(result["lineage_trail"]):
            assert "conclusion" in entry, (
                f"lineage_trail[{i}] is missing 'conclusion' field"
            )
            assert entry["conclusion"] is not None and len(entry["conclusion"]) > 0, (
                f"lineage_trail[{i}] has empty conclusion"
            )

    @given(bundle_data=evidence_bundle_strategy())
    @settings(max_examples=100)
    def test_evidence_id_present_in_all_lineage_entries(self, bundle_data: dict):
        """Every lineage entry in the response has a non-empty evidence_id field.

        **Validates: Requirements 15.4**

        Each lineage entry must contain an evidence_id linking the conclusion
        to its source evidence chunk.
        """
        execution_id = "exec-prop32-evidence-id"
        result = asyncio.get_event_loop().run_until_complete(
            _store_bundle_and_query(execution_id, bundle_data)
        )

        for i, entry in enumerate(result["lineage_trail"]):
            assert "evidence_id" in entry, (
                f"lineage_trail[{i}] is missing 'evidence_id' field"
            )
            assert entry["evidence_id"] is not None and len(entry["evidence_id"]) > 0, (
                f"lineage_trail[{i}] has empty evidence_id"
            )

    @given(bundle_data=evidence_bundle_strategy())
    @settings(max_examples=100)
    def test_timestamp_present_in_all_lineage_entries(self, bundle_data: dict):
        """Every lineage entry in the response has a non-empty timestamp field.

        **Validates: Requirements 15.4**

        Each lineage entry must contain a retrieval timestamp indicating
        when the evidence was retrieved.
        """
        execution_id = "exec-prop32-timestamp"
        result = asyncio.get_event_loop().run_until_complete(
            _store_bundle_and_query(execution_id, bundle_data)
        )

        for i, entry in enumerate(result["lineage_trail"]):
            assert "timestamp" in entry, (
                f"lineage_trail[{i}] is missing 'timestamp' field"
            )
            assert entry["timestamp"] is not None and len(entry["timestamp"]) > 0, (
                f"lineage_trail[{i}] has empty timestamp"
            )

    @given(bundle_data=evidence_bundle_strategy())
    @settings(max_examples=100)
    def test_lineage_conclusions_match_stored_data(self, bundle_data: dict):
        """Conclusions in the response match the originally stored conclusions.

        **Validates: Requirements 15.4**

        The API must faithfully render the conclusions without modification —
        each response conclusion must match its stored counterpart in order.
        """
        execution_id = "exec-prop32-match"
        result = asyncio.get_event_loop().run_until_complete(
            _store_bundle_and_query(execution_id, bundle_data)
        )

        stored_conclusions = [e["conclusion"] for e in bundle_data["lineage_trail"]]
        response_conclusions = [e["conclusion"] for e in result["lineage_trail"]]
        assert response_conclusions == stored_conclusions, (
            f"Conclusion mismatch.\nStored: {stored_conclusions}\n"
            f"Response: {response_conclusions}"
        )

    @given(bundle_data=evidence_bundle_strategy())
    @settings(max_examples=100)
    def test_lineage_evidence_ids_match_stored_data(self, bundle_data: dict):
        """Evidence IDs in the response match the originally stored evidence_ids.

        **Validates: Requirements 15.4**

        Each lineage entry's evidence_id in the response must match the
        stored value, preserving the link to source evidence.
        """
        execution_id = "exec-prop32-ids-match"
        result = asyncio.get_event_loop().run_until_complete(
            _store_bundle_and_query(execution_id, bundle_data)
        )

        stored_ids = [e["evidence_id"] for e in bundle_data["lineage_trail"]]
        response_ids = [e["evidence_id"] for e in result["lineage_trail"]]
        assert response_ids == stored_ids, (
            f"Evidence ID mismatch.\nStored: {stored_ids}\n"
            f"Response: {response_ids}"
        )
