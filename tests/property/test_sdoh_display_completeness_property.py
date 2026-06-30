"""Property-based tests for SDOH Display Completeness.

**Validates: Requirements 15.6**

Property 34: SDOH Display Completeness
- For any inferred SDOH factor returned from the API: includes type, category
  (for sdoh_factor), conclusion, confidence score (0.00-1.00), and inference chain.
- All required fields are present and non-null for every inferred fact.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from hypothesis import given, settings
from hypothesis import strategies as st

from src.clinical_reasoning_fabric.beacon.audit_trail_service import (
    AuditTrailService,
    InMemoryAppendOnlyStorage,
)
from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    create_frontend_router,
)
from src.clinical_reasoning_fabric.models.core import TraceCategory


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Valid inference types
inference_type_strategy = st.sampled_from([
    "sdoh_factor", "medication_adherence_risk", "care_access_barrier",
])

# Valid SDOH categories
sdoh_category_strategy = st.sampled_from([
    "housing_instability",
    "transportation_barriers",
    "medication_storage_limitations",
    "food_insecurity",
    "caregiver_availability",
])

# Confidence scores
confidence_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Non-empty identifier strings
identifier_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=30,
)

# Conclusion/text strings
text_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.,;:()",
    min_size=5,
    max_size=200,
)

# Source text (up to 500 chars)
source_text_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.,;:()",
    min_size=10,
    max_size=500,
)


@st.composite
def inference_chain_hop_strategy(draw, hop_number: int = 1):
    """Generate a single inference chain hop."""
    return {
        "hop_number": hop_number,
        "source_text": draw(text_strategy),
        "intermediate_conclusion": draw(text_strategy),
        "confidence": draw(confidence_strategy),
    }


@st.composite
def inference_chain_strategy(draw):
    """Generate a complete inference chain with 1-3 hops."""
    num_hops = draw(st.integers(min_value=1, max_value=3))
    hops = []
    cumulative_confidence = 1.0
    for i in range(num_hops):
        hop = draw(inference_chain_hop_strategy(hop_number=i + 1))
        hops.append(hop)
        cumulative_confidence *= hop["confidence"]

    return {
        "chain_id": draw(identifier_strategy),
        "hops": hops,
        "cumulative_confidence": cumulative_confidence,
        "final_conclusion": draw(text_strategy),
    }


@st.composite
def inferred_fact_strategy(draw):
    """Generate a single inferred SDOH fact with all required fields."""
    inference_type = draw(inference_type_strategy)
    # Category is required for sdoh_factor type
    category = draw(sdoh_category_strategy) if inference_type == "sdoh_factor" else None

    return {
        "fact_id": draw(identifier_strategy),
        "type": inference_type,
        "category": category,
        "conclusion": draw(text_strategy),
        "confidence": draw(confidence_strategy),
        "chain": draw(inference_chain_strategy()),
        "source_text": draw(source_text_strategy),
    }


@st.composite
def inferred_facts_list_strategy(draw):
    """Generate a list of 1-10 inferred facts with unique fact_ids."""
    num_facts = draw(st.integers(min_value=1, max_value=10))
    facts = []
    used_ids: set[str] = set()
    for _ in range(num_facts):
        fact = draw(inferred_fact_strategy())
        # Ensure unique fact_id
        while fact["fact_id"] in used_ids:
            fact["fact_id"] = draw(identifier_strategy)
        used_ids.add(fact["fact_id"])
        facts.append(fact)
    return facts


# =============================================================================
# Helpers
# =============================================================================


def _create_app_and_service():
    """Create a fresh FastAPI app and AuditTrailService for each test."""
    storage = InMemoryAppendOnlyStorage()
    audit_service = AuditTrailService(storage_backend=storage)
    app = FastAPI()
    router = create_frontend_router(audit_service)
    app.include_router(router)
    return app, audit_service


async def _store_facts_and_query(member_id: str, facts: list[dict]) -> dict:
    """Store inferred facts via audit trail and query the SDOH API endpoint.

    Returns the API response as a parsed dictionary.
    """
    app, audit_service = _create_app_and_service()
    request_id = "test-request-prop34"

    # Store inferred facts as an agent_action entry with member_id reference
    await audit_service.record_entry(
        request_id=request_id,
        identity_id="test-user",
        category=TraceCategory.AGENT_ACTION,
        details={
            "member_id": member_id,
            "inferred_facts": facts,
        },
    )

    # Query the API endpoint
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"/api/inference/sdoh/{member_id}")
        assert response.status_code == 200
        return response.json()


# =============================================================================
# Property 34: SDOH Display Completeness
# =============================================================================


@pytest.mark.property
class TestSDOHDisplayCompleteness:
    """Property 34: SDOH Display Completeness.

    **Validates: Requirements 15.6**

    For any inferred SDOH factor from the API: includes type, category
    (for sdoh_factor), conclusion, confidence score, and inference chain.
    """

    @given(facts=inferred_facts_list_strategy())
    @settings(max_examples=100)
    def test_all_inferred_facts_present_in_response(self, facts: list[dict]):
        """All stored inferred facts appear in the API response.

        **Validates: Requirements 15.6**

        For N inferred facts stored, the API response must contain
        exactly N inferred facts.
        """
        member_id = "member-prop34-count"
        result = asyncio.get_event_loop().run_until_complete(
            _store_facts_and_query(member_id, facts)
        )

        expected_count = len(facts)
        actual_count = len(result["inferred_facts"])
        assert actual_count == expected_count, (
            f"Expected {expected_count} inferred facts, got {actual_count}"
        )

    @given(facts=inferred_facts_list_strategy())
    @settings(max_examples=100)
    def test_type_present_in_all_inferred_facts(self, facts: list[dict]):
        """Every inferred fact in the response has a non-empty type field.

        **Validates: Requirements 15.6**

        The type field identifies the inference category and must be one of
        the valid inference types.
        """
        member_id = "member-prop34-type"
        result = asyncio.get_event_loop().run_until_complete(
            _store_facts_and_query(member_id, facts)
        )

        valid_types = {"sdoh_factor", "medication_adherence_risk", "care_access_barrier"}
        for i, fact in enumerate(result["inferred_facts"]):
            assert "type" in fact, f"inferred_facts[{i}] is missing 'type' field"
            assert fact["type"] is not None and len(fact["type"]) > 0, (
                f"inferred_facts[{i}] has empty type"
            )
            assert fact["type"] in valid_types, (
                f"inferred_facts[{i}] has invalid type '{fact['type']}'"
            )

    @given(facts=inferred_facts_list_strategy())
    @settings(max_examples=100)
    def test_category_present_for_sdoh_factor_type(self, facts: list[dict]):
        """Every inferred fact with type 'sdoh_factor' has a non-null category.

        **Validates: Requirements 15.6**

        The category field is required for sdoh_factor type facts and
        must be from the valid SDOH categories set.
        """
        member_id = "member-prop34-category"
        result = asyncio.get_event_loop().run_until_complete(
            _store_facts_and_query(member_id, facts)
        )

        valid_categories = {
            "housing_instability",
            "transportation_barriers",
            "medication_storage_limitations",
            "food_insecurity",
            "caregiver_availability",
        }
        for i, fact in enumerate(result["inferred_facts"]):
            if fact["type"] == "sdoh_factor":
                assert fact.get("category") is not None, (
                    f"inferred_facts[{i}] with type 'sdoh_factor' has null category"
                )
                assert fact["category"] in valid_categories, (
                    f"inferred_facts[{i}] has invalid category '{fact['category']}'"
                )

    @given(facts=inferred_facts_list_strategy())
    @settings(max_examples=100)
    def test_conclusion_present_in_all_inferred_facts(self, facts: list[dict]):
        """Every inferred fact in the response has a non-empty conclusion field.

        **Validates: Requirements 15.6**

        The conclusion field describes the inferred finding and must be
        present and non-empty for every inferred fact.
        """
        member_id = "member-prop34-conclusion"
        result = asyncio.get_event_loop().run_until_complete(
            _store_facts_and_query(member_id, facts)
        )

        for i, fact in enumerate(result["inferred_facts"]):
            assert "conclusion" in fact, (
                f"inferred_facts[{i}] is missing 'conclusion' field"
            )
            assert fact["conclusion"] is not None and len(fact["conclusion"]) > 0, (
                f"inferred_facts[{i}] has empty conclusion"
            )

    @given(facts=inferred_facts_list_strategy())
    @settings(max_examples=100)
    def test_confidence_present_and_valid_in_all_inferred_facts(self, facts: list[dict]):
        """Every inferred fact has a confidence score in [0.0, 1.0].

        **Validates: Requirements 15.6**

        The confidence score quantifies the inference certainty and must
        be a float between 0.0 and 1.0 inclusive.
        """
        member_id = "member-prop34-confidence"
        result = asyncio.get_event_loop().run_until_complete(
            _store_facts_and_query(member_id, facts)
        )

        for i, fact in enumerate(result["inferred_facts"]):
            assert "confidence" in fact, (
                f"inferred_facts[{i}] is missing 'confidence' field"
            )
            confidence = fact["confidence"]
            assert confidence is not None, (
                f"inferred_facts[{i}] has null confidence"
            )
            assert 0.0 <= confidence <= 1.0, (
                f"inferred_facts[{i}] confidence {confidence} not in [0.0, 1.0]"
            )

    @given(facts=inferred_facts_list_strategy())
    @settings(max_examples=100)
    def test_chain_present_in_all_inferred_facts(self, facts: list[dict]):
        """Every inferred fact has a non-null inference chain with hops.

        **Validates: Requirements 15.6**

        The chain field provides the complete reasoning pathway and
        must be present with at least one hop for every inferred fact.
        """
        member_id = "member-prop34-chain"
        result = asyncio.get_event_loop().run_until_complete(
            _store_facts_and_query(member_id, facts)
        )

        for i, fact in enumerate(result["inferred_facts"]):
            assert "chain" in fact, (
                f"inferred_facts[{i}] is missing 'chain' field"
            )
            chain = fact["chain"]
            assert chain is not None, (
                f"inferred_facts[{i}] has null chain"
            )
            assert "hops" in chain, (
                f"inferred_facts[{i}].chain is missing 'hops' field"
            )
            assert len(chain["hops"]) >= 1, (
                f"inferred_facts[{i}].chain has no hops"
            )
            assert "chain_id" in chain, (
                f"inferred_facts[{i}].chain is missing 'chain_id'"
            )
