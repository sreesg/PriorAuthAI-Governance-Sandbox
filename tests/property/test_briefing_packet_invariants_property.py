"""Property-based tests for Briefing Packet Assembly Invariants.

**Validates: Requirements 4.2, 4.3, 4.4**

Property 7: Briefing Packet Assembly Invariants
- For any valid PA request: at most 20 snippets, all scores >= 0.5,
  only CPT-relevant diagnoses, and schema conformance with all required fields.
- The BriefingPacket always contains non-null request_id, member_id, cpt_code,
  and active_clinical_state.
- verified_evidence_snippets is always a list (may be empty).
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.context_planner_service import (
    CPT_CONDITION_CATEGORIES,
    ContextPlannerService,
    PARequest,
)
from clinical_reasoning_fabric.models.core import (
    BriefingPacket,
    ChunkProvenance,
    KMSSignature,
    MemberActiveState,
    RetrievalResult,
    ScoredChunk,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_kms_signature() -> KMSSignature:
    """Create a valid KMSSignature for testing."""
    return KMSSignature(
        key_id="arn:aws:kms:us-east-1:123456789012:key/test-key-id",
        signature="dGVzdC1zaWduYXR1cmUtYmFzZTY0",
        algorithm="RSASSA_PKCS1_V1_5_SHA_256",
        signed_at=datetime.now(timezone.utc),
    )


def _make_content_hash(seed: str) -> str:
    """Create a valid 64-char hex content hash from a seed string."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _make_scored_chunk(
    chunk_id: str, score: float, text: str = ""
) -> ScoredChunk:
    """Create a valid ScoredChunk with given id, score, and text."""
    content_hash = _make_content_hash(chunk_id)
    if not text:
        text = f"Clinical text for chunk {chunk_id}"
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        provenance=ChunkProvenance(
            document_id="doc-001",
            content_hash=content_hash,
            kms_signature=_make_kms_signature(),
            chunk_index=0,
            ingestion_timestamp=datetime.now(timezone.utc),
        ),
        dense_rank=None,
        sparse_rank=None,
    )


# Known CPT codes from the service's mapping
KNOWN_CPT_CODES = list(CPT_CONDITION_CATEGORIES.keys())


# =============================================================================
# Hypothesis strategies
# =============================================================================

# Strategy for non-empty identifiers
identifier_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")

# Strategy for CPT codes - mix of known and unknown codes
cpt_code_strategy = st.one_of(
    st.sampled_from(KNOWN_CPT_CODES),
    st.from_regex(r"[0-9]{5}", fullmatch=True),
)

# Strategy for valid PA requests
pa_request_strategy = st.builds(
    PARequest,
    request_id=st.uuids().map(str),
    member_id=identifier_strategy,
    cpt_code=cpt_code_strategy,
    clinical_context=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
)


@st.composite
def scored_chunks_strategy(draw, min_size=0, max_size=30):
    """Generate a list of ScoredChunks with scores in [0.5, 1.0].

    All generated chunks have valid scores above the minimum threshold,
    simulating a retrieval service that already filters by min_score.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    chunks = []
    for i in range(size):
        score = draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False))
        # Include relevant clinical terms in text so snippets pass relevance filter
        category_terms = []
        for categories in CPT_CONDITION_CATEGORIES.values():
            category_terms.extend(categories)
        # Pick a random term to include in text to make it relevant
        term = draw(st.sampled_from(category_terms)) if category_terms else "clinical"
        text = f"Patient presents with {term} findings documented in record {i}"
        chunks.append(_make_scored_chunk(f"chunk-{i:04d}", score, text))
    return chunks


@st.composite
def diagnosis_strategy(draw):
    """Generate a diagnosis record dict with clinical terms."""
    # Include a condition keyword that maps to a CPT category
    all_categories = []
    for categories in CPT_CONDITION_CATEGORIES.values():
        all_categories.extend(categories)
    condition = draw(st.sampled_from(all_categories)) if all_categories else "general"
    return {
        "condition_code": f"ICD-{draw(st.integers(min_value=1, max_value=999)):03d}",
        "description": f"Diagnosis involving {condition}",
        "status": "active",
    }


@st.composite
def prescription_strategy(draw):
    """Generate a prescription record dict."""
    all_categories = []
    for categories in CPT_CONDITION_CATEGORIES.values():
        all_categories.extend(categories)
    term = draw(st.sampled_from(all_categories)) if all_categories else "medication"
    return {
        "medication_name": f"Drug for {term}",
        "dosage": "10mg",
        "status": "active",
    }


@st.composite
def member_active_state_strategy(draw, member_id: str = "test-member"):
    """Generate a MemberActiveState with random diagnoses and prescriptions."""
    n_diagnoses = draw(st.integers(min_value=0, max_value=10))
    n_prescriptions = draw(st.integers(min_value=0, max_value=5))

    diagnoses = [draw(diagnosis_strategy()) for _ in range(n_diagnoses)]
    prescriptions = [draw(prescription_strategy()) for _ in range(n_prescriptions)]

    return MemberActiveState(
        member_id=member_id,
        active_diagnoses=diagnoses,
        active_prescriptions=prescriptions,
        sdoh_factors=[],
        governing_policies=[],
        last_updated=datetime.now(timezone.utc),
    )


@st.composite
def full_assembly_scenario_strategy(draw):
    """Generate a complete scenario for Briefing Packet assembly.

    Returns (pa_request, member_state, retrieval_result) tuple to be used
    with mock services.
    """
    pa_request = draw(pa_request_strategy)
    member_state = draw(member_active_state_strategy(member_id=pa_request.member_id))
    chunks = draw(scored_chunks_strategy(min_size=0, max_size=30))

    no_evidence_found = len(chunks) == 0

    retrieval_result = RetrievalResult(
        verified_chunks=chunks,
        tamper_alerts=[],
        no_evidence_found=no_evidence_found,
        degraded_search=False,
        total_candidates=len(chunks),
    )

    return pa_request, member_state, retrieval_result


# =============================================================================
# Property 7: Briefing Packet Assembly Invariants
# =============================================================================


@pytest.mark.property
class TestBriefingPacketAssemblyInvariants:
    """Property 7: Briefing Packet Assembly Invariants.

    **Validates: Requirements 4.2, 4.3, 4.4**

    Tests that for any valid PA request the assembled BriefingPacket satisfies:
    - At most 20 snippets
    - All snippet scores >= 0.5
    - Schema conformance with all required fields non-null
    - verified_evidence_snippets is always a list
    """

    @given(scenario=full_assembly_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_at_most_20_snippets(self, scenario):
        """BriefingPacket contains at most 20 evidence snippets.

        **Validates: Requirements 4.2**

        Regardless of how many chunks the retrieval service returns,
        the assembled packet must never contain more than 20 snippets.
        """
        pa_request, member_state, retrieval_result = scenario

        # Create mock services
        graph_service = AsyncMock()
        graph_service.get_member_active_state = AsyncMock(return_value=member_state)

        retrieval_service = AsyncMock()
        retrieval_service.retrieve = AsyncMock(return_value=retrieval_result)

        service = ContextPlannerService(
            graph_service=graph_service,
            retrieval_service=retrieval_service,
        )

        packet = await service.assemble_briefing_packet(pa_request)

        assert len(packet.verified_evidence_snippets) <= 20, (
            f"BriefingPacket has {len(packet.verified_evidence_snippets)} snippets, "
            f"exceeding the maximum of 20"
        )

    @given(scenario=full_assembly_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_all_snippet_scores_above_threshold(self, scenario):
        """All included snippet scores are >= 0.5.

        **Validates: Requirements 4.2**

        The Context Planner must only include snippets with a minimum
        relevance score threshold of 0.5 in the BriefingPacket.
        """
        pa_request, member_state, retrieval_result = scenario

        graph_service = AsyncMock()
        graph_service.get_member_active_state = AsyncMock(return_value=member_state)

        retrieval_service = AsyncMock()
        retrieval_service.retrieve = AsyncMock(return_value=retrieval_result)

        service = ContextPlannerService(
            graph_service=graph_service,
            retrieval_service=retrieval_service,
        )

        packet = await service.assemble_briefing_packet(pa_request)

        for snippet in packet.verified_evidence_snippets:
            assert snippet.score >= 0.5, (
                f"Snippet '{snippet.chunk_id}' has score {snippet.score} "
                f"below minimum threshold of 0.5"
            )

    @given(scenario=full_assembly_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_required_fields_non_null(self, scenario):
        """BriefingPacket has all required fields non-null.

        **Validates: Requirements 4.3**

        The BriefingPacket must conform to the JSON schema with:
        request_id, member_id, cpt_code, and active_clinical_state
        all present and non-null.
        """
        pa_request, member_state, retrieval_result = scenario

        graph_service = AsyncMock()
        graph_service.get_member_active_state = AsyncMock(return_value=member_state)

        retrieval_service = AsyncMock()
        retrieval_service.retrieve = AsyncMock(return_value=retrieval_result)

        service = ContextPlannerService(
            graph_service=graph_service,
            retrieval_service=retrieval_service,
        )

        packet = await service.assemble_briefing_packet(pa_request)

        # Verify all required fields are present and non-null
        assert packet.request_id is not None and len(packet.request_id) > 0, (
            "request_id must be non-null and non-empty"
        )
        assert packet.member_id is not None and len(packet.member_id) > 0, (
            "member_id must be non-null and non-empty"
        )
        assert packet.cpt_code is not None and len(packet.cpt_code) > 0, (
            "cpt_code must be non-null and non-empty"
        )
        assert packet.active_clinical_state is not None, (
            "active_clinical_state must be non-null"
        )
        assert isinstance(packet.active_clinical_state, MemberActiveState), (
            f"active_clinical_state must be MemberActiveState, "
            f"got {type(packet.active_clinical_state)}"
        )

    @given(scenario=full_assembly_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_verified_evidence_snippets_is_list(self, scenario):
        """verified_evidence_snippets is always a list (may be empty).

        **Validates: Requirements 4.3**

        The BriefingPacket's verified_evidence_snippets field must always
        be a list instance, even when no evidence is found.
        """
        pa_request, member_state, retrieval_result = scenario

        graph_service = AsyncMock()
        graph_service.get_member_active_state = AsyncMock(return_value=member_state)

        retrieval_service = AsyncMock()
        retrieval_service.retrieve = AsyncMock(return_value=retrieval_result)

        service = ContextPlannerService(
            graph_service=graph_service,
            retrieval_service=retrieval_service,
        )

        packet = await service.assemble_briefing_packet(pa_request)

        assert isinstance(packet.verified_evidence_snippets, list), (
            f"verified_evidence_snippets must be a list, "
            f"got {type(packet.verified_evidence_snippets)}"
        )

    @given(scenario=full_assembly_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_request_id_matches_pa_request(self, scenario):
        """BriefingPacket request_id matches the input PA request's request_id.

        **Validates: Requirements 4.3**

        The assembled packet must carry the same request_id from the
        input PA request for correlation.
        """
        pa_request, member_state, retrieval_result = scenario

        graph_service = AsyncMock()
        graph_service.get_member_active_state = AsyncMock(return_value=member_state)

        retrieval_service = AsyncMock()
        retrieval_service.retrieve = AsyncMock(return_value=retrieval_result)

        service = ContextPlannerService(
            graph_service=graph_service,
            retrieval_service=retrieval_service,
        )

        packet = await service.assemble_briefing_packet(pa_request)

        assert packet.request_id == pa_request.request_id, (
            f"Packet request_id '{packet.request_id}' does not match "
            f"PA request_id '{pa_request.request_id}'"
        )
        assert packet.member_id == pa_request.member_id, (
            f"Packet member_id '{packet.member_id}' does not match "
            f"PA member_id '{pa_request.member_id}'"
        )
        assert packet.cpt_code == pa_request.cpt_code, (
            f"Packet cpt_code '{packet.cpt_code}' does not match "
            f"PA cpt_code '{pa_request.cpt_code}'"
        )

    @given(scenario=full_assembly_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_only_cpt_relevant_diagnoses_when_categories_known(self, scenario):
        """Only CPT-relevant diagnoses are included when CPT code has known categories.

        **Validates: Requirements 4.4**

        When the CPT code maps to known clinical condition categories,
        the BriefingPacket should only contain diagnoses relevant to
        those categories (or all if CPT has no known mapping).
        """
        pa_request, member_state, retrieval_result = scenario

        graph_service = AsyncMock()
        graph_service.get_member_active_state = AsyncMock(return_value=member_state)

        retrieval_service = AsyncMock()
        retrieval_service.retrieve = AsyncMock(return_value=retrieval_result)

        service = ContextPlannerService(
            graph_service=graph_service,
            retrieval_service=retrieval_service,
        )

        packet = await service.assemble_briefing_packet(pa_request)

        # If the CPT code has known categories, verify filtering was applied
        categories = CPT_CONDITION_CATEGORIES.get(pa_request.cpt_code, [])

        if categories:
            # Each returned diagnosis should contain at least one category keyword
            # or the CPT code itself in its text fields
            for dx in packet.active_clinical_state.active_diagnoses:
                searchable = " ".join(
                    str(v).lower() for v in dx.values() if isinstance(v, str)
                )
                is_relevant = (
                    pa_request.cpt_code.lower() in searchable
                    or any(cat.lower() in searchable for cat in categories)
                )
                assert is_relevant, (
                    f"Diagnosis {dx} does not appear relevant to CPT code "
                    f"'{pa_request.cpt_code}' with categories {categories}"
                )

    @given(scenario=full_assembly_scenario_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_packet_is_valid_briefing_packet_instance(self, scenario):
        """The assembled result is a valid BriefingPacket Pydantic model.

        **Validates: Requirements 4.3**

        The returned object must be a properly validated BriefingPacket
        instance, confirming schema conformance through Pydantic validators
        (including max 20 snippets and score >= 0.5 checks).
        """
        pa_request, member_state, retrieval_result = scenario

        graph_service = AsyncMock()
        graph_service.get_member_active_state = AsyncMock(return_value=member_state)

        retrieval_service = AsyncMock()
        retrieval_service.retrieve = AsyncMock(return_value=retrieval_result)

        service = ContextPlannerService(
            graph_service=graph_service,
            retrieval_service=retrieval_service,
        )

        packet = await service.assemble_briefing_packet(pa_request)

        # Validate it's a proper BriefingPacket instance
        assert isinstance(packet, BriefingPacket), (
            f"Result must be BriefingPacket, got {type(packet)}"
        )

        # Re-validate by re-constructing (triggers all Pydantic validators)
        revalidated = BriefingPacket.model_validate(packet.model_dump())
        assert revalidated.request_id == packet.request_id
        assert len(revalidated.verified_evidence_snippets) == len(
            packet.verified_evidence_snippets
        )
