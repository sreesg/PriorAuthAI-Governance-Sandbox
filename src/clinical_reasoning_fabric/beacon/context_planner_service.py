"""
Context Planner Service — BEACON Layer 2 for Briefing Packet assembly.

Assembles Briefing Packets by querying the Causal Ontology Graph for member
active clinical state and the Qdrant vector store for relevant evidence snippets.
Integrates the Clinical Inference Engine to derive implied clinical conclusions
from retrieved snippets. Applies a 30-second timeout to the entire assembly
process and filters results to only information relevant to the specific PA request.

Requirements referenced: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 14.1, 14.8, 14.9
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from clinical_reasoning_fabric.models.core import (
    BriefingPacket,
    MemberActiveState,
    RetrievalResult,
    ScoredChunk,
)
from clinical_reasoning_fabric.models.exceptions import MemberNotFoundError

logger = logging.getLogger(__name__)

# Timeout for complete Briefing Packet assembly (Requirement 4.6)
ASSEMBLY_TIMEOUT_SECONDS = 30

# Overall timeout for all inference engine calls (Requirement 14.9)
INFERENCE_TIMEOUT_SECONDS = 30


@dataclass
class PARequest:
    """Prior Authorization request input for Briefing Packet assembly.

    Contains the identifiers and context needed to assemble a Briefing Packet.
    """

    request_id: str
    member_id: str
    cpt_code: str
    clinical_context: Optional[str] = None


# CPT code to clinical condition category mapping for relevance filtering.
# In production this would be loaded from a clinical terminology service.
CPT_CONDITION_CATEGORIES: dict[str, list[str]] = {
    # Radiology
    "72148": ["lumbar", "spine", "back pain", "radiculopathy"],
    "72141": ["cervical", "neck", "spine"],
    "70553": ["brain", "neurological", "headache"],
    # Oncology
    "78816": ["oncology", "cancer", "tumor", "metastasis", "pet"],
    "77386": ["radiation", "oncology", "cancer"],
    # Surgery
    "29881": ["knee", "meniscus", "orthopedic", "joint"],
    "27447": ["knee", "arthroplasty", "joint replacement"],
    "63030": ["lumbar", "spine", "laminotomy", "disc"],
    # Specialty Rx
    "96413": ["infusion", "biologic", "immunotherapy"],
    "J0585": ["dupixent", "atopic dermatitis", "asthma", "biologic"],
}


class ContextPlannerService:
    """Assembles Briefing Packets by querying graph and vector stores.

    Coordinates between the Causal Ontology Graph (Neo4j), hybrid retrieval
    service (Qdrant), and Clinical Inference Engine to assemble a pre-packaged
    context for bounded agent reasoning.

    Requirements:
        4.1: Query graph for member active clinical state
        4.2: Query Qdrant for evidence snippets (max 20, min score 0.5)
        4.3: Package into BriefingPacket with all required fields
        4.4: Filter to CPT-relevant diagnoses, medications, and evidence
        4.5: Raise MemberNotFoundError if member not found
        4.6: Complete within 30 seconds or log timeout error
        4.7: Handle zero evidence with empty snippets and no_evidence_found flag
        14.1: Invoke Clinical Inference Engine on each retrieved snippet
        14.8: Present inferred facts in distinct section with confidence and chain
        14.9: Proceed without inferred facts if engine unavailable, set degraded_inference
    """

    def __init__(
        self,
        graph_service: Any,
        retrieval_service: Any,
        inference_engine: Optional[Any] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        """Initialize ContextPlannerService.

        Args:
            graph_service: CausalOntologyGraphService for Neo4j queries.
            retrieval_service: HybridRetrievalService for Qdrant queries.
            inference_engine: Optional ClinicalInferenceEngine for deriving
                implied conclusions from retrieved snippets. If None, inference
                is skipped and degraded_inference is set.
            confidence_threshold: Minimum confidence for including inferred facts.
                Defaults to 0.3.
        """
        self.graph = graph_service
        self.retrieval = retrieval_service
        self.inference_engine = inference_engine
        self.confidence_threshold = confidence_threshold

    async def assemble_briefing_packet(self, pa_request: PARequest) -> BriefingPacket:
        """Assemble a Briefing Packet within a 30-second timeout.

        Steps:
            1. Query Neo4j for member active clinical state
            2. Query Qdrant for relevant evidence snippets (max 20, min score 0.5)
            3. Filter to relevant diagnoses/medications for the CPT code
            4. Package into BriefingPacket schema

        Args:
            pa_request: The PA request containing member_id, cpt_code, etc.

        Returns:
            BriefingPacket with assembled clinical context.

        Raises:
            MemberNotFoundError: If the member is not found in the graph.
            asyncio.TimeoutError: If assembly exceeds 30 seconds.
        """
        try:
            return await asyncio.wait_for(
                self._assemble(pa_request),
                timeout=ASSEMBLY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Briefing Packet assembly for request %s (member %s) exceeded "
                "%d-second timeout",
                pa_request.request_id,
                pa_request.member_id,
                ASSEMBLY_TIMEOUT_SECONDS,
            )
            raise

    async def _assemble(self, pa_request: PARequest) -> BriefingPacket:
        """Internal assembly logic without timeout wrapper.

        Extended assembly flow (Requirements 14.1, 14.8, 14.9):
            1. Query Neo4j for member active state
            2. Query Qdrant for relevant evidence snippets
            3. For each snippet, invoke Clinical_Inference_Engine (30s overall timeout)
            4. Filter inferences by confidence threshold
            5. Package into BriefingPacket with separate inferred_facts section

        If inference engine unavailable: proceed without inferred facts,
        set degraded_inference=True, log warning.

        Args:
            pa_request: The PA request.

        Returns:
            Assembled BriefingPacket.

        Raises:
            MemberNotFoundError: If the member is not found.
        """
        # Step 1: Query Neo4j for member active clinical state (Requirement 4.1)
        member_state = await self._get_member_state(pa_request.member_id)

        # Step 2: Query Qdrant for relevant evidence snippets (Requirement 4.2)
        retrieval_result = await self._retrieve_evidence(pa_request)

        # Step 3: Filter to CPT-relevant content (Requirement 4.4)
        filtered_state = self._filter_relevant_state(member_state, pa_request.cpt_code)
        filtered_snippets = self._filter_relevant_snippets(
            retrieval_result.verified_chunks, pa_request.cpt_code
        )

        # Step 4: Invoke Clinical Inference Engine on each snippet (Req 14.1, 14.8, 14.9)
        inferred_facts, degraded_inference = await self._run_inference(
            filtered_snippets, pa_request.member_id
        )

        # Determine if no evidence was found (Requirement 4.7)
        no_evidence_found = retrieval_result.no_evidence_found or len(filtered_snippets) == 0

        # Step 5: Package into BriefingPacket (Requirement 4.3)
        return BriefingPacket(
            request_id=pa_request.request_id,
            member_id=pa_request.member_id,
            cpt_code=pa_request.cpt_code,
            active_clinical_state=filtered_state,
            verified_evidence_snippets=filtered_snippets,
            inferred_facts=inferred_facts,
            no_evidence_found=no_evidence_found,
            degraded_inference=degraded_inference,
        )

    async def _run_inference(
        self, snippets: list[ScoredChunk], member_id: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Run Clinical Inference Engine on each snippet with overall 30s timeout.

        For each retrieved snippet, invokes the inference engine to derive implied
        clinical conclusions. Collects all inferred facts and filters by confidence
        threshold. If the inference engine is unavailable or fails, proceeds without
        inferred facts and sets degraded_inference=True.

        Args:
            snippets: The filtered evidence snippets to analyze.
            member_id: The member ID for inference context.

        Returns:
            Tuple of (inferred_facts_as_dicts, degraded_inference_flag).

        Requirements:
            14.1: Analyze each retrieved snippet for implied conclusions
            14.8: Present inferred facts with confidence score and inference chain
            14.9: On failure/unavailability, set degraded_inference=True, log warning
        """
        # If no inference engine configured, proceed in degraded mode
        if self.inference_engine is None:
            logger.warning(
                "Clinical Inference Engine not configured. Proceeding without "
                "inferred facts (degraded_inference=True)."
            )
            return [], True

        # If no snippets to analyze, no inference needed
        if not snippets:
            return [], False

        try:
            # Apply 30-second overall timeout for all inference calls (Req 14.9)
            inferred_facts = await asyncio.wait_for(
                self._analyze_all_snippets(snippets, member_id),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )

            # Filter by confidence threshold
            filtered_facts = [
                f for f in inferred_facts
                if f.get("confidence", 0.0) >= self.confidence_threshold
            ]

            return filtered_facts, False

        except asyncio.TimeoutError:
            logger.warning(
                "Clinical Inference Engine exceeded %ds overall timeout for "
                "member '%s'. Proceeding without inferred facts (degraded_inference=True).",
                INFERENCE_TIMEOUT_SECONDS,
                member_id,
            )
            return [], True

        except Exception as e:
            logger.warning(
                "Clinical Inference Engine unavailable or failed for member '%s': %s. "
                "Proceeding without inferred facts (degraded_inference=True).",
                member_id,
                str(e),
            )
            return [], True

    async def _analyze_all_snippets(
        self, snippets: list[ScoredChunk], member_id: str
    ) -> list[dict[str, Any]]:
        """Analyze all snippets sequentially using the inference engine.

        For each snippet, invokes analyze_snippet and collects results.
        Individual snippet failures are logged but do not halt the process.

        Args:
            snippets: The evidence snippets to analyze.
            member_id: The member ID for context.

        Returns:
            List of inferred facts as dictionaries with confidence and chain info.
        """
        all_inferred_facts: list[dict[str, Any]] = []

        for snippet in snippets:
            try:
                result = await self.inference_engine.analyze_snippet(snippet, member_id)

                # Convert inferred facts to dict format for BriefingPacket
                # (Requirement 14.8: each fact includes confidence score and inference chain)
                for fact in result.inferred_facts:
                    fact_dict = {
                        "fact_id": fact.fact_id,
                        "inference_type": fact.inference_type,
                        "sdoh_category": fact.sdoh_category,
                        "conclusion": fact.conclusion,
                        "confidence": fact.confidence,
                        "source_text_excerpt": fact.source_text_excerpt,
                        "inferred_at": fact.inferred_at.isoformat(),
                        "inference_chain": {
                            "chain_id": fact.inference_chain.chain_id,
                            "hops": [
                                {
                                    "hop_number": hop.hop_number,
                                    "source_text": hop.source_text,
                                    "intermediate_conclusion": hop.intermediate_conclusion,
                                    "confidence": hop.confidence,
                                }
                                for hop in fact.inference_chain.hops
                            ],
                            "cumulative_confidence": fact.inference_chain.cumulative_confidence,
                            "final_conclusion": fact.inference_chain.final_conclusion,
                        },
                        "source_snippet_id": fact.inference_chain.source_snippet_id,
                    }
                    all_inferred_facts.append(fact_dict)

            except Exception as e:
                # Individual snippet failure: log and continue (Req 14.9)
                logger.warning(
                    "Inference failed for snippet '%s': %s. Skipping.",
                    snippet.chunk_id,
                    str(e),
                )
                continue

        return all_inferred_facts

    async def _get_member_state(self, member_id: str) -> MemberActiveState:
        """Query the graph service for member active state.

        Args:
            member_id: The member identifier.

        Returns:
            MemberActiveState from the graph.

        Raises:
            MemberNotFoundError: If the member does not exist in the graph.
        """
        try:
            return await self.graph.get_member_active_state(member_id)
        except MemberNotFoundError:
            logger.error(
                "Member '%s' not found in Causal Ontology Graph. Halting processing.",
                member_id,
            )
            raise

    async def _retrieve_evidence(self, pa_request: PARequest) -> RetrievalResult:
        """Retrieve relevant evidence from Qdrant using CPT code and clinical context.

        Constructs a retrieval query from the CPT code and optional clinical context,
        requesting max 20 snippets with a minimum relevance score of 0.5.

        Args:
            pa_request: The PA request with cpt_code and optional clinical_context.

        Returns:
            RetrievalResult from the hybrid retrieval service.
        """
        query = self._build_retrieval_query(pa_request)
        return await self.retrieval.retrieve(query=query, top_k=20, min_score=0.5)

    def _build_retrieval_query(self, pa_request: PARequest) -> str:
        """Build a retrieval query string from the PA request.

        Combines CPT code with clinical context and associated condition categories
        to form a comprehensive retrieval query.

        Args:
            pa_request: The PA request.

        Returns:
            A query string for hybrid retrieval.
        """
        parts = [f"CPT {pa_request.cpt_code}"]

        # Add clinical context if provided
        if pa_request.clinical_context:
            parts.append(pa_request.clinical_context)

        # Add associated condition categories
        categories = self._get_condition_categories(pa_request.cpt_code)
        if categories:
            parts.append(" ".join(categories))

        return " ".join(parts)

    def _get_condition_categories(self, cpt_code: str) -> list[str]:
        """Get clinical condition categories associated with a CPT code.

        Args:
            cpt_code: The CPT procedure code.

        Returns:
            List of associated condition category keywords.
        """
        return CPT_CONDITION_CATEGORIES.get(cpt_code, [])

    def _filter_relevant_state(
        self, member_state: MemberActiveState, cpt_code: str
    ) -> MemberActiveState:
        """Filter member state to only CPT-relevant diagnoses and medications.

        Restricts the active clinical state to diagnoses, medications, and SDOH
        factors that match the requested CPT code or its associated clinical
        condition categories (Requirement 4.4).

        Args:
            member_state: The full active clinical state.
            cpt_code: The CPT code to filter by.

        Returns:
            MemberActiveState with only relevant entries.
        """
        categories = self._get_condition_categories(cpt_code)

        # If no known categories for this CPT code, return full state
        # (we don't want to accidentally filter out relevant info)
        if not categories:
            return member_state

        # Filter diagnoses by category relevance
        filtered_diagnoses = [
            dx for dx in member_state.active_diagnoses
            if self._is_relevant_to_categories(dx, categories, cpt_code)
        ]

        # Filter prescriptions by category relevance
        filtered_prescriptions = [
            rx for rx in member_state.active_prescriptions
            if self._is_relevant_to_categories(rx, categories, cpt_code)
        ]

        return MemberActiveState(
            member_id=member_state.member_id,
            active_diagnoses=filtered_diagnoses,
            active_prescriptions=filtered_prescriptions,
            sdoh_factors=member_state.sdoh_factors,
            governing_policies=member_state.governing_policies,
            last_updated=member_state.last_updated,
        )

    def _filter_relevant_snippets(
        self, snippets: list[ScoredChunk], cpt_code: str
    ) -> list[ScoredChunk]:
        """Filter evidence snippets to those relevant to the CPT code.

        Returns snippets whose text contains the CPT code or any associated
        clinical condition category keywords (Requirement 4.4).

        Snippets that pass the retrieval min score threshold (0.5) but have
        no textual relevance to the CPT code are excluded.

        Args:
            snippets: The verified evidence snippets from retrieval.
            cpt_code: The CPT code to filter by.

        Returns:
            List of relevant ScoredChunk objects (max 20, all scores >= 0.5).
        """
        categories = self._get_condition_categories(cpt_code)

        # If no known categories, return all snippets (retrieval already
        # used the CPT code in the query, so they should be relevant)
        if not categories:
            return snippets[:20]

        relevant = []
        for snippet in snippets:
            if self._snippet_matches_context(snippet, cpt_code, categories):
                relevant.append(snippet)
            if len(relevant) >= 20:
                break

        return relevant

    def _is_relevant_to_categories(
        self, record: dict[str, Any], categories: list[str], cpt_code: str
    ) -> bool:
        """Check if a clinical record is relevant to the given categories or CPT code.

        Examines common fields in the record (description, condition_code,
        medication_name, etc.) for matches against the category keywords or
        the CPT code itself.

        Args:
            record: A diagnosis or prescription record dict.
            categories: List of condition category keywords.
            cpt_code: The CPT code.

        Returns:
            True if the record is relevant, False otherwise.
        """
        # Flatten record values to a searchable text
        searchable_text = " ".join(
            str(v).lower() for v in record.values() if isinstance(v, str)
        )

        # Check if CPT code appears directly
        if cpt_code.lower() in searchable_text:
            return True

        # Check against condition categories
        for category in categories:
            if category.lower() in searchable_text:
                return True

        return False

    def _snippet_matches_context(
        self, snippet: ScoredChunk, cpt_code: str, categories: list[str]
    ) -> bool:
        """Check if a snippet's text is relevant to the CPT code context.

        Args:
            snippet: The scored chunk to check.
            cpt_code: The CPT code.
            categories: Associated condition category keywords.

        Returns:
            True if the snippet text matches the context.
        """
        text_lower = snippet.text.lower()

        # Check CPT code directly
        if cpt_code.lower() in text_lower:
            return True

        # Check condition categories
        for category in categories:
            if category.lower() in text_lower:
                return True

        return False
