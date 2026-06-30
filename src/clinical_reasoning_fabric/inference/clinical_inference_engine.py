"""
Clinical Inference Engine - LLM-powered component that derives implied clinical
conclusions from contextual clues in clinical notes.

Produces inferred SDOH factors, medication adherence risks, and care access barriers
that are implied but not explicitly stated in clinical documentation.

Requirements referenced: 14.1, 14.2, 14.6
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

from clinical_reasoning_fabric.models.core import ScoredChunk
from clinical_reasoning_fabric.models.exceptions import InferenceTimeoutError


# =============================================================================
# Protocols
# =============================================================================


class LLMClient(Protocol):
    """Protocol for LLM client used by the inference engine.

    Any LLM client implementation must provide an async `infer` method
    that accepts a prompt string and returns a response string.
    """

    async def infer(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text."""
        ...


class GraphService(Protocol):
    """Protocol for graph service used to link inferred facts."""

    async def upsert_node(
        self, node_type: str, node_id: str, properties: dict, execution_id: str = None
    ) -> None:
        ...

    async def upsert_relationship(
        self, source_id: str, target_id: str, rel_type: str, properties: dict
    ) -> None:
        ...


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class InferenceHop:
    """A single reasoning step in an inference chain.

    Requirement 14.5: Each hop records source_text, intermediate_conclusion,
    and confidence score.
    """

    hop_number: int  # 1-indexed
    source_text: str  # The text evidence for this hop
    intermediate_conclusion: str  # What was concluded at this hop
    confidence: float  # 0.0 to 1.0 for this individual hop

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Hop confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.hop_number < 1:
            raise ValueError(f"hop_number must be >= 1, got {self.hop_number}")


@dataclass
class InferenceChain:
    """Ordered sequence of reasoning steps from source to conclusion.

    Requirement 14.5: Multi-hop inference chains with intermediate conclusions.
    """

    chain_id: str
    hops: list[InferenceHop]
    cumulative_confidence: float  # Product of all hop confidences
    source_snippet_id: str
    final_conclusion: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.cumulative_confidence <= 1.0:
            raise ValueError(
                f"Cumulative confidence must be between 0.0 and 1.0, got {self.cumulative_confidence}"
            )


@dataclass
class InferredFact:
    """A single inferred clinical conclusion.

    Requirement 14.2: Tagged with inference_type and confidence score.
    """

    fact_id: str
    inference_type: str  # "sdoh_factor" | "medication_adherence_risk" | "care_access_barrier"
    sdoh_category: Optional[str]  # Only for sdoh_factor type, from SDOH_CATEGORIES
    conclusion: str
    confidence: float  # 0.0 to 1.0
    inference_chain: InferenceChain
    source_text_excerpt: str  # Up to 500 characters from source
    inferred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.inference_type not in ClinicalInferenceEngine.INFERENCE_TYPES:
            raise ValueError(
                f"inference_type must be one of {ClinicalInferenceEngine.INFERENCE_TYPES}, "
                f"got '{self.inference_type}'"
            )
        if self.inference_type == "sdoh_factor":
            if self.sdoh_category is None:
                raise ValueError("sdoh_category is required when inference_type is 'sdoh_factor'")
            if self.sdoh_category not in ClinicalInferenceEngine.SDOH_CATEGORIES:
                raise ValueError(
                    f"sdoh_category must be one of {ClinicalInferenceEngine.SDOH_CATEGORIES}, "
                    f"got '{self.sdoh_category}'"
                )
        # Truncate source_text_excerpt to 500 characters
        if len(self.source_text_excerpt) > 500:
            self.source_text_excerpt = self.source_text_excerpt[:500]


@dataclass
class InferenceResult:
    """Result of analyzing a single snippet.

    Requirement 14.1, 14.6: Max 10 inferred facts per snippet.
    """

    snippet_id: str
    inferred_facts: list[InferredFact]  # Max 10 per snippet
    processing_time_ms: int
    depth_used: str  # "shallow" or "deep"
    total_hops_executed: int

    def __post_init__(self) -> None:
        if len(self.inferred_facts) > ClinicalInferenceEngine.MAX_INFERRED_FACTS_PER_SNIPPET:
            raise ValueError(
                f"Max {ClinicalInferenceEngine.MAX_INFERRED_FACTS_PER_SNIPPET} inferred facts "
                f"per snippet, got {len(self.inferred_facts)}"
            )


# =============================================================================
# Clinical Inference Engine
# =============================================================================


class ClinicalInferenceEngine:
    """
    LLM-powered component that derives implied clinical conclusions from
    contextual clues in clinical notes. Produces inferred SDOH factors,
    medication adherence risks, and care access barriers.

    Requirements: 14.1, 14.2, 14.6
    """

    INFERENCE_TYPES = {"sdoh_factor", "medication_adherence_risk", "care_access_barrier"}
    SDOH_CATEGORIES = {
        "housing_instability",
        "transportation_barriers",
        "medication_storage_limitations",
        "food_insecurity",
        "caregiver_availability",
    }
    MAX_INFERRED_FACTS_PER_SNIPPET = 10
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    DEFAULT_DEPTH = "shallow"
    MAX_HOPS = 3
    SNIPPET_TIMEOUT_SECONDS = 15

    def __init__(
        self,
        llm_client: LLMClient,
        graph_service: GraphService,
        confidence_threshold: float = 0.3,
        inference_depth: str = "shallow",
    ):
        """Initialize the Clinical Inference Engine.

        Args:
            llm_client: LLM client implementing the LLMClient protocol.
            graph_service: Graph service for linking inferred facts to Neo4j.
            confidence_threshold: Minimum confidence score for including inferred facts.
                Defaults to 0.3.
            inference_depth: Inference depth - 'shallow' (1-hop) or 'deep' (up to 3-hop).
                Defaults to 'shallow'.
        """
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        if inference_depth not in ("shallow", "deep"):
            raise ValueError(
                f"inference_depth must be 'shallow' or 'deep', got '{inference_depth}'"
            )

        self.llm = llm_client
        self.graph = graph_service
        self.confidence_threshold = confidence_threshold
        self.depth = inference_depth

    async def analyze_snippet(
        self, snippet: ScoredChunk, member_id: str
    ) -> InferenceResult:
        """Analyze a single clinical note snippet for implied conclusions.

        Must complete within 15 seconds. Produces up to 10 inferred facts,
        each tagged with inference_type and confidence score.

        Args:
            snippet: The clinical note snippet to analyze.
            member_id: The member ID for graph linking context.

        Returns:
            InferenceResult with up to 10 inferred facts.

        Raises:
            InferenceTimeoutError: If processing exceeds 15 seconds.
        """
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._perform_analysis(snippet, member_id),
                timeout=self.SNIPPET_TIMEOUT_SECONDS,
            )
            return result
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(
                reason=f"Inference exceeded {self.SNIPPET_TIMEOUT_SECONDS}s timeout for snippet",
                snippet_id=snippet.chunk_id,
                timeout_seconds=self.SNIPPET_TIMEOUT_SECONDS,
            )

    async def _perform_analysis(
        self, snippet: ScoredChunk, member_id: str
    ) -> InferenceResult:
        """Internal analysis logic wrapped by the timeout in analyze_snippet."""
        start_time = time.time()

        # Build prompt for LLM inference
        prompt = self._build_analysis_prompt(snippet.text, self.depth)

        # Call LLM
        llm_response = await self.llm.infer(prompt)

        # Parse LLM response into inferred facts
        raw_facts = self._parse_llm_response(llm_response, snippet.chunk_id)

        # Apply threshold filter
        filtered_facts = self.apply_threshold_filter(raw_facts)

        # Enforce max 10 facts per snippet
        if len(filtered_facts) > self.MAX_INFERRED_FACTS_PER_SNIPPET:
            # Keep the highest-confidence facts
            filtered_facts.sort(key=lambda f: f.confidence, reverse=True)
            filtered_facts = filtered_facts[: self.MAX_INFERRED_FACTS_PER_SNIPPET]

        # Calculate total hops
        total_hops = sum(
            len(fact.inference_chain.hops) for fact in filtered_facts
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return InferenceResult(
            snippet_id=snippet.chunk_id,
            inferred_facts=filtered_facts,
            processing_time_ms=processing_time_ms,
            depth_used=self.depth,
            total_hops_executed=total_hops,
        )

    async def derive_sdoh_factors(
        self, text: str, depth: str = None
    ) -> list[InferredFact]:
        """Derive SDOH factors from text using LLM reasoning.

        Args:
            text: The clinical note text to analyze for SDOH factors.
            depth: Inference depth override. 'shallow' = 1-hop direct implications only.
                   'deep' = up to 3-hop inference chains. Defaults to engine's configured depth.

        Returns:
            List of InferredFact with confidence >= threshold, filtered to only
            sdoh_factor inference_type.
        """
        effective_depth = depth if depth in ("shallow", "deep") else self.depth

        prompt = self._build_sdoh_prompt(text, effective_depth)

        llm_response = await self.llm.infer(prompt)

        # Parse response and filter to only SDOH factors
        raw_facts = self._parse_llm_response(
            llm_response, source_snippet_id=f"sdoh-{uuid.uuid4().hex[:8]}"
        )

        # Filter to only sdoh_factor types
        sdoh_facts = [f for f in raw_facts if f.inference_type == "sdoh_factor"]

        # Apply threshold filter
        filtered_facts = self.apply_threshold_filter(sdoh_facts)

        # Enforce max 10 facts
        if len(filtered_facts) > self.MAX_INFERRED_FACTS_PER_SNIPPET:
            filtered_facts.sort(key=lambda f: f.confidence, reverse=True)
            filtered_facts = filtered_facts[: self.MAX_INFERRED_FACTS_PER_SNIPPET]

        return filtered_facts

    def compute_chain_confidence(self, chain: InferenceChain) -> float:
        """Compute cumulative confidence as product of individual hop scores.

        cumulative = hop_1_confidence * hop_2_confidence * ... * hop_n_confidence

        Args:
            chain: The inference chain to compute confidence for.

        Returns:
            The cumulative confidence score (product of all hop confidences).
        """
        confidence = 1.0
        for hop in chain.hops:
            confidence *= hop.confidence
        return confidence

    def apply_threshold_filter(self, facts: list[InferredFact]) -> list[InferredFact]:
        """Discard any inferred fact with confidence below threshold.

        Only facts >= self.confidence_threshold are returned.

        Args:
            facts: List of inferred facts to filter.

        Returns:
            Filtered list containing only facts with confidence >= threshold.
        """
        return [f for f in facts if f.confidence >= self.confidence_threshold]

    def _build_analysis_prompt(self, text: str, depth: str) -> str:
        """Build the LLM prompt for clinical note analysis.

        Args:
            text: The clinical note text to analyze.
            depth: The inference depth ('shallow' or 'deep').

        Returns:
            Formatted prompt string for the LLM.
        """
        depth_instruction = (
            "Produce only direct single-hop implications from the text."
            if depth == "shallow"
            else "Produce multi-hop reasoning chains (up to 3 hops) connecting "
            "observations to deeper clinical implications."
        )

        return f"""Analyze the following clinical note snippet for implied clinical conclusions.
Identify Social Determinants of Health (SDOH) factors, medication adherence risks,
and care access barriers that are IMPLIED but not explicitly stated.

SDOH Categories to detect:
- housing_instability
- transportation_barriers
- medication_storage_limitations
- food_insecurity
- caregiver_availability

Inference Types:
- sdoh_factor: Social determinant of health factor
- medication_adherence_risk: Risk of medication non-adherence
- care_access_barrier: Barrier to accessing care

Depth Instruction: {depth_instruction}

Clinical Note Snippet:
---
{text}
---

Respond with a JSON array of inferred facts. Each fact should have:
- inference_type: one of "sdoh_factor", "medication_adherence_risk", "care_access_barrier"
- sdoh_category: (only for sdoh_factor) one of the SDOH categories listed above
- conclusion: a clear statement of the implied conclusion
- confidence: a float between 0.0 and 1.0
- hops: array of reasoning steps, each with source_text, intermediate_conclusion, and confidence

Example response format:
[
  {{
    "inference_type": "sdoh_factor",
    "sdoh_category": "transportation_barriers",
    "conclusion": "Patient likely faces transportation barriers to appointments",
    "confidence": 0.75,
    "hops": [
      {{
        "source_text": "Patient mentions relying on bus to get to clinic",
        "intermediate_conclusion": "Patient lacks personal transportation",
        "confidence": 0.75
      }}
    ]
  }}
]

Return ONLY the JSON array. Maximum 10 facts."""

    def _build_sdoh_prompt(self, text: str, depth: str) -> str:
        """Build the LLM prompt specifically for SDOH factor derivation.

        Args:
            text: The clinical note text to analyze.
            depth: The inference depth ('shallow' or 'deep').

        Returns:
            Formatted prompt string for the LLM.
        """
        depth_instruction = (
            "Produce only direct single-hop implications from the text."
            if depth == "shallow"
            else "Produce multi-hop reasoning chains (up to 3 hops) connecting "
            "observations to deeper SDOH implications."
        )

        return f"""Analyze the following clinical text specifically for Social Determinants of Health (SDOH) factors.
Focus ONLY on implied SDOH factors that are not explicitly stated.

SDOH Categories to detect:
- housing_instability: Signs of unstable housing, homelessness, frequent moves
- transportation_barriers: Difficulty getting to appointments, no personal vehicle, reliance on public transit
- medication_storage_limitations: Unable to store medications properly (temperature, security)
- food_insecurity: Difficulty affording or accessing nutritious food
- caregiver_availability: Lack of support system, isolation, no one to help with care

Depth Instruction: {depth_instruction}

Clinical Text:
---
{text}
---

Respond with a JSON array of inferred SDOH facts. Each fact should have:
- inference_type: "sdoh_factor"
- sdoh_category: one of the SDOH categories listed above
- conclusion: a clear statement of the implied SDOH factor
- confidence: a float between 0.0 and 1.0
- hops: array of reasoning steps, each with source_text, intermediate_conclusion, and confidence

Return ONLY the JSON array. Maximum 10 facts."""

    def _parse_llm_response(
        self, response: str, source_snippet_id: str
    ) -> list[InferredFact]:
        """Parse the LLM response JSON into InferredFact instances.

        Args:
            response: The raw LLM response string (expected to be JSON).
            source_snippet_id: The source snippet ID for building inference chains.

        Returns:
            List of parsed InferredFact instances.
        """
        try:
            # Try to parse the response as JSON
            data = json.loads(response.strip())
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            return []

        if not isinstance(data, list):
            return []

        facts: list[InferredFact] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                fact = self._build_fact_from_dict(item, source_snippet_id)
                if fact is not None:
                    facts.append(fact)
            except (ValueError, KeyError, TypeError):
                # Skip malformed entries
                continue

        return facts

    def _build_fact_from_dict(
        self, item: dict, source_snippet_id: str
    ) -> Optional[InferredFact]:
        """Build an InferredFact from a parsed JSON dictionary.

        Args:
            item: Dictionary from parsed LLM response.
            source_snippet_id: Source snippet ID for the inference chain.

        Returns:
            InferredFact instance, or None if the item is invalid.
        """
        inference_type = item.get("inference_type", "")
        if inference_type not in self.INFERENCE_TYPES:
            return None

        sdoh_category = item.get("sdoh_category")
        if inference_type == "sdoh_factor":
            if sdoh_category not in self.SDOH_CATEGORIES:
                return None

        conclusion = item.get("conclusion", "")
        if not conclusion:
            return None

        confidence = item.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            return None
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            return None

        # Parse hops
        hops_data = item.get("hops", [])
        if not isinstance(hops_data, list) or len(hops_data) == 0:
            # Default to a single hop with the conclusion
            hops_data = [
                {
                    "source_text": conclusion,
                    "intermediate_conclusion": conclusion,
                    "confidence": confidence,
                }
            ]

        # Enforce max hops
        hops_data = hops_data[: self.MAX_HOPS]

        hops: list[InferenceHop] = []
        for i, hop_data in enumerate(hops_data):
            if not isinstance(hop_data, dict):
                continue
            hop_confidence = hop_data.get("confidence", confidence)
            if not isinstance(hop_confidence, (int, float)):
                hop_confidence = confidence
            hop_confidence = max(0.0, min(1.0, float(hop_confidence)))

            hops.append(
                InferenceHop(
                    hop_number=i + 1,
                    source_text=str(hop_data.get("source_text", ""))[:500],
                    intermediate_conclusion=str(
                        hop_data.get("intermediate_conclusion", "")
                    ),
                    confidence=hop_confidence,
                )
            )

        if not hops:
            return None

        # Compute chain confidence as product of hop confidences
        chain_id = f"chain-{uuid.uuid4().hex[:8]}"
        chain = InferenceChain(
            chain_id=chain_id,
            hops=hops,
            cumulative_confidence=self._compute_hops_confidence(hops),
            source_snippet_id=source_snippet_id,
            final_conclusion=conclusion,
        )

        # Use chain cumulative confidence as the fact confidence
        fact_confidence = chain.cumulative_confidence

        source_text_excerpt = ""
        if hops:
            source_text_excerpt = hops[0].source_text[:500]

        return InferredFact(
            fact_id=f"fact-{uuid.uuid4().hex[:8]}",
            inference_type=inference_type,
            sdoh_category=sdoh_category if inference_type == "sdoh_factor" else None,
            conclusion=conclusion,
            confidence=fact_confidence,
            inference_chain=chain,
            source_text_excerpt=source_text_excerpt,
            inferred_at=datetime.now(timezone.utc),
        )

    async def link_to_graph(self, member_id: str, fact: InferredFact) -> None:
        """Create INFERRED_FROM relationship in Neo4j linking the inferred
        SDOH factor to its source evidence with full chain metadata.

        Only links facts with confidence >= self.confidence_threshold.
        Creates an SDOH_Factor node with origin="inferred" and links it
        to an EvidenceSource node via INFERRED_FROM relationship.

        Args:
            member_id: The member ID for context (used in provenance).
            fact: The inferred fact to link to the graph.

        Requirements referenced: 14.3
        """
        # Only link facts that meet the confidence threshold
        if fact.confidence < self.confidence_threshold:
            return

        # Generate a unique SDOH ID for the node
        sdoh_id = f"sdoh-inferred-{uuid.uuid4().hex[:12]}"

        # Create SDOH_Factor node with origin="inferred"
        sdoh_properties = {
            "type": fact.sdoh_category,
            "origin": "inferred",
            "confidence": fact.confidence,
            "conclusion": fact.conclusion,
            "member_id": member_id,
            "inferred_at": fact.inferred_at.isoformat(),
        }
        await self.graph.upsert_node(
            node_type="SDOH_Factor",
            node_id=sdoh_id,
            properties=sdoh_properties,
        )

        # Serialize the inference chain to JSON
        inference_chain_json = json.dumps({
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
        })

        # Create INFERRED_FROM relationship from SDOH_Factor to EvidenceSource
        relationship_properties = {
            "source_text": fact.source_text_excerpt,
            "inference_chain_json": inference_chain_json,
            "confidence": fact.confidence,
            "inferred_at": fact.inferred_at.isoformat(),
        }
        await self.graph.upsert_relationship(
            source_id=sdoh_id,
            target_id=fact.inference_chain.source_snippet_id,
            rel_type="INFERRED_FROM",
            properties=relationship_properties,
        )

    def _compute_hops_confidence(self, hops: list[InferenceHop]) -> float:
        """Compute cumulative confidence as product of hop confidences.

        Args:
            hops: List of InferenceHop instances.

        Returns:
            Product of all hop confidence scores.
        """
        confidence = 1.0
        for hop in hops:
            confidence *= hop.confidence
        return confidence
