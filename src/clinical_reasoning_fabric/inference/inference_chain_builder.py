"""
Inference Chain Builder - Dedicated builder for constructing inference chains
with configurable depth (shallow vs deep).

Shallow depth: produces exactly 1-hop chains (direct single-hop implications).
Deep depth: produces multi-hop chains (up to 3 hops).
No chain ever exceeds 3 hops regardless of configuration or input.

Requirements referenced: 14.4, 14.5
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    InferenceChain,
    InferenceHop,
)


# Maximum number of hops allowed in any inference chain, regardless of depth config.
MAX_CHAIN_HOPS = 3

# Default inference depth.
DEFAULT_DEPTH = "shallow"


class InferenceChainBuilder:
    """Builder for constructing inference chains with configurable depth.

    Supports two depth modes:
    - "shallow": produces exactly 1-hop chains (direct single-hop implications only)
    - "deep": produces multi-hop chains with up to 3 hops

    No chain ever exceeds 3 hops regardless of configuration or input data.

    Requirements: 14.4, 14.5
    """

    MAX_HOPS = MAX_CHAIN_HOPS

    def __init__(self, depth: str = DEFAULT_DEPTH):
        """Initialize the builder with a depth configuration.

        Args:
            depth: Inference depth mode. Must be "shallow" or "deep".
                   Defaults to "shallow".

        Raises:
            ValueError: If depth is not "shallow" or "deep".
        """
        if depth not in ("shallow", "deep"):
            raise ValueError(f"depth must be 'shallow' or 'deep', got '{depth}'")
        self.depth = depth

    def build_shallow_chain(
        self,
        source_text: str,
        conclusion: str,
        confidence: float,
        source_snippet_id: str = "",
    ) -> InferenceChain:
        """Build a shallow (1-hop) inference chain.

        Creates a chain with exactly one hop representing a direct implication
        from source text to conclusion.

        Args:
            source_text: The source text evidence for the single hop.
            conclusion: The conclusion derived from the source text.
            confidence: Confidence score for this implication (0.0 to 1.0).
            source_snippet_id: Optional identifier for the source snippet.

        Returns:
            InferenceChain with exactly 1 hop.

        Raises:
            ValueError: If confidence is not between 0.0 and 1.0.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        hop = InferenceHop(
            hop_number=1,
            source_text=source_text,
            intermediate_conclusion=conclusion,
            confidence=confidence,
        )

        chain_id = f"chain-{uuid.uuid4().hex[:8]}"
        return InferenceChain(
            chain_id=chain_id,
            hops=[hop],
            cumulative_confidence=confidence,
            source_snippet_id=source_snippet_id,
            final_conclusion=conclusion,
        )

    def build_deep_chain(
        self,
        hops_data: list[dict],
        source_snippet_id: str = "",
        final_conclusion: str = "",
    ) -> InferenceChain:
        """Build a deep (multi-hop) inference chain from hop data.

        Creates a chain with up to 3 hops. If more than 3 hops are provided,
        only the first 3 are used. The 3-hop maximum is enforced regardless
        of input size.

        Each hop dict should contain:
            - source_text (str): The text evidence for this hop.
            - intermediate_conclusion (str): What was concluded at this hop.
            - confidence (float): Confidence score for this hop (0.0 to 1.0).

        Args:
            hops_data: List of dictionaries describing each reasoning hop.
                       Truncated to MAX_HOPS (3) if more are provided.
            source_snippet_id: Optional identifier for the source snippet.
            final_conclusion: The final conclusion of the chain. If empty,
                              uses the last hop's intermediate_conclusion.

        Returns:
            InferenceChain with 1 to 3 hops and cumulative confidence
            computed as the product of individual hop confidences.

        Raises:
            ValueError: If hops_data is empty or contains invalid confidence values.
        """
        if not hops_data:
            raise ValueError("hops_data must contain at least one hop")

        # Enforce maximum 3 hops regardless of input
        truncated_hops_data = hops_data[: self.MAX_HOPS]

        hops: list[InferenceHop] = []
        for i, hop_data in enumerate(truncated_hops_data):
            if not isinstance(hop_data, dict):
                raise ValueError(f"Each hop must be a dictionary, got {type(hop_data)} at index {i}")

            source_text = str(hop_data.get("source_text", ""))
            intermediate_conclusion = str(hop_data.get("intermediate_conclusion", ""))
            confidence = hop_data.get("confidence", 0.0)

            if not isinstance(confidence, (int, float)):
                raise ValueError(
                    f"Hop confidence must be a number, got {type(confidence)} at hop {i + 1}"
                )
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Hop confidence must be between 0.0 and 1.0, got {confidence} at hop {i + 1}"
                )

            hops.append(
                InferenceHop(
                    hop_number=i + 1,
                    source_text=source_text,
                    intermediate_conclusion=intermediate_conclusion,
                    confidence=confidence,
                )
            )

        if not hops:
            raise ValueError("No valid hops could be constructed from the input data")

        # Compute cumulative confidence as product of hop confidences
        cumulative_confidence = self._compute_cumulative_confidence(hops)

        # Determine final conclusion
        effective_final_conclusion = final_conclusion or hops[-1].intermediate_conclusion

        chain_id = f"chain-{uuid.uuid4().hex[:8]}"
        return InferenceChain(
            chain_id=chain_id,
            hops=hops,
            cumulative_confidence=cumulative_confidence,
            source_snippet_id=source_snippet_id,
            final_conclusion=effective_final_conclusion,
        )

    def build_chain(
        self,
        hops_data: list[dict],
        source_snippet_id: str = "",
        final_conclusion: str = "",
    ) -> InferenceChain:
        """Build an inference chain respecting the configured depth.

        For "shallow" depth: uses only the first hop from hops_data to create
        a 1-hop chain, regardless of how many hops are provided.

        For "deep" depth: uses up to 3 hops from hops_data.

        Args:
            hops_data: List of hop dictionaries.
            source_snippet_id: Optional source snippet identifier.
            final_conclusion: Optional final conclusion override.

        Returns:
            InferenceChain respecting the configured depth.

        Raises:
            ValueError: If hops_data is empty or contains invalid data.
        """
        if not hops_data:
            raise ValueError("hops_data must contain at least one hop")

        if self.depth == "shallow":
            # Shallow: use only the first hop
            first_hop = hops_data[0]
            if not isinstance(first_hop, dict):
                raise ValueError(f"Each hop must be a dictionary, got {type(first_hop)}")

            source_text = str(first_hop.get("source_text", ""))
            conclusion = final_conclusion or str(first_hop.get("intermediate_conclusion", ""))
            confidence = first_hop.get("confidence", 0.0)

            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Hop confidence must be a number, got {type(confidence)}")
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Hop confidence must be between 0.0 and 1.0, got {confidence}")

            return self.build_shallow_chain(
                source_text=source_text,
                conclusion=conclusion,
                confidence=confidence,
                source_snippet_id=source_snippet_id,
            )
        else:
            # Deep: use up to 3 hops
            return self.build_deep_chain(
                hops_data=hops_data,
                source_snippet_id=source_snippet_id,
                final_conclusion=final_conclusion,
            )

    @staticmethod
    def _compute_cumulative_confidence(hops: list[InferenceHop]) -> float:
        """Compute cumulative confidence as product of individual hop confidences.

        Args:
            hops: List of InferenceHop instances.

        Returns:
            Product of all hop confidence scores.
        """
        confidence = 1.0
        for hop in hops:
            confidence *= hop.confidence
        return confidence
