"""Clinical Inference Engine module.

Provides LLM-powered inference for deriving implied clinical conclusions
from contextual clues in clinical notes.
"""

from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    ClinicalInferenceEngine,
    InferenceChain,
    InferenceHop,
    InferenceResult,
    InferredFact,
    LLMClient,
    GraphService,
)
from clinical_reasoning_fabric.inference.inference_chain_builder import (
    InferenceChainBuilder,
)

__all__ = [
    "ClinicalInferenceEngine",
    "InferenceChain",
    "InferenceChainBuilder",
    "InferenceHop",
    "InferenceResult",
    "InferredFact",
    "LLMClient",
    "GraphService",
]
