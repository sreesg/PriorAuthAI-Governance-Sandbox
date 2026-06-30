"""Core data models, enums, and custom exceptions for the CRF system."""

from clinical_reasoning_fabric.models.exceptions import (
    CRFError,
    BundleValidationError,
    InferenceTimeoutError,
    IngestionError,
    InvalidNamespaceError,
    KMSUnavailableError,
    MemberNotFoundError,
    PIIScrubError,
    TraceRecordingError,
    ToolValidationError,
    UnauthorizedError,
    UnmappableRecordError,
)

from clinical_reasoning_fabric.models.core import (
    AuthResult,
    BriefingPacket,
    CDCEvent,
    ChunkProvenance,
    CriterionAssessment,
    CriterionStatus,
    Disposition,
    DocumentChunk,
    EvidenceBundle,
    EventCheckpoint,
    IngestionResult,
    KMSSignature,
    LineageEntry,
    MemberActiveState,
    RBACPolicy,
    RetrievalResult,
    ScoredChunk,
    TamperAlert,
    ToolDefinition,
    ToolResult,
    TraceCategory,
    TraceEntry,
    VerificationResult,
)

__all__ = [
    # Exceptions
    "CRFError",
    "BundleValidationError",
    "InferenceTimeoutError",
    "IngestionError",
    "InvalidNamespaceError",
    "KMSUnavailableError",
    "MemberNotFoundError",
    "PIIScrubError",
    "TraceRecordingError",
    "ToolValidationError",
    "UnauthorizedError",
    "UnmappableRecordError",
    # Enums
    "TraceCategory",
    "CriterionStatus",
    "VerificationResult",
    "Disposition",
    # Cryptographic Provenance
    "KMSSignature",
    "ChunkProvenance",
    "DocumentChunk",
    "IngestionResult",
    # Retrieval
    "ScoredChunk",
    "TamperAlert",
    "RetrievalResult",
    # Graph State
    "MemberActiveState",
    # CDC Pipeline
    "CDCEvent",
    "EventCheckpoint",
    # Briefing Packet
    "BriefingPacket",
    # Decision
    "CriterionAssessment",
    # Evidence Bundle & Lineage
    "LineageEntry",
    "EvidenceBundle",
    # Trace
    "TraceEntry",
    # Tool Orchestration
    "ToolDefinition",
    "ToolResult",
    # Identity & Auth
    "RBACPolicy",
    "AuthResult",
]
