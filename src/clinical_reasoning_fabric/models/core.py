"""
Core data models and enums for the Clinical Reasoning Fabric.

Implements all shared dataclasses and enums used across the CRF system
including Axisweave Retrieval Stack, Causal Ontology Graph, BEACON Harness,
and Evidence Bundle production.

Requirements referenced: 8.1, 10.1, 3.1, 4.3
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================


class TraceCategory(str, Enum):
    """Categories for execution trace entries (Requirement 8.1).

    Every trace entry must be categorized as one of these four types.
    """

    AGENT_ACTION = "agent_action"
    TOOL_INVOCATION = "tool_invocation"
    CONTEXT_RETRIEVAL = "context_retrieval"
    DECISION_STEP = "decision_step"


class CriterionStatus(str, Enum):
    """Status of a clinical necessity criterion evaluation (Requirement 9.1)."""

    MET = "MET"
    NOT_MET = "NOT_MET"
    INDETERMINATE = "INDETERMINATE"


class VerificationResult(str, Enum):
    """Result of OPA Challenger Agent verification (Requirement 7.2)."""

    PASS = "PASS"
    FAIL = "FAIL"


class Disposition(str, Enum):
    """Final disposition of a PA request (Requirement 9.2).

    NEVER DENIED — the system only auto-approves or escalates.
    """

    APPROVED = "APPROVED"
    ESCALATED = "ESCALATED"


# =============================================================================
# Cryptographic Provenance Models
# =============================================================================


class KMSSignature(BaseModel):
    """AWS KMS cryptographic signature attached to a document record.

    Requirement 1.4: KMS_Signer signs content hash with AWS KMS asymmetric key.
    """

    key_id: str = Field(..., min_length=1, description="AWS KMS key identifier")
    signature: str = Field(..., min_length=1, description="Base64-encoded signature")
    algorithm: str = Field(
        default="RSASSA_PKCS1_V1_5_SHA_256",
        description="Signing algorithm used",
    )
    signed_at: datetime = Field(..., description="Timestamp when signature was created")

    @field_validator("signature")
    @classmethod
    def signature_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Signature must not be empty or whitespace-only")
        return v


class ChunkProvenance(BaseModel):
    """Provenance metadata for a stored document chunk.

    Requirement 1.5: Every chunk stored with source document_id, content_hash,
    KMS_signature, chunk_index, and ingestion_timestamp.
    """

    document_id: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hex digest")
    kms_signature: KMSSignature
    chunk_index: int = Field(..., ge=0)
    ingestion_timestamp: datetime

    @field_validator("content_hash")
    @classmethod
    def validate_sha256_format(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError("content_hash must be a 64-character SHA-256 hex digest")
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("content_hash must be a valid hexadecimal string")
        return v.lower()


class DocumentChunk(BaseModel):
    """A semantically chunked segment of a clinical document.

    Requirement 1.5: Chunks stored in Qdrant with provenance metadata.
    """

    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    embedding: Optional[list[float]] = Field(default=None, description="Dense vector embedding")
    provenance: ChunkProvenance
    namespace: Optional[str] = Field(default=None, description="Tenant namespace for multi-tenant isolation")
    document_category: Optional[str] = Field(default=None, max_length=64)


class IngestionResult(BaseModel):
    """Result of successful document ingestion pipeline.

    Requirement 1.1-1.5: Full pipeline result with document_id, content_hash,
    signature, chunk_count, and ingestion_timestamp.
    """

    document_id: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=64, max_length=64)
    signature: KMSSignature
    chunk_count: int = Field(..., ge=1)
    ingestion_timestamp: datetime

    @field_validator("content_hash")
    @classmethod
    def validate_sha256_format(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError("content_hash must be a 64-character SHA-256 hex digest")
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("content_hash must be a valid hexadecimal string")
        return v.lower()


# =============================================================================
# Retrieval Models
# =============================================================================


class ScoredChunk(BaseModel):
    """A document chunk with retrieval score from hybrid search.

    Requirement 2.3: Chunks ranked by Reciprocal Rank Fusion score.
    Requirement 13.5/13.7: Namespace metadata for multi-tenant isolation.
    """

    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0, description="RRF or similarity score")
    provenance: ChunkProvenance
    dense_rank: Optional[int] = Field(default=None, ge=1, description="Rank in dense search results")
    sparse_rank: Optional[int] = Field(default=None, ge=1, description="Rank in BM25 sparse results")
    namespace: Optional[str] = Field(
        default=None,
        description="Namespace this chunk belongs to; used for multi-tenant isolation in shared vector indices",
    )


class TamperAlert(BaseModel):
    """Alert generated when a chunk's KMS signature verification fails.

    Requirement 2.5: Excluded chunks produce tamper alerts logged to observability.
    """

    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=1)
    expected_signature: Optional[str] = Field(default=None)
    reason: str = Field(..., min_length=1, description="Reason for verification failure")
    detected_at: datetime


class RetrievalResult(BaseModel):
    """Result of hybrid retrieval with signature-verified chunks.

    Requirement 2.3-2.5: Combined RRF results, verified chunks, and tamper alerts.
    """

    verified_chunks: list[ScoredChunk] = Field(default_factory=list)
    tamper_alerts: list[TamperAlert] = Field(default_factory=list)
    no_evidence_found: bool = Field(default=False)
    degraded_search: bool = Field(
        default=False,
        description="True if one search method failed and results are from the other",
    )
    total_candidates: int = Field(default=0, ge=0)


# =============================================================================
# Graph State Models
# =============================================================================


class MemberActiveState(BaseModel):
    """Active clinical state of a member from the Causal Ontology Graph.

    Requirement 3.1, 3.4: Current diagnoses, active prescriptions,
    linked SDOH factors, and governing policy rules.
    """

    member_id: str = Field(..., min_length=1)
    active_diagnoses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Current active diagnosis records (ICD-10 codes, descriptions)",
    )
    active_prescriptions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active medication prescriptions",
    )
    sdoh_factors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Social Determinants of Health factors",
    )
    governing_policies: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Policy rules governing this member's conditions",
    )
    last_updated: Optional[datetime] = Field(default=None)


# =============================================================================
# CDC Pipeline Models
# =============================================================================


class CDCEvent(BaseModel):
    """Change Data Capture event from Snowflake/Iceberg source tables.

    Requirement 12.1: Represents a detected state change to process.
    """

    event_id: str = Field(..., min_length=1)
    entity_type: str = Field(
        ...,
        min_length=1,
        description="Node type: Member, Event, PolicyRule, SDOH_Factor, or EvidenceSource",
    )
    entity_id: str = Field(..., min_length=1)
    operation: str = Field(
        ...,
        description="Operation type: INSERT, UPDATE, or DELETE",
    )
    properties: dict[str, Any] = Field(default_factory=dict)
    source_table: str = Field(..., min_length=1)
    source_commit_timestamp: datetime = Field(
        ..., description="Timestamp of the source commit for ordering"
    )
    detected_at: datetime

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        allowed = {"INSERT", "UPDATE", "DELETE"}
        if v.upper() not in allowed:
            raise ValueError(f"operation must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        allowed = {"Member", "Event", "PolicyRule", "SDOH_Factor", "EvidenceSource"}
        if v not in allowed:
            raise ValueError(f"entity_type must be one of {allowed}, got '{v}'")
        return v


class EventCheckpoint(BaseModel):
    """Checkpoint for CDC pipeline resume-from-last-processed semantics.

    Requirement 12.5: Pipeline restarts resume from last successfully processed event.
    """

    last_event_id: str = Field(..., min_length=1)
    last_source_commit_timestamp: datetime
    last_processed_at: datetime
    total_events_processed: int = Field(default=0, ge=0)


# =============================================================================
# Briefing Packet Model
# =============================================================================


class BriefingPacket(BaseModel):
    """Pre-assembled context package for bounded agent reasoning.

    Requirement 4.3: Contains request_id, member_id, cpt_code,
    active_clinical_state, and verified_evidence_snippets with provenance.
    """

    request_id: str = Field(..., min_length=1)
    member_id: str = Field(..., min_length=1)
    cpt_code: str = Field(..., min_length=1, description="CPT procedure code for the PA request")
    active_clinical_state: MemberActiveState
    verified_evidence_snippets: list[ScoredChunk] = Field(default_factory=list)
    inferred_facts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Inferred SDOH factors and clinical conclusions",
    )
    no_evidence_found: bool = Field(default=False)
    degraded_inference: bool = Field(
        default=False,
        description="True if inference engine was unavailable",
    )

    @field_validator("verified_evidence_snippets")
    @classmethod
    def max_twenty_snippets(cls, v: list[ScoredChunk]) -> list[ScoredChunk]:
        if len(v) > 20:
            raise ValueError(
                f"BriefingPacket allows at most 20 evidence snippets, got {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def validate_snippet_scores(self) -> "BriefingPacket":
        """Ensure all included snippets meet minimum relevance threshold of 0.5."""
        for snippet in self.verified_evidence_snippets:
            if snippet.score < 0.5:
                raise ValueError(
                    f"Snippet {snippet.chunk_id} has score {snippet.score} below minimum 0.5"
                )
        return self


# =============================================================================
# Decision and Assessment Models
# =============================================================================


class CriterionAssessment(BaseModel):
    """Assessment of a single clinical necessity criterion.

    Requirement 9.1, 9.3: Per-criterion MET/NOT_MET/INDETERMINATE status.
    """

    criterion_id: str = Field(..., min_length=1)
    criterion_name: str = Field(..., min_length=1)
    status: CriterionStatus
    evidence_references: list[str] = Field(
        default_factory=list,
        description="IDs of evidence snippets supporting this assessment",
    )
    reasoning: Optional[str] = Field(default=None, description="Explanation for the status determination")


# =============================================================================
# Evidence Bundle and Lineage Models
# =============================================================================


class LineageEntry(BaseModel):
    """A single entry in the Evidence Bundle lineage trail.

    Requirement 10.2: Each entry contains conclusion, evidence_id, retrieval_timestamp.
    """

    conclusion: str = Field(..., min_length=1, description="The conclusion statement derived")
    evidence_id: str = Field(
        ..., min_length=1, description="ID of evidence snippet or graph query that produced it"
    )
    retrieval_timestamp: datetime = Field(
        ..., description="When the evidence was retrieved"
    )


class EvidenceBundle(BaseModel):
    """Complete evidence package for a PA decision.

    Requirement 10.1: Contains execution_id, decision, reason, lineage_trail,
    and original_document_signatures.
    """

    execution_id: str = Field(..., min_length=1)
    decision: str = Field(..., min_length=1, description="The PA decision (approve/escalate)")
    reason: str = Field(..., min_length=1, description="Reasoning for the decision")
    lineage_trail: list[LineageEntry] = Field(
        ..., min_length=1, description="Ordered lineage entries"
    )
    original_document_signatures: list[KMSSignature] = Field(
        ..., min_length=1, description="KMS signatures for every referenced source document"
    )
    execution_trace: Optional[list["TraceEntry"]] = Field(
        default=None, description="Complete execution trace attached to bundle"
    )

    @field_validator("lineage_trail")
    @classmethod
    def lineage_trail_not_empty(cls, v: list[LineageEntry]) -> list[LineageEntry]:
        if len(v) < 1:
            raise ValueError("lineage_trail must contain at least one entry")
        return v

    @field_validator("original_document_signatures")
    @classmethod
    def signatures_not_empty(cls, v: list[KMSSignature]) -> list[KMSSignature]:
        if len(v) < 1:
            raise ValueError("original_document_signatures must contain at least one signature")
        return v


# =============================================================================
# Execution Trace Models
# =============================================================================


class TraceEntry(BaseModel):
    """A single entry in the immutable execution trace.

    Requirement 8.1, 8.2: Entries have monotonically increasing sequence number,
    UTC ISO-8601 timestamp with millisecond precision, request_id, identity_id,
    and category.
    """

    sequence_number: int = Field(..., ge=0, description="Monotonically increasing sequence number")
    timestamp: str = Field(
        ...,
        min_length=1,
        description="UTC ISO-8601 timestamp with millisecond precision",
    )
    request_id: str = Field(..., min_length=1)
    identity_id: str = Field(..., min_length=1, description="Authenticated identity for attribution")
    category: TraceCategory
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional structured data for this trace entry"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_iso8601_timestamp(cls, v: str) -> str:
        """Validate UTC ISO-8601 format with millisecond precision."""
        import re

        # Pattern: YYYY-MM-DDTHH:MM:SS.mmmZ or with timezone offset
        iso8601_ms_pattern = (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(Z|[+-]\d{2}:\d{2})$"
        )
        if not re.match(iso8601_ms_pattern, v):
            raise ValueError(
                f"Timestamp must be UTC ISO-8601 with millisecond precision "
                f"(e.g., '2024-01-15T10:30:45.123Z'), got '{v}'"
            )
        return v


# =============================================================================
# Tool Orchestration Models (MCP Gateway)
# =============================================================================


class ToolDefinition(BaseModel):
    """Definition of an approved tool in the MCP Gateway catalog.

    Requirement 6.1: Tool catalog entry with name, schema, and description.
    """

    tool_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_schema: dict[str, Any] = Field(
        ..., description="JSON Schema defining permitted input parameters"
    )
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    @field_validator("input_schema")
    @classmethod
    def validate_schema_has_type(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "type" not in v:
            raise ValueError("input_schema must contain a 'type' field")
        return v


class ToolResult(BaseModel):
    """Result of a tool invocation through the MCP Gateway.

    Requirement 6.5: Records tool_name, input_parameters, output_result,
    duration_ms, and success/failure status.
    """

    tool_name: str = Field(..., min_length=1)
    input_parameters: dict[str, Any] = Field(default_factory=dict)
    output_result: Optional[Any] = Field(default=None)
    duration_ms: int = Field(..., ge=0)
    success: bool
    error_category: Optional[str] = Field(
        default=None, description="Error category if invocation failed"
    )
    invoked_at: datetime
    agent_identity: str = Field(..., min_length=1)


# =============================================================================
# Identity and Authorization Models
# =============================================================================


class RBACPolicy(BaseModel):
    """Role-Based Access Control policy configuration.

    Requirement 5.2: RBAC policy for authenticating and authorizing agent requests.
    """

    policy_id: str = Field(..., min_length=1)
    roles: dict[str, list[str]] = Field(
        ..., description="Mapping of role names to lists of permitted operations"
    )
    identity_role_assignments: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of identity_id to assigned role",
    )

    @field_validator("roles")
    @classmethod
    def roles_not_empty(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        if not v:
            raise ValueError("RBAC policy must define at least one role")
        return v


class AuthResult(BaseModel):
    """Result of successful authentication and authorization.

    Requirement 5.2, 5.4: Contains identity_id and granted permissions.
    """

    identity_id: str = Field(..., min_length=1)
    granted_permissions: list[str] = Field(default_factory=list)
    authenticated_at: datetime
    session_id: Optional[str] = Field(default=None)


# =============================================================================
# Forward reference resolution for EvidenceBundle -> TraceEntry
# =============================================================================

EvidenceBundle.model_rebuild()
