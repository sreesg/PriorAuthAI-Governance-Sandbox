"""Frontend API endpoints for BEACON status, Axisweave context, Evidence Bundle,
Graph, SDOH Inference, and Medical Director Queue panels.

Provides read-only FastAPI endpoints that the frontend visualization panels
query to display BEACON harness execution status, retrieved evidence chunks,
Evidence Bundle with lineage trail, member clinical graph state, inferred SDOH
factors, and escalated Medical Director queue cases.

Requirements:
    15.1: BEACON_Harness_Visualization displays 7-layer execution status
    15.2: Expands existing pipeline to 7 BEACON layers
    15.3: Axisweave_Context_Panel displays evidence chunks with provenance
    15.4: Evidence_Bundle_Viewer displays lineage trail linking conclusions to source
    15.5: Causal_Graph_Visualization renders member active clinical state as graph
    15.6: SDOH_Inference_Display shows inferred factors with chain and confidence
    15.7: Medical_Director_Queue_View displays escalated cases with artifacts
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.clinical_reasoning_fabric.beacon.audit_trail_service import AuditTrailService
from src.clinical_reasoning_fabric.models.core import TraceCategory, TraceEntry

# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

BEACON_LAYERS = [
    {"id": "L1", "name": "Identity"},
    {"id": "L2", "name": "Context"},
    {"id": "L3", "name": "MCP Gateway"},
    {"id": "L4", "name": "Sandbox"},
    {"id": "L5", "name": "Verification"},
    {"id": "L6", "name": "Observability"},
    {"id": "L7", "name": "Human Gates"},
]


class BeaconLayerStatus(BaseModel):
    """Status of a single BEACON layer."""

    id: str = Field(..., description="Layer identifier, e.g. 'L1'")
    name: str = Field(..., description="Human-readable layer name")
    state: str = Field(
        default="pending",
        description="Layer state: pending | active | passed | failed",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of the last state transition",
    )


class BeaconStatusResponse(BaseModel):
    """Response model for GET /api/beacon/status/{request_id}."""

    request_id: str
    layers: list[BeaconLayerStatus]
    current_layer: int = Field(
        default=0,
        description="Zero-based index of the currently active layer",
    )


class EvidenceChunkResponse(BaseModel):
    """A single evidence chunk with provenance metadata."""

    chunk_id: str
    text: str
    document_id: str
    content_hash: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    kms_status: str = Field(..., description="'valid' or 'invalid'")
    chunk_index: int
    ingestion_timestamp: str


class AxisweaveContextResponse(BaseModel):
    """Response model for GET /api/axisweave/context/{request_id}."""

    request_id: str
    chunks: list[EvidenceChunkResponse]


# ---------------------------------------------------------------------------
# SDOH Inference Response Models (Requirement 15.6)
# ---------------------------------------------------------------------------


class InferenceChainHopResponse(BaseModel):
    """A single hop in an inference chain."""

    hop_number: int
    source_text: str
    intermediate_conclusion: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class InferenceChainResponse(BaseModel):
    """Complete inference chain showing reasoning steps."""

    chain_id: str
    hops: list[InferenceChainHopResponse]
    cumulative_confidence: float = Field(..., ge=0.0, le=1.0)
    final_conclusion: str


class InferredFactResponse(BaseModel):
    """A single inferred SDOH factor or clinical conclusion."""

    fact_id: str
    type: str = Field(
        ...,
        description="Inference type: sdoh_factor | medication_adherence_risk | care_access_barrier",
    )
    category: Optional[str] = Field(
        default=None,
        description="SDOH category (only for sdoh_factor type)",
    )
    conclusion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    chain: InferenceChainResponse
    source_text: str = Field(..., description="Source text excerpt (up to 500 chars)")
    origin: str = Field(default="inferred", description="'inferred' or 'explicit'")


class ExplicitFactResponse(BaseModel):
    """An explicitly stated clinical fact from the graph."""

    fact_id: str
    type: str
    category: Optional[str] = None
    conclusion: str
    origin: str = Field(default="explicit")


class SDOHInferenceResponse(BaseModel):
    """Response model for GET /api/inference/sdoh/{member_id}."""

    member_id: str
    inferred_facts: list[InferredFactResponse]
    explicit_facts: list[ExplicitFactResponse]


# ---------------------------------------------------------------------------
# Medical Director Queue Response Models (Requirement 15.7)
# ---------------------------------------------------------------------------


class CriterionAssessmentResponse(BaseModel):
    """Per-criterion assessment status."""

    criterion: str
    status: str = Field(
        ...,
        description="Status: met | not_met | indeterminate | not_evaluated",
    )


class MDQueueCaseResponse(BaseModel):
    """A single escalated case in the Medical Director queue."""

    case_id: str
    briefing_summary: str
    criteria_assessment: list[CriterionAssessmentResponse]
    challenger_findings: str
    trace_summary: str
    escalated_at: str = Field(..., description="ISO-8601 timestamp of escalation")


class MDQueueResponse(BaseModel):
    """Response model for GET /api/md-queue."""

    cases: list[MDQueueCaseResponse]


# ---------------------------------------------------------------------------
# Layer-to-category mapping for deriving BEACON status from trace entries
# ---------------------------------------------------------------------------

_LAYER_CATEGORY_MAP: dict[str, str] = {
    "L1": "agent_action",       # Identity/RBAC actions
    "L2": "context_retrieval",  # Context Planner retrieval
    "L3": "tool_invocation",    # MCP Gateway tool calls
    "L4": "tool_invocation",    # Sandbox execution (sub-type of tool)
    "L5": "decision_step",      # OPA Challenger verification
    "L6": "agent_action",       # Observability (meta)
    "L7": "decision_step",      # Human Gates routing
}

# Map trace detail keys to layer IDs for more precise detection
_LAYER_DETAIL_KEYS: dict[str, str] = {
    "identity": "L1",
    "rbac": "L1",
    "authentication": "L1",
    "context_planner": "L2",
    "briefing_packet": "L2",
    "retrieval": "L2",
    "mcp_gateway": "L3",
    "tool_catalog": "L3",
    "sandbox": "L4",
    "opa_challenger": "L5",
    "verification": "L5",
    "signature_verification": "L5",
    "audit_trail": "L6",
    "observability": "L6",
    "human_gate": "L7",
    "escalation": "L7",
    "auto_approve": "L7",
}


def _derive_layer_id_from_entry(entry: TraceEntry) -> Optional[str]:
    """Derive which BEACON layer a trace entry corresponds to."""
    # First, check details keys for precise layer detection
    if entry.details:
        for key in entry.details:
            key_lower = key.lower()
            for detail_key, layer_id in _LAYER_DETAIL_KEYS.items():
                if detail_key in key_lower:
                    return layer_id

    # Fallback: map category to most likely layer
    category_to_layer: dict[str, str] = {
        TraceCategory.AGENT_ACTION: "L1",
        TraceCategory.CONTEXT_RETRIEVAL: "L2",
        TraceCategory.TOOL_INVOCATION: "L3",
        TraceCategory.DECISION_STEP: "L5",
    }
    return category_to_layer.get(entry.category)


def _derive_beacon_status(
    request_id: str, trace_entries: list[TraceEntry]
) -> BeaconStatusResponse:
    """Derive BEACON layer statuses from execution trace entries.

    Analyzes the trace to determine each layer's state:
    - pending: no trace entries for this layer
    - active: entries exist but no completion/failure indicator
    - passed: layer processing completed successfully
    - failed: layer processing encountered an error
    """
    # Initialize all layers as pending
    layer_states: dict[str, BeaconLayerStatus] = {}
    for layer_def in BEACON_LAYERS:
        layer_states[layer_def["id"]] = BeaconLayerStatus(
            id=layer_def["id"],
            name=layer_def["name"],
            state="pending",
            timestamp=None,
        )

    # Process trace entries to derive layer states
    current_layer = 0
    for entry in trace_entries:
        layer_id = _derive_layer_id_from_entry(entry)
        if layer_id and layer_id in layer_states:
            layer = layer_states[layer_id]
            layer.timestamp = entry.timestamp

            # Determine state from entry details
            is_error = False
            if entry.details:
                is_error = any(
                    k in ("error", "failure", "failed", "denied")
                    for k in entry.details
                )

            if is_error:
                layer.state = "failed"
            else:
                layer.state = "passed"

            # Track highest layer reached
            layer_index = next(
                (i for i, l in enumerate(BEACON_LAYERS) if l["id"] == layer_id),
                0,
            )
            if layer_index >= current_layer:
                current_layer = layer_index

    # The current layer (if not failed) should be marked 'active' if it's the last one
    # that has entries but processing may still be ongoing
    last_layer_id = BEACON_LAYERS[current_layer]["id"]
    if layer_states[last_layer_id].state == "passed":
        # Check if there are layers after this one that are still pending
        # If so, the current one is truly passed
        pass
    elif layer_states[last_layer_id].state == "pending":
        # If no entries for this layer yet, mark the previous completed one
        if current_layer > 0:
            current_layer -= 1

    return BeaconStatusResponse(
        request_id=request_id,
        layers=list(layer_states.values()),
        current_layer=current_layer,
    )


def _derive_axisweave_context(
    request_id: str, trace_entries: list[TraceEntry]
) -> AxisweaveContextResponse:
    """Derive Axisweave evidence chunks from trace entries.

    Looks for context_retrieval entries that contain evidence chunk data
    in their details field.
    """
    chunks: list[EvidenceChunkResponse] = []

    for entry in trace_entries:
        if entry.category != TraceCategory.CONTEXT_RETRIEVAL:
            continue

        if not entry.details:
            continue

        # Look for chunks stored in trace entry details
        entry_chunks = entry.details.get("chunks", [])
        if not entry_chunks and "chunk_id" in entry.details:
            # Single chunk in the entry details
            entry_chunks = [entry.details]

        for chunk_data in entry_chunks:
            if not isinstance(chunk_data, dict):
                continue
            try:
                chunk_resp = EvidenceChunkResponse(
                    chunk_id=chunk_data.get("chunk_id", ""),
                    text=chunk_data.get("text", ""),
                    document_id=chunk_data.get("document_id", ""),
                    content_hash=chunk_data.get("content_hash", ""),
                    relevance_score=float(chunk_data.get("relevance_score", chunk_data.get("score", 0.0))),
                    kms_status=chunk_data.get("kms_status", "valid"),
                    chunk_index=int(chunk_data.get("chunk_index", 0)),
                    ingestion_timestamp=chunk_data.get("ingestion_timestamp", ""),
                )
                chunks.append(chunk_resp)
            except (ValueError, TypeError):
                # Skip malformed chunk data
                continue

    return AxisweaveContextResponse(request_id=request_id, chunks=chunks)


def _derive_sdoh_inferences(
    member_id: str, trace_entries: list[TraceEntry]
) -> SDOHInferenceResponse:
    """Derive SDOH inferred and explicit facts from trace entries.

    Looks for trace entries that contain inference results or SDOH factor data
    in their details field (typically from agent_action or decision_step entries
    related to the Clinical Inference Engine).
    """
    inferred_facts: list[InferredFactResponse] = []
    explicit_facts: list[ExplicitFactResponse] = []

    for entry in trace_entries:
        if not entry.details:
            continue

        # Extract inferred facts from inference engine trace entries
        entry_inferred = entry.details.get("inferred_facts", [])
        for fact_data in entry_inferred:
            if not isinstance(fact_data, dict):
                continue
            try:
                # Build inference chain from data
                chain_data = fact_data.get("chain", fact_data.get("inference_chain", {}))
                hops_data = chain_data.get("hops", []) if isinstance(chain_data, dict) else []
                hops = []
                for hop in hops_data:
                    if isinstance(hop, dict):
                        hops.append(InferenceChainHopResponse(
                            hop_number=int(hop.get("hop_number", 1)),
                            source_text=hop.get("source_text", ""),
                            intermediate_conclusion=hop.get("intermediate_conclusion", ""),
                            confidence=float(hop.get("confidence", 0.0)),
                        ))

                chain = InferenceChainResponse(
                    chain_id=chain_data.get("chain_id", "") if isinstance(chain_data, dict) else "",
                    hops=hops,
                    cumulative_confidence=float(
                        chain_data.get("cumulative_confidence", fact_data.get("confidence", 0.0))
                        if isinstance(chain_data, dict) else fact_data.get("confidence", 0.0)
                    ),
                    final_conclusion=chain_data.get("final_conclusion", fact_data.get("conclusion", ""))
                    if isinstance(chain_data, dict) else fact_data.get("conclusion", ""),
                )

                inferred_facts.append(InferredFactResponse(
                    fact_id=fact_data.get("fact_id", ""),
                    type=fact_data.get("type", fact_data.get("inference_type", "sdoh_factor")),
                    category=fact_data.get("category", fact_data.get("sdoh_category")),
                    conclusion=fact_data.get("conclusion", ""),
                    confidence=float(fact_data.get("confidence", 0.0)),
                    chain=chain,
                    source_text=fact_data.get("source_text", fact_data.get("source_text_excerpt", ""))[:500],
                    origin="inferred",
                ))
            except (ValueError, TypeError):
                continue

        # Extract explicit facts from graph state trace entries
        entry_explicit = entry.details.get("explicit_facts", [])
        for fact_data in entry_explicit:
            if not isinstance(fact_data, dict):
                continue
            try:
                explicit_facts.append(ExplicitFactResponse(
                    fact_id=fact_data.get("fact_id", ""),
                    type=fact_data.get("type", ""),
                    category=fact_data.get("category"),
                    conclusion=fact_data.get("conclusion", ""),
                    origin="explicit",
                ))
            except (ValueError, TypeError):
                continue

        # Also extract SDOH factors from graph query results
        sdoh_factors = entry.details.get("sdoh_factors", [])
        for factor in sdoh_factors:
            if not isinstance(factor, dict):
                continue
            try:
                origin = factor.get("origin", "explicit")
                if origin == "inferred":
                    # Inferred factor stored in graph
                    chain_data = factor.get("chain", factor.get("inference_chain", {}))
                    hops_data = chain_data.get("hops", []) if isinstance(chain_data, dict) else []
                    hops = [
                        InferenceChainHopResponse(
                            hop_number=int(h.get("hop_number", 1)),
                            source_text=h.get("source_text", ""),
                            intermediate_conclusion=h.get("intermediate_conclusion", ""),
                            confidence=float(h.get("confidence", 0.0)),
                        )
                        for h in hops_data if isinstance(h, dict)
                    ]
                    chain = InferenceChainResponse(
                        chain_id=chain_data.get("chain_id", "") if isinstance(chain_data, dict) else "",
                        hops=hops,
                        cumulative_confidence=float(
                            chain_data.get("cumulative_confidence", factor.get("confidence", 0.0))
                            if isinstance(chain_data, dict) else factor.get("confidence", 0.0)
                        ),
                        final_conclusion=chain_data.get("final_conclusion", factor.get("conclusion", ""))
                        if isinstance(chain_data, dict) else factor.get("conclusion", ""),
                    )
                    inferred_facts.append(InferredFactResponse(
                        fact_id=factor.get("fact_id", factor.get("sdoh_id", "")),
                        type="sdoh_factor",
                        category=factor.get("category", factor.get("sdoh_category")),
                        conclusion=factor.get("conclusion", factor.get("description", "")),
                        confidence=float(factor.get("confidence", 0.0)),
                        chain=chain,
                        source_text=factor.get("source_text", "")[:500],
                        origin="inferred",
                    ))
                else:
                    explicit_facts.append(ExplicitFactResponse(
                        fact_id=factor.get("fact_id", factor.get("sdoh_id", "")),
                        type="sdoh_factor",
                        category=factor.get("category", factor.get("sdoh_category")),
                        conclusion=factor.get("conclusion", factor.get("description", "")),
                        origin="explicit",
                    ))
            except (ValueError, TypeError):
                continue

    return SDOHInferenceResponse(
        member_id=member_id,
        inferred_facts=inferred_facts,
        explicit_facts=explicit_facts,
    )


def _derive_md_queue_cases(trace_entries: list[TraceEntry]) -> MDQueueResponse:
    """Derive Medical Director queue cases from trace entries.

    Looks for decision_step entries that contain escalation data
    (human_gate or escalation details).
    """
    cases: list[MDQueueCaseResponse] = []

    for entry in trace_entries:
        if not entry.details:
            continue

        # Look for escalation entries (from Human Gates layer)
        escalation_data = entry.details.get("escalation")
        if escalation_data and isinstance(escalation_data, dict):
            try:
                # Extract criteria assessment
                criteria_raw = escalation_data.get("criteria_assessment", [])
                criteria = []
                for c in criteria_raw:
                    if isinstance(c, dict):
                        criteria.append(CriterionAssessmentResponse(
                            criterion=c.get("criterion", c.get("name", "")),
                            status=c.get("status", "not_evaluated").lower().replace(" ", "_"),
                        ))

                # Build challenger findings summary
                challenger_raw = escalation_data.get("challenger_findings", "")
                if isinstance(challenger_raw, dict):
                    findings_parts = []
                    if challenger_raw.get("tamper_alerts"):
                        findings_parts.append(
                            f"Tamper alerts: {len(challenger_raw['tamper_alerts'])}"
                        )
                    if challenger_raw.get("violated_rules"):
                        findings_parts.append(
                            f"Violated rules: {len(challenger_raw['violated_rules'])}"
                        )
                    if challenger_raw.get("verification_result"):
                        findings_parts.append(
                            f"Verification: {challenger_raw['verification_result']}"
                        )
                    challenger_findings = "; ".join(findings_parts) if findings_parts else "No findings"
                else:
                    challenger_findings = str(challenger_raw) if challenger_raw else "No findings"

                cases.append(MDQueueCaseResponse(
                    case_id=escalation_data.get("case_id", entry.request_id),
                    briefing_summary=escalation_data.get("briefing_summary", ""),
                    criteria_assessment=criteria,
                    challenger_findings=challenger_findings,
                    trace_summary=escalation_data.get("trace_summary", ""),
                    escalated_at=escalation_data.get(
                        "escalated_at", entry.timestamp
                    ),
                ))
            except (ValueError, TypeError):
                continue

        # Also check for md_queue entries directly in details
        md_queue_cases = entry.details.get("md_queue_cases", [])
        for case_data in md_queue_cases:
            if not isinstance(case_data, dict):
                continue
            try:
                criteria_raw = case_data.get("criteria_assessment", [])
                criteria = [
                    CriterionAssessmentResponse(
                        criterion=c.get("criterion", c.get("name", "")),
                        status=c.get("status", "not_evaluated").lower().replace(" ", "_"),
                    )
                    for c in criteria_raw if isinstance(c, dict)
                ]

                cases.append(MDQueueCaseResponse(
                    case_id=case_data.get("case_id", ""),
                    briefing_summary=case_data.get("briefing_summary", ""),
                    criteria_assessment=criteria,
                    challenger_findings=case_data.get("challenger_findings", "No findings"),
                    trace_summary=case_data.get("trace_summary", ""),
                    escalated_at=case_data.get("escalated_at", entry.timestamp),
                ))
            except (ValueError, TypeError):
                continue

    return MDQueueResponse(cases=cases)


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------


def create_frontend_router(audit_trail_service: AuditTrailService) -> APIRouter:
    """Create a FastAPI router with frontend panel endpoints.

    Args:
        audit_trail_service: The AuditTrailService instance used to query
            execution trace data for deriving BEACON status and evidence context.

    Returns:
        APIRouter with /api/beacon/status/{request_id} and
        /api/axisweave/context/{request_id} endpoints.
    """
    router = APIRouter(prefix="/api", tags=["frontend"])

    @router.get(
        "/beacon/status/{request_id}",
        response_model=BeaconStatusResponse,
        summary="Get BEACON harness execution status",
        description=(
            "Returns the 7-layer BEACON harness execution status for a given "
            "PA request. Each layer shows its current state (pending, active, "
            "passed, or failed) and timestamp of last state transition."
        ),
    )
    async def get_beacon_status(request_id: str) -> BeaconStatusResponse:
        """Query BEACON layer execution status for frontend visualization.

        Validates: Requirements 15.1, 15.2
        """
        try:
            trace_entries = await audit_trail_service.get_trace(request_id)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"No execution trace found for request_id: {request_id}",
            )

        if not trace_entries:
            # Return all-pending status if no trace entries exist
            return BeaconStatusResponse(
                request_id=request_id,
                layers=[
                    BeaconLayerStatus(id=l["id"], name=l["name"], state="pending")
                    for l in BEACON_LAYERS
                ],
                current_layer=0,
            )

        return _derive_beacon_status(request_id, trace_entries)

    @router.get(
        "/axisweave/context/{request_id}",
        response_model=AxisweaveContextResponse,
        summary="Get Axisweave retrieved evidence context",
        description=(
            "Returns retrieved evidence chunks with provenance metadata "
            "for a given PA request. Each chunk includes document_id, "
            "content_hash, relevance_score, and KMS verification status."
        ),
    )
    async def get_axisweave_context(request_id: str) -> AxisweaveContextResponse:
        """Query retrieved evidence chunks for frontend context panel.

        Validates: Requirements 15.3
        """
        try:
            trace_entries = await audit_trail_service.get_trace(request_id)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"No execution trace found for request_id: {request_id}",
            )

        if not trace_entries:
            return AxisweaveContextResponse(request_id=request_id, chunks=[])

        return _derive_axisweave_context(request_id, trace_entries)

    # ------------------------------------------------------------------
    # SDOH Inference endpoint (Requirement 15.6)
    # ------------------------------------------------------------------

    @router.get(
        "/inference/sdoh/{member_id}",
        response_model=SDOHInferenceResponse,
        summary="Get SDOH inferred and explicit facts for a member",
        description=(
            "Returns inferred SDOH factors with confidence scores, inference "
            "chains, and source text excerpts, plus explicitly stated factors. "
            "Each inferred fact includes a visual origin indicator distinguishing "
            "it from explicit facts."
        ),
    )
    async def get_sdoh_inferences(member_id: str) -> SDOHInferenceResponse:
        """Query inferred and explicit SDOH factors for a member.

        Derives data from audit trail trace entries that contain inference
        results from the Clinical Inference Engine and graph query results.

        Validates: Requirements 15.6
        """
        # Collect all trace entries that reference this member
        # Since traces are keyed by request_id, we scan all entries
        # that reference the member_id in their details
        try:
            all_entries = audit_trail_service.storage.get_all_entries()
        except Exception:
            # If storage doesn't support get_all_entries, return empty
            return SDOHInferenceResponse(
                member_id=member_id, inferred_facts=[], explicit_facts=[]
            )

        # Filter entries that reference this member
        member_entries: list[TraceEntry] = []
        for entry in all_entries:
            if not entry.details:
                continue
            # Check if entry is associated with this member
            entry_member = entry.details.get("member_id", "")
            if entry_member == member_id:
                member_entries.append(entry)
                continue
            # Also check nested structures for member_id
            if any(
                isinstance(v, dict) and v.get("member_id") == member_id
                for v in entry.details.values()
                if isinstance(v, dict)
            ):
                member_entries.append(entry)

        if not member_entries:
            return SDOHInferenceResponse(
                member_id=member_id, inferred_facts=[], explicit_facts=[]
            )

        return _derive_sdoh_inferences(member_id, member_entries)

    # ------------------------------------------------------------------
    # Medical Director Queue endpoint (Requirement 15.7)
    # ------------------------------------------------------------------

    @router.get(
        "/md-queue",
        response_model=MDQueueResponse,
        summary="Get escalated cases in the Medical Director queue",
        description=(
            "Returns all cases escalated to the Medical Director queue with "
            "the full artifact package: Briefing Packet summary, criteria "
            "assessment, OPA Challenger findings, and execution trace summary."
        ),
    )
    async def get_md_queue() -> MDQueueResponse:
        """Query escalated cases in the Medical Director review queue.

        Derives escalated cases from audit trail trace entries that contain
        escalation data from the Human Gates layer.

        Validates: Requirements 15.7
        """
        try:
            all_entries = audit_trail_service.storage.get_all_entries()
        except Exception:
            return MDQueueResponse(cases=[])

        # Filter to entries that contain escalation or MD queue data
        escalation_entries: list[TraceEntry] = []
        for entry in all_entries:
            if not entry.details:
                continue
            if (
                "escalation" in entry.details
                or "md_queue_cases" in entry.details
                or entry.details.get("human_gate") == "escalated"
            ):
                escalation_entries.append(entry)

        if not escalation_entries:
            return MDQueueResponse(cases=[])

        return _derive_md_queue_cases(escalation_entries)

    return router


# ---------------------------------------------------------------------------
# Evidence Bundle Response Models
# ---------------------------------------------------------------------------


class LineageEntryResponse(BaseModel):
    """A single lineage entry in the Evidence Bundle."""

    conclusion: str = Field(..., description="The conclusion statement derived")
    evidence_id: str = Field(
        ..., description="ID of evidence snippet or graph query that produced it"
    )
    timestamp: str = Field(
        ..., description="ISO-8601 retrieval timestamp"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Confidence score for this conclusion (0.00-1.00)",
    )


class SignatureResponse(BaseModel):
    """A KMS signature reference."""

    key_id: str = Field(..., description="KMS key identifier")
    signature: str = Field(..., description="Base64-encoded signature value")
    algorithm: str = Field(default="RSASSA_PKCS1_V1_5_SHA_256")


class EvidenceBundleResponse(BaseModel):
    """Response model for GET /api/evidence-bundle/{execution_id}."""

    execution_id: str
    decision: str = Field(..., description="PA decision (approve/escalate)")
    reason: str = Field(..., description="Reasoning for the decision")
    lineage_trail: list[LineageEntryResponse] = Field(
        ..., description="Ordered lineage entries linking conclusions to evidence"
    )
    signatures: list[SignatureResponse] = Field(
        ..., description="KMS signatures for referenced source documents"
    )


# ---------------------------------------------------------------------------
# Graph Visualization Response Models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A node in the member clinical state graph."""

    id: str = Field(..., description="Unique node identifier")
    type: str = Field(
        ...,
        description="Node type: diagnosis, medication, sdoh_factor, policy_rule, member",
    )
    label: str = Field(..., description="Human-readable label for display")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional node properties"
    )


class GraphEdge(BaseModel):
    """A directed edge in the member clinical state graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: str = Field(
        ...,
        description="Relationship type: HAS_CONDITION, IS_PRESCRIBED, TRIGGERED_BY, GOVERNED_BY, EVIDENCED_BY, INFERRED_FROM",
    )
    label: str = Field(..., description="Human-readable edge label")


class MemberGraphResponse(BaseModel):
    """Response model for GET /api/graph/member/{member_id}."""

    member_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# In-memory stores for Evidence Bundle and Graph data (frontend API layer)
# ---------------------------------------------------------------------------

# Simple in-memory stores keyed by execution_id / member_id.
# These are populated by the backend services and read by the frontend endpoints.
_evidence_bundle_store: dict[str, dict[str, Any]] = {}
_member_graph_store: dict[str, dict[str, Any]] = {}


def store_evidence_bundle(execution_id: str, bundle_data: dict[str, Any]) -> None:
    """Store an evidence bundle for frontend retrieval.

    Args:
        execution_id: The unique execution identifier.
        bundle_data: Dictionary containing decision, reason, lineage_trail,
            and signatures fields.
    """
    _evidence_bundle_store[execution_id] = bundle_data


def store_member_graph(member_id: str, graph_data: dict[str, Any]) -> None:
    """Store a member graph for frontend retrieval.

    Args:
        member_id: The member identifier.
        graph_data: Dictionary containing nodes and edges arrays.
    """
    _member_graph_store[member_id] = graph_data


def get_evidence_bundle_store() -> dict[str, dict[str, Any]]:
    """Access the evidence bundle store (for testing)."""
    return _evidence_bundle_store


def get_member_graph_store() -> dict[str, dict[str, Any]]:
    """Access the member graph store (for testing)."""
    return _member_graph_store


# ---------------------------------------------------------------------------
# Evidence Bundle & Graph Router Factory
# ---------------------------------------------------------------------------


def create_evidence_graph_router() -> APIRouter:
    """Create a FastAPI router with Evidence Bundle and Graph endpoints.

    Provides:
        GET /api/evidence-bundle/{execution_id} — Evidence Bundle with lineage trail
        GET /api/graph/member/{member_id} — Member active clinical state graph

    Returns:
        APIRouter with evidence bundle and graph visualization endpoints.
    """
    router = APIRouter(prefix="/api", tags=["frontend"])

    @router.get(
        "/evidence-bundle/{execution_id}",
        response_model=EvidenceBundleResponse,
        summary="Get Evidence Bundle with lineage trail",
        description=(
            "Returns the complete Evidence Bundle for a given execution, "
            "including the decision, reasoning, ordered lineage trail linking "
            "each conclusion to its source evidence, and KMS document signatures."
        ),
    )
    async def get_evidence_bundle(execution_id: str) -> EvidenceBundleResponse:
        """Query Evidence Bundle for the frontend Evidence Bundle Viewer.

        Validates: Requirements 15.4
        """
        bundle_data = _evidence_bundle_store.get(execution_id)
        if bundle_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No evidence bundle found for execution_id: {execution_id}",
            )

        # Build lineage trail response
        lineage_trail = []
        for entry in bundle_data.get("lineage_trail", []):
            lineage_trail.append(
                LineageEntryResponse(
                    conclusion=entry.get("conclusion", ""),
                    evidence_id=entry.get("evidence_id", ""),
                    timestamp=entry.get("timestamp", ""),
                    confidence=entry.get("confidence"),
                )
            )

        # Build signatures response
        signatures = []
        for sig in bundle_data.get("signatures", []):
            signatures.append(
                SignatureResponse(
                    key_id=sig.get("key_id", ""),
                    signature=sig.get("signature", ""),
                    algorithm=sig.get("algorithm", "RSASSA_PKCS1_V1_5_SHA_256"),
                )
            )

        return EvidenceBundleResponse(
            execution_id=execution_id,
            decision=bundle_data.get("decision", ""),
            reason=bundle_data.get("reason", ""),
            lineage_trail=lineage_trail,
            signatures=signatures,
        )

    @router.get(
        "/graph/member/{member_id}",
        response_model=MemberGraphResponse,
        summary="Get member active clinical state graph",
        description=(
            "Returns the member's active clinical state from the Neo4j graph "
            "as a collection of typed nodes (diagnoses, medications, SDOH factors, "
            "policy rules) and directed edges representing clinical relationships."
        ),
    )
    async def get_member_graph(member_id: str) -> MemberGraphResponse:
        """Query member clinical state graph for the Causal Graph Visualization panel.

        Validates: Requirements 15.5
        """
        graph_data = _member_graph_store.get(member_id)
        if graph_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No graph data found for member_id: {member_id}",
            )

        # Build nodes response
        nodes = []
        for node in graph_data.get("nodes", []):
            nodes.append(
                GraphNode(
                    id=node.get("id", ""),
                    type=node.get("type", ""),
                    label=node.get("label", ""),
                    properties=node.get("properties", {}),
                )
            )

        # Build edges response
        edges = []
        for edge in graph_data.get("edges", []):
            edges.append(
                GraphEdge(
                    source=edge.get("source", ""),
                    target=edge.get("target", ""),
                    type=edge.get("type", ""),
                    label=edge.get("label", ""),
                )
            )

        return MemberGraphResponse(
            member_id=member_id,
            nodes=nodes,
            edges=edges,
        )

    return router
