# Implementation Plan: Clinical Reasoning Fabric

## Overview

This implementation plan breaks down the Clinical Reasoning Fabric (CRF) into incremental coding tasks across the Axisweave Retrieval Stack, Causal Ontology Graph, CDC Pipeline, BEACON 7-Layer Safety Harness, Axisweave Service API, Clinical Inference Engine, and Frontend Application Integration. Each task builds on prior tasks, with property-based tests validating correctness properties and checkpoints ensuring incremental validation. The implementation uses Python with Hypothesis for property-based testing, and vanilla JavaScript for frontend panels.

## Tasks

- [x] 1. Set up project structure, core data models, and testing framework
  - [x] 1.1 Create project directory structure and install dependencies
    - Create `src/clinical_reasoning_fabric/` package with `__init__.py`
    - Create subdirectories: `ingestion/`, `retrieval/`, `graph/`, `cdc/`, `beacon/`, `models/`, `api/`, `inference/`, `frontend/`
    - Create `tests/property/`, `tests/unit/`, `tests/integration/` directories
    - Update `requirements.txt` with: `hypothesis`, `qdrant-client`, `neo4j`, `docling`, `chonkie`, `opa-python`, `dagster`, `dbt-core`, `dbt-snowflake`, `pydantic`, `cryptography`, `boto3`, `fastapi`, `uvicorn`
    - Set up `pytest.ini` or `pyproject.toml` with test configuration
    - _Requirements: 1.1, 1.5, 2.1, 3.1, 13.1, 14.1, 15.8_

  - [x] 1.2 Implement core data models and enums
    - Create `src/clinical_reasoning_fabric/models/core.py` with all dataclasses: `KMSSignature`, `ChunkProvenance`, `DocumentChunk`, `IngestionResult`, `ScoredChunk`, `TamperAlert`, `RetrievalResult`, `MemberActiveState`, `CDCEvent`, `EventCheckpoint`, `BriefingPacket`, `CriterionAssessment`, `LineageEntry`, `EvidenceBundle`, `TraceEntry`, `ToolDefinition`, `ToolResult`, `RBACPolicy`, `AuthResult`
    - Create enums: `TraceCategory`, `CriterionStatus`, `VerificationResult`, `Disposition`
    - Add Pydantic validators for schema conformance checks
    - _Requirements: 8.1, 10.1, 3.1, 4.3_

  - [x] 1.3 Implement custom exception hierarchy
    - Create `src/clinical_reasoning_fabric/models/exceptions.py`
    - Define: `IngestionError`, `PIIScrubError`, `KMSUnavailableError`, `UnauthorizedError`, `TraceRecordingError`, `MemberNotFoundError`, `BundleValidationError`, `UnmappableRecordError`, `ToolValidationError`, `InvalidNamespaceError`, `InferenceTimeoutError`
    - Each exception captures structured context (document_id, identity, timestamp, reason)
    - _Requirements: 1.6, 1.7, 1.8, 5.3, 5.5, 8.5, 10.5, 12.6, 13.8, 14.9_

- [x] 2. Implement Document Ingestion Service (Axisweave Retrieval Stack)
  - [x] 2.1 Implement PII Scrubber with HIPAA Safe Harbor compliance
    - Create `src/clinical_reasoning_fabric/ingestion/pii_scrubber.py`
    - Implement regex-based detection and removal of all 18 HIPAA Safe Harbor identifier categories: names, geographic data (smaller than state), dates (except year), phone numbers, fax numbers, email addresses, SSNs, medical record numbers, health plan numbers, account numbers, certificate/license numbers, VINs, device identifiers, URLs, IP addresses, biometric identifiers, photographic images references, and any other unique identifying codes
    - Raise `PIIScrubError` on any failure; halt processing and prevent document storage
    - _Requirements: 1.2, 1.8_

  - [x] 2.2 Write property test for PII scrubbing completeness
    - **Property 1: PII Scrubbing Completeness**
    - Test that for any generated text containing HIPAA identifiers, scrubbing removes all identifier patterns while preserving non-PII content
    - Use Hypothesis strategies to generate SSNs, phone numbers, emails, dates, MRNs, etc.
    - **Validates: Requirements 1.2, 5.1**

  - [x] 2.3 Implement DocumentIngestionService with Docling parsing, SHA-256 hashing, and KMS signing
    - Create `src/clinical_reasoning_fabric/ingestion/document_ingestion_service.py`
    - Implement `parse_pdf()` using Docling PDF parser; raise `IngestionError` on corrupted/unsupported formats
    - Implement `compute_hash()` using hashlib SHA-256
    - Implement `sign_hash()` using boto3 KMS client with asymmetric key; raise `KMSUnavailableError` on failure
    - Implement `ingest_document()` orchestrating full pipeline: parse → scrub → hash → sign → chunk → store
    - On KMS failure: halt ingestion, discard unsigned chunks, log to audit trail
    - _Requirements: 1.1, 1.3, 1.4, 1.6, 1.7_

  - [x] 2.4 Write property test for cryptographic provenance round-trip
    - **Property 2: Cryptographic Provenance Round-Trip**
    - Test that for any text string, hashing and signing then verifying produces valid result; same text always produces same hash
    - **Validates: Requirements 1.3, 1.4**

  - [x] 2.5 Implement Chonkie semantic chunking and Qdrant storage
    - Create `src/clinical_reasoning_fabric/ingestion/chunker.py`
    - Implement semantic chunking using Chonkie library
    - Store each chunk in Qdrant with provenance metadata: `document_id`, `content_hash`, `kms_signature`, `chunk_index`, `ingestion_timestamp`
    - Create Qdrant collection with dense vector (1024 dims, cosine) and BM25 sparse vector configuration
    - Include `namespace`, `document_category`, and `tenant_id` payload fields for multi-tenant support
    - _Requirements: 1.5, 13.2_

  - [x] 2.6 Write property test for chunking content preservation
    - **Property 3: Chunking Content Preservation with Provenance**
    - Test that for any non-empty text, chunking produces >= 1 chunk, union of chunk texts contains original content, every chunk has all required provenance fields with valid values
    - **Validates: Requirements 1.5**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Hybrid Retrieval Service
  - [x] 4.1 Implement HybridRetrievalService with dense search, BM25 sparse search, and RRF
    - Create `src/clinical_reasoning_fabric/retrieval/hybrid_retrieval_service.py`
    - Implement `dense_search()`: query Qdrant with embedding vector, return top 50 by cosine similarity
    - Implement `sparse_search()`: query Qdrant BM25 index, return top 50 by BM25 score
    - Implement `reciprocal_rank_fusion()`: combine using RRF formula `score(d) = Σ 1/(k + rank_i(d))`, k=60, return top-k (default 20)
    - Handle search timeout (10s): if one method fails, proceed with available results and log degraded warning
    - Return empty result with `no_evidence_found` indicator if both searches return zero matches
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7_

  - [x] 4.2 Write property test for Reciprocal Rank Fusion correctness
    - **Property 4: Reciprocal Rank Fusion Correctness**
    - Test that for any two ranked lists: output is sorted descending by RRF score, bounded to top_k, and items appearing in both lists score higher than items in only one list at equivalent ranks
    - **Validates: Requirements 2.3**

  - [x] 4.3 Implement KMS signature verification filter
    - Add `verify_signatures()` to `HybridRetrievalService`
    - Verify each chunk's KMS signature against its content_hash
    - Exclude chunks with invalid/missing signatures from results
    - Produce a `TamperAlert` for each excluded chunk and log to observability layer
    - _Requirements: 2.4, 2.5_

  - [x] 4.4 Write property test for signature verification filter
    - **Property 5: Signature Verification Filter**
    - Test that for any set of chunks with mixed valid/invalid signatures, the filter includes exactly valid-signature chunks and excludes exactly invalid ones, producing tamper alerts for each exclusion
    - **Validates: Requirements 2.4, 7.1, 11.2**

- [x] 5. Implement Causal Ontology Graph Service
  - [x] 5.1 Implement CausalOntologyGraphService with Neo4j operations
    - Create `src/clinical_reasoning_fabric/graph/causal_ontology_graph_service.py`
    - Implement `get_member_active_state()`: Cypher query returning active diagnoses, prescriptions, SDOH factors, governing policies; must complete within 2 seconds
    - Implement active-state filtering: return only records NOT marked resolved/discontinued/closed/superseded
    - Implement `upsert_node()`: upsert graph nodes with optional `execution_id` provenance
    - Implement `upsert_relationship()`: upsert directed relationships between nodes
    - Implement `query_related_evidence()`: retrieve evidence nodes linked to a member's condition
    - Define Neo4j constraints for Member, Event, PolicyRule, SDOH_Factor, EvidenceSource uniqueness
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

  - [x] 5.2 Write property test for active state exclusion
    - **Property 6: Active State Exclusion of Resolved Records**
    - Test that for any set of records with mixed statuses, active state query returns only non-resolved/non-discontinued/non-closed/non-superseded records
    - **Validates: Requirements 3.5**

- [x] 6. Implement CDC Pipeline Service
  - [x] 6.1 Implement CDCPipelineService with dbt transformation and Dagster orchestration
    - Create `src/clinical_reasoning_fabric/cdc/cdc_pipeline_service.py`
    - Implement `process_change_event()`: transform CDC event to graph entity and apply upsert
    - Implement `transform_to_graph_entity()`: map source records to graph node/relationship types using dbt model definitions; raise `UnmappableRecordError` for invalid mappings
    - Implement `apply_upsert()`: update target node/relationship properties WITHOUT removing unrelated relationships
    - Retry up to 3 times with exponential backoff (5s, 10s, 20s) on apply failure
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x] 6.2 Implement CDC event checkpoint and ordering logic
    - Implement `get_checkpoint()` and `update_checkpoint()`: persist last successfully processed event
    - On pipeline restart, resume from last checkpoint without duplicating mutations
    - Sort and apply multiple events for the same entity by source-commit-timestamp order
    - Skip unmappable records with logging; continue processing subsequent events
    - _Requirements: 12.5, 12.6, 12.7_

  - [x] 6.3 Write property tests for CDC pipeline
    - **Property 18: CDC Transformation Type Correctness** — For any valid source record with recognized entity_type, transformation produces correct node type with required fields
    - **Property 19: CDC Upsert Relationship Preservation** — Upsert updates only referenced properties/relationships, leaving others unchanged
    - **Property 20: CDC Checkpoint Idempotency** — Replay from checkpoint does not duplicate nodes/relationships
    - **Property 21: CDC Temporal Ordering** — Events applied in source-commit-timestamp order yield latest state
    - **Validates: Requirements 12.2, 12.3, 12.5, 12.7**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement BEACON Layer 1 — Identity and Permissions
  - [x] 8.1 Implement IdentityService with authentication, RBAC, and PHI masking
    - Create `src/clinical_reasoning_fabric/beacon/identity_service.py`
    - Implement `authenticate_and_authorize()`: verify credentials and check RBAC policy; raise `UnauthorizedError` with no clinical data exposure on failure
    - Implement `mask_phi()`: replace all PII/PHI fields with irreversible masked tokens in trace logs and observability outputs
    - Implement `create_trace_context()`: associate every action with authenticated identity_id
    - Log unauthorized access attempts with: requesting identity, requested operation, timestamp, missing permission
    - Log authentication failures with: timestamp, reason (missing/invalid/expired credentials)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 8.2 Write property tests for identity isolation and attribution
    - **Property 8: Unauthorized Response Data Isolation** — Denial responses contain zero clinical data fields; audit log contains required fields
    - **Property 9: Trace Entry Identity Attribution** — Every trace entry for an authenticated session contains the correct identity_id
    - **Validates: Requirements 5.3, 5.4**

- [x] 9. Implement BEACON Layer 2 — Context Planner
  - [x] 9.1 Implement ContextPlannerService for Briefing Packet assembly
    - Create `src/clinical_reasoning_fabric/beacon/context_planner_service.py`
    - Implement `assemble_briefing_packet()` with 30-second timeout:
      1. Query Neo4j for member active clinical state (raise `MemberNotFoundError` if not found)
      2. Query Qdrant for relevant evidence snippets using CPT code and diagnosis context (max 20 snippets, min relevance 0.5)
      3. Filter to only diagnoses, medications, and evidence matching the requested CPT code or associated clinical condition categories
      4. Package into `BriefingPacket` conforming to JSON schema with all required fields
    - Handle zero evidence: assemble with empty `verified_evidence_snippets` and set `no_evidence_found` flag
    - Log timeout error if assembly exceeds 30 seconds
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 9.2 Write property test for Briefing Packet assembly invariants
    - **Property 7: Briefing Packet Assembly Invariants**
    - Test that for any valid PA request: at most 20 snippets, all scores >= 0.5, only CPT-relevant diagnoses, and schema conformance with all required fields
    - **Validates: Requirements 4.2, 4.3, 4.4**

- [x] 10. Implement BEACON Layer 3 — MCP Gateway
  - [x] 10.1 Implement MCPGatewayService with tool catalog and sandboxed execution
    - Create `src/clinical_reasoning_fabric/beacon/mcp_gateway_service.py`
    - Implement `validate_tool_request()`: verify tool name exists in catalog and parameters conform to permitted schema (JSON Schema validation)
    - Implement `invoke_tool()`: validate request, execute in sandboxed environment with configurable timeout (default 30s), record invocation trace
    - Reject unapproved tools with logging: agent identity, tool name, timestamp
    - Record all invocations (success or fail) with: tool_name, input_parameters, output_result, duration_ms, success/failure status
    - On tool failure (timeout/crash/error): record failure with error category, return structured error to agent, do not retry
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 10.2 Write property tests for MCP Gateway
    - **Property 10: MCP Tool Catalog Validation** — Tool invocation accepted iff tool exists in catalog AND parameters match schema; all others rejected
    - **Property 11: Tool Invocation Record Completeness** — Every invocation trace entry contains all 5 required fields
    - **Validates: Requirements 6.2, 6.5**

- [x] 11. Implement BEACON Layer 5 — OPA Challenger Agent
  - [x] 11.1 Implement OPAChallengerService with KMS verification and OPA evaluation
    - Create `src/clinical_reasoning_fabric/beacon/opa_challenger_service.py`
    - Implement `verify_signatures()`: verify KMS signatures on all evidence snippets within 30 seconds; report invalid/missing signatures
    - Implement `evaluate_policy()`: evaluate decision against OPA `rules.rego` within 10 seconds; produce PASS/FAIL with violated rule identifiers
    - Implement `verify_decision()`: orchestrate signature verification then policy evaluation; return `VerificationResult`
    - On any signature invalid/missing: reject decision, escalate to Medical Director with tamper alert identifying affected snippets
    - On OPA rule violation: route to Medical Director queue with violated rule identifiers and descriptions
    - On infrastructure failure (KMS unavailable, rules.rego loading failure): treat as FAIL, halt decision, escalate with indication verification could not be completed
    - Ensure no shared mutable state with primary reasoning agent
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 11.2 Write property test for OPA policy evaluation determinism
    - **Property 12: OPA Policy Evaluation Determinism**
    - Test that for any valid input conforming to rules.rego schema, evaluating the same input always produces the same PASS/FAIL result; FAIL always cites specific violated rule identifiers
    - **Validates: Requirements 7.2, 7.4**

- [x] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement BEACON Layer 6 — Audit Trail Service
  - [x] 13.1 Implement AuditTrailService with immutable append-only trace recording
    - Create `src/clinical_reasoning_fabric/beacon/audit_trail_service.py`
    - Implement `record_entry()`: assign monotonically increasing sequence numbers, include UTC ISO-8601 timestamps with ms precision, request_id, identity_id, category (agent_action | tool_invocation | context_retrieval | decision_step)
    - Implement append-only storage backend that prohibits modification or deletion of historical entries
    - Implement 7-year retention policy
    - Implement `get_trace()`: retrieve complete trace by request_id within 30 seconds
    - On trace recording failure: HALT PA processing, return error (no unaudited decisions)
    - _Requirements: 8.1, 8.2, 8.4, 8.5, 8.6_

  - [x] 13.2 Write property test for execution trace ordering invariant
    - **Property 13: Execution Trace Ordering Invariant**
    - Test that for any sequence of trace entries: sequence_numbers are strictly increasing, timestamps are valid UTC ISO-8601 with ms precision, request_id is non-empty and matching, identity_id is non-empty, category is valid TraceCategory enum
    - **Validates: Requirements 8.1, 8.2**

- [x] 14. Implement BEACON Layer 7 — Human Gates Service
  - [x] 14.1 Implement HumanGateService with no-automated-denial enforcement
    - Create `src/clinical_reasoning_fabric/beacon/human_gate_service.py`
    - Implement `route_decision()`:
      - All criteria MET + verification PASS → Auto-Approve (no MD review required)
      - Any criterion NOT_MET or INDETERMINATE → Escalate to MD within 30 seconds
      - Verification FAIL → Route to MD with challenge findings
      - NEVER produce automated denial regardless of any input combination
    - Include all four artifacts on escalation: Briefing_Packet, criteria assessment (per-criterion MET/NOT_MET/INDETERMINATE), OPA findings, complete execution trace
    - On MD queue unavailable: hold in pending state, retry delivery at 30-second intervals up to 10 attempts, log each failed attempt
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 14.2 Write property tests for decision routing and escalation
    - **Property 14: Decision Routing Correctness** — All MET + PASS → approved; any NOT_MET/INDETERMINATE → escalated; verification FAIL → escalated; NEVER denied
    - **Property 15: Escalation Artifact Completeness** — Every escalation includes all four non-null artifacts: Briefing_Packet, criteria assessment, OPA findings, execution trace
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.5**

- [x] 15. Implement Evidence Bundle Service
  - [x] 15.1 Implement EvidenceBundleService with schema validation and lineage assembly
    - Create `src/clinical_reasoning_fabric/beacon/evidence_bundle_service.py`
    - Implement `produce_bundle()`:
      - Produce Evidence_Bundle with: execution_id, decision, reason, lineage_trail (ordered entries with conclusion, evidence_id, retrieval_timestamp), original_document_signatures (KMS signatures for every referenced source document)
      - Validate all required schema fields present, non-null, correct data type
      - Validate lineage_trail has >= 1 entry and original_document_signatures has >= 1 signature
      - Attach complete execution trace to bundle
    - On validation failure: halt decision, log bundle integrity error with missing/invalid fields, escalate to MD
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 8.3_

  - [x] 15.2 Write property test for Evidence Bundle schema conformance
    - **Property 16: Evidence Bundle Schema Conformance**
    - Test that any produced bundle has all required fields present and non-null, lineage_trail has >= 1 entry (each with conclusion, evidence_id, retrieval_timestamp), signatures has >= 1 valid KMS signature, and execution trace attached
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 8.3**

- [x] 16. Implement Bounded Agent Reasoning with Targeted Retrieval
  - [x] 16.1 Implement bounded retrieval logic within agent reasoning flow
    - Create `src/clinical_reasoning_fabric/beacon/bounded_retrieval.py`
    - Implement retrieval call limiting: configurable maximum (default 10, range 1-50)
    - All additional retrieval calls must go through MCP_Gateway using approved retrieval tool
    - Verify KMS signatures on newly retrieved chunks; discard and log tamper alert for invalid ones
    - Record graph updates with `execution_id` as provenance
    - On retrieval limit reached: proceed to decision with available evidence, record `retrieval_limit_reached` event in trace with call count and limit value
    - Include all additionally retrieved snippets with provenance in Evidence_Bundle lineage_trail
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [x] 16.2 Write property test for bounded retrieval with lineage tracking
    - **Property 17: Bounded Retrieval with Lineage Tracking**
    - Test that total retrieval calls never exceed configured max, all valid retrieved snippets appear in lineage_trail with provenance, all graph updates have execution_id provenance
    - **Validates: Requirements 11.4, 11.5, 11.7**

- [x] 17. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 18. Implement Axisweave Service API (Standalone Microservice Layer)
  - [x] 18.1 Implement namespace model and validation
    - Create `src/clinical_reasoning_fabric/api/namespace.py`
    - Implement `Namespace` dataclass with: namespace_id, owner_tenant_id, created_at, cross_namespace_grants
    - Implement `validate_namespace()`: validate format (1-128 alphanumeric, hyphen, underscore chars via regex `^[a-zA-Z0-9_-]{1,128}$`)
    - Raise `InvalidNamespaceError` for non-conforming strings (empty, >128 chars, special chars)
    - Implement `NamespaceRegistry` for managing namespace lifecycle and cross-namespace grants
    - _Requirements: 13.2, 13.8_

  - [x] 18.2 Write property test for namespace format validation
    - **Property 22: Namespace Format Validation**
    - Test that any string of 1-128 alphanumeric/hyphen/underscore characters is accepted; empty, >128, or strings with other characters are rejected
    - Use Hypothesis strategies: `st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_", min_size=1, max_size=128)` for valid, and arbitrary text for invalid
    - **Validates: Requirements 13.2, 13.8**

  - [x] 18.3 Implement API authentication and tenant isolation
    - Create `src/clinical_reasoning_fabric/api/auth_provider.py`
    - Implement `APIAuthProvider` with API key validation and tenant_id extraction
    - Implement `check_namespace_access()`: verify caller can access target namespace (own namespace by default, cross-namespace requires explicit grant in token scope)
    - On invalid/missing credentials: reject request, expose zero document data, log auth failure with timestamp and source identifier
    - Implement `APICredentials` dataclass with api_key, tenant_id, authorized_namespaces
    - _Requirements: 13.5, 13.9_

  - [x] 18.4 Write property tests for Axisweave authentication data isolation
    - **Property 25: Axisweave Authentication Data Isolation**
    - Test that for any request with invalid/missing credentials, the response contains zero document data fields (no chunk text, no content hashes, no signatures, no provenance metadata) and audit log records timestamp and source identifier
    - **Validates: Requirements 13.9**

  - [x] 18.5 Implement AxisweaveServiceAPI with versioned REST endpoints
    - Create `src/clinical_reasoning_fabric/api/axisweave_service_api.py`
    - Implement `ingest()`: accept document ingestion with namespace and document_category (1-64 chars), validate namespace, authenticate, ingest via DocumentIngestionService, return document_id, content_hash, signature, chunk_count
    - Implement `retrieve()`: execute hybrid search scoped to caller namespace, enforce namespace isolation on results, return chunks with full provenance metadata within 10 seconds
    - Implement `verify()`: verify KMS provenance for specified document/chunks, independent of prior ingestion by same caller
    - Each operation callable independently without requiring prior invocation of other operations
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [x] 18.6 Write property test for API operation independence
    - **Property 24: API Operation Independence**
    - Test that for any valid ingest, retrieve, or verify request with proper auth and valid namespace, the operation succeeds without requiring prior invocation of any other operation
    - **Validates: Requirements 13.1**

  - [x] 18.7 Implement namespace-scoped retrieval isolation and cross-namespace access
    - Add namespace filtering to retrieval: all returned chunks must have namespace metadata matching caller's authorized namespaces
    - Implement shared vector index partitioning: apply namespace-scoped access control before returning results
    - Cross-namespace access requires explicit grant in caller token/API key scope
    - No chunks from unauthorized namespaces included in responses regardless of shared index configuration
    - _Requirements: 13.5, 13.7_

  - [x] 18.8 Write property test for namespace isolation in retrieval
    - **Property 23: Namespace Isolation in Retrieval**
    - Test that for any retrieval query specifying namespace A, all returned chunks have namespace metadata equal to A; no chunks from namespace B appear unless explicit cross-namespace authorization exists
    - **Validates: Requirements 13.3, 13.5, 13.7**

  - [x] 18.9 Implement API versioning router and backwards compatibility
    - Create `src/clinical_reasoning_fabric/api/version_router.py`
    - Implement `APIVersionRouter` with semantic versioning (major.minor.patch)
    - Route requests to correct handler version based on request version header
    - Support prior major version for minimum 6 months after new major version release
    - Return 410 Gone for unsupported versions
    - Implement `APIVersionInfo` dataclass with current_version, supported_versions, deprecation_schedule
    - _Requirements: 13.6_

- [x] 19. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 20. Implement Clinical Inference Engine
  - [x] 20.1 Implement ClinicalInferenceEngine core with LLM-powered inference
    - Create `src/clinical_reasoning_fabric/inference/clinical_inference_engine.py`
    - Implement `analyze_snippet()`: analyze a single clinical note snippet for implied conclusions within 15-second timeout
    - Implement `derive_sdoh_factors()`: use LLM reasoning to derive SDOH factors from text
    - Support SDOH categories: housing_instability, transportation_barriers, medication_storage_limitations, food_insecurity, caregiver_availability
    - Produce max 10 inferred facts per snippet, each tagged with inference_type (sdoh_factor, medication_adherence_risk, care_access_barrier) and confidence score (0.0-1.0)
    - _Requirements: 14.1, 14.2, 14.6_

  - [x] 20.2 Write property test for inference output invariants
    - **Property 27: Inference Output Invariants**
    - Test that for any inferred fact: inference_type is one of {sdoh_factor, medication_adherence_risk, care_access_barrier}, confidence is in [0.0, 1.0], if sdoh_factor then sdoh_category is from valid set, max 10 facts per snippet
    - **Validates: Requirements 14.2, 14.6**

  - [x] 20.3 Implement inference chain builder with configurable depth
    - Implement `InferenceChain`, `InferenceHop` dataclasses
    - Implement shallow depth (1-hop): direct single-hop implications only
    - Implement deep depth (up to 3-hop): multi-hop reasoning chains with intermediate conclusions
    - Default depth: shallow; configurable to deep
    - Each hop records: source_text, intermediate_conclusion, confidence score
    - No chain exceeds 3 hops regardless of configuration
    - _Requirements: 14.4, 14.5_

  - [x] 20.4 Write property test for inference depth constraint
    - **Property 29: Inference Depth Constraint**
    - Test that shallow depth produces exactly 1 hop, deep depth produces 1-3 hops, no chain exceeds 3 hops
    - **Validates: Requirements 14.4**

  - [x] 20.5 Implement confidence calculation and threshold filtering
    - Implement `compute_chain_confidence()`: cumulative confidence = product of individual hop confidences (h1 × h2 × ... × hn)
    - Implement `apply_threshold_filter()`: discard any inferred fact with confidence below configurable threshold (default 0.3)
    - Only facts with confidence >= threshold returned to downstream consumers
    - _Requirements: 14.5, 14.7_

  - [x] 20.6 Write property tests for confidence calculation and threshold
    - **Property 26: Inference Confidence Calculation** — For any chain with N hops (1≤N≤3), cumulative confidence equals product of hop scores
    - **Property 28: Inference Threshold Filtering** — Every fact in downstream output has confidence >= threshold; no fact below threshold appears
    - **Validates: Requirements 14.5, 14.7**

  - [x] 20.7 Implement graph linking for inferred SDOH factors
    - Implement `link_to_graph()`: create INFERRED_FROM relationship in Neo4j linking inferred SDOH factor to source evidence
    - Relationship properties: source_text, inference_chain_json, confidence, inferred_at
    - Only link facts with confidence >= threshold
    - Create SDOH_Factor node with origin="inferred" and link to EvidenceSource node
    - _Requirements: 14.3_

  - [x] 20.8 Integrate Clinical Inference Engine with Context Planner
    - Update `ContextPlannerService.assemble_briefing_packet()` to invoke ClinicalInferenceEngine for each retrieved snippet
    - Add 30-second overall timeout for inference engine calls
    - If inference engine unavailable: proceed without inferred facts, set `degraded_inference=True`, log warning
    - Package inferred facts in distinct `inferred_facts` section of BriefingPacket, separate from explicit facts
    - Each inferred fact displays confidence score and source inference chain
    - _Requirements: 14.1, 14.8, 14.9_

  - [x] 20.9 Write property test for inferred facts separation in Briefing Packet
    - **Property 30: Inferred Facts Separation in Briefing Packet**
    - Test that inferred facts are in a structurally distinct section from explicit facts, each includes confidence score and inference chain, no inferred fact appears in explicit facts section
    - **Validates: Requirements 14.8**

- [x] 21. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 22. Implement Frontend API Endpoints (server.py)
  - [x] 22.1 Implement BEACON status and Axisweave context API endpoints
    - Add `GET /beacon/status?request_id=<id>` endpoint to `server.py`: returns layers array with id, name, state, timestamp and current_layer index
    - Add `GET /axisweave/context?request_id=<id>` endpoint: returns chunks array with chunk_id, text, document_id, content_hash, relevance_score, kms_status, chunk_index, ingestion_timestamp
    - Wire to AuditTrailService and HybridRetrievalService respectively
    - _Requirements: 15.1, 15.3_

  - [x] 22.2 Implement Evidence Bundle and Graph API endpoints
    - Add `GET /evidence-bundle/<execution_id>` endpoint: returns execution_id, decision, reason, lineage_trail (each with conclusion, evidence_id, timestamp, confidence), signatures
    - Add `GET /graph/member/<member_id>` endpoint: returns nodes array (id, type, label, properties) and edges array (source, target, type, label)
    - Wire to EvidenceBundleService and CausalOntologyGraphService
    - _Requirements: 15.4, 15.5_

  - [x] 22.3 Implement SDOH Inference and Medical Director Queue API endpoints
    - Add `GET /inference/sdoh/<member_id>` endpoint: returns inferred_facts array (fact_id, type, category, conclusion, confidence, chain, source_text, origin) and explicit_facts array
    - Add `GET /md-queue` endpoint: returns cases array (case_id, briefing_summary, criteria_assessment, challenger_findings, trace_summary, escalated_at)
    - Wire to ClinicalInferenceEngine and HumanGateService/MedicalDirectorQueue
    - _Requirements: 15.6, 15.7_

- [x] 23. Implement Frontend Panels (JavaScript Modules)
  - [x] 23.1 Implement BEACONHarnessVisualization panel
    - Create `src/clinical_reasoning_fabric/frontend/beacon_harness_viz.js`
    - Implement 7-layer sequential left-to-right flow with arrow connectors (Identity, Context, MCP Gateway, Sandbox, Verification, Observability, Human Gates)
    - Layer states: pending, active, passed, failed; update within 2 seconds of actual transition
    - Expand existing 5-step pipeline pattern to 7 layers preserving glass-card CSS and sidebar navigation
    - Implement non-blocking error handling: show error message, retain previous state
    - _Requirements: 15.1, 15.2, 15.8_

  - [x] 23.2 Implement AxisweaveContextPanel
    - Create `src/clinical_reasoning_fabric/frontend/axisweave_context_panel.js`
    - Display up to 50 evidence chunks with provenance metadata
    - Each chunk shows: document_id, content_hash, relevance score (0.00-1.00), KMS signature status (valid/invalid)
    - Fetch from `/axisweave/context?request_id=...`
    - Implement non-blocking error: show message, retain previousContent
    - _Requirements: 15.3, 15.8, 15.9_

  - [x] 23.3 Implement EvidenceBundleViewer panel
    - Create `src/clinical_reasoning_fabric/frontend/evidence_bundle_viewer.js`
    - Display full Evidence Bundle with lineage trail
    - Each lineage entry: conclusion statement, source chunk link, retrieval timestamp, confidence score (0.00-1.00)
    - Fetch from `/evidence-bundle/:id`
    - Non-blocking error handling
    - _Requirements: 15.4, 15.8, 15.9_

  - [x] 23.4 Implement CausalGraphVisualization panel
    - Create `src/clinical_reasoning_fabric/frontend/causal_graph_viz.js`
    - Render member clinical state as visual node/edge graph
    - Node types: diagnoses, medications, sdoh_factors, policy_rules with typed styling
    - Edge types: HAS_CONDITION, IS_PRESCRIBED, TRIGGERED_BY, GOVERNED_BY, EVIDENCED_BY, INFERRED_FROM with labels
    - Support up to 200 nodes in rendered view
    - Fetch from `/graph/member/:id`; non-blocking error, retain previous graph
    - _Requirements: 15.5, 15.8, 15.9_

  - [x] 23.5 Implement SDOHInferenceDisplay panel
    - Create `src/clinical_reasoning_fabric/frontend/sdoh_inference_display.js`
    - Display inferred SDOH factors with: source text excerpt (up to 500 chars), complete inference chain steps, confidence score (0.00-1.00)
    - Visual origin indicator: labeled tag/icon distinguishing "EXPLICIT" vs "INFERRED" factors
    - Fetch from `/inference/sdoh/:id`; non-blocking error handling
    - _Requirements: 15.6, 15.8, 15.9_

  - [x] 23.6 Implement MedicalDirectorQueueView panel
    - Create `src/clinical_reasoning_fabric/frontend/md_queue_view.js`
    - Display escalated cases with all 4 required artifacts: Briefing Packet summary, criteria assessment (per-criterion: met/not_met/not_evaluated), OPA Challenger findings, execution trace summary
    - Fetch from `/md-queue`; non-blocking error handling
    - _Requirements: 15.7, 15.8, 15.9_

  - [x] 23.7 Implement shared PanelErrorHandler and integrate panels into sidebar navigation
    - Create `src/clinical_reasoning_fabric/frontend/panel_error_handler.js`
    - Implement non-blocking error display: show error with data source name, retain previous content, auto-dismiss after 10 seconds
    - Register all 6 panels in sidebar navigation within existing `index.html` layout
    - Ensure panels use glass-card CSS pattern and don't interfere with existing UI
    - _Requirements: 15.8, 15.9_

- [x] 24. Write property tests for Frontend Panels (Correctness Properties 31-36)
  - [x] 24.1 Write property test for evidence chunk rendering completeness
    - **Property 31: Evidence Chunk Rendering Completeness**
    - Test that for any evidence chunk displayed, rendered output contains: document_id, content_hash, relevance score (0.00-1.00), KMS signature status (valid/invalid); no required field omitted
    - **Validates: Requirements 15.3**

  - [x] 24.2 Write property test for lineage trail rendering completeness
    - **Property 32: Lineage Trail Rendering Completeness**
    - Test that for any lineage entry displayed, rendered output contains: conclusion statement, link to source chunk, retrieval timestamp, confidence score; each conclusion traceable to source
    - **Validates: Requirements 15.4**

  - [x] 24.3 Write property test for graph visualization node capacity
    - **Property 33: Graph Visualization Node Capacity**
    - Test that for any member state with N nodes (N ≤ 200), visualization renders exactly N typed nodes with correct type labels and all corresponding labeled directed edges
    - **Validates: Requirements 15.5**

  - [x] 24.4 Write property test for SDOH display completeness
    - **Property 34: SDOH Display Completeness**
    - Test that for any inferred SDOH factor displayed: includes source text (up to 500 chars), complete inference chain, confidence score (0.00-1.00), visual origin indicator distinguishing from explicit factors
    - **Validates: Requirements 15.6**

  - [x] 24.5 Write property test for Medical Director queue artifact completeness
    - **Property 35: Medical Director Queue Artifact Completeness**
    - Test that for any escalated case displayed: all four artifact sections rendered (Briefing Packet summary, criteria assessment with per-criterion status, OPA findings, trace summary) — all non-null
    - **Validates: Requirements 15.7**

  - [x] 24.6 Write property test for panel error graceful degradation
    - **Property 36: Panel Error Graceful Degradation**
    - Test that for any panel receiving a backend error: displays non-blocking error identifying failed data source, retains previously loaded content, does not cause errors in other panels
    - **Validates: Requirements 15.9**

- [x] 25. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 26. Integration wiring and end-to-end PA request flow
  - [x] 26.1 Wire all services together into the CRF orchestrator
    - Create `src/clinical_reasoning_fabric/orchestrator.py`
    - Wire the full PA request flow: Identity → Context Planner → Clinical Inference → Agent Reasoning → MCP Gateway → OPA Challenger → Human Gates → Evidence Bundle
    - Inject AuditTrailService at every step to record trace entries
    - Handle error propagation: trace failure halts processing, KMS failure escalates, queue failure retries
    - Integrate with existing `server.py` endpoint for PA request processing
    - Ensure all components use the same authenticated identity context throughout the request
    - Wire Axisweave Service API as standalone microservice endpoint separate from PA flow
    - _Requirements: 1.1-1.8, 2.1-2.7, 3.1-3.5, 4.1-4.7, 5.1-5.5, 6.1-6.6, 7.1-7.6, 8.1-8.6, 9.1-9.6, 10.1-10.5, 11.1-11.7, 12.1-12.7, 13.1-13.9, 14.1-14.9_

  - [x] 26.2 Write integration tests for end-to-end PA flow
    - Test auto-approve path: all criteria MET, all signatures valid, verification PASS
    - Test escalation path: criterion NOT_MET, evidence bundle with complete artifacts to MD
    - Test error paths: KMS unavailable → escalate, trace failure → halt, queue unavailable → retry
    - Test no-automated-denial invariant across all scenarios
    - Test inference engine integration: inferred SDOH factors appear in Briefing Packet
    - Test Axisweave API multi-tenant namespace isolation
    - _Requirements: 9.1, 9.2, 9.3, 7.1, 8.5, 13.5, 14.1_

  - [x] 26.3 Write integration tests for frontend panel API integration
    - Test all 6 API endpoints return correct data format
    - Test panel error handling with simulated backend failures
    - Test BEACON harness visualization updates within 2-second constraint
    - Test Medical Director queue view shows all 4 artifacts
    - _Requirements: 15.1, 15.3, 15.4, 15.5, 15.6, 15.7, 15.9_

- [x] 27. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis (min 100 examples per property)
- Unit tests validate specific examples and edge cases
- The implementation uses Python throughout for backend, consistent with the existing codebase (server.py, agent_engine.py, challenger_agent.py)
- Frontend panels use vanilla JavaScript with glass-card CSS pattern, consistent with existing index.html and app.js
- External service integrations (KMS, Qdrant, Neo4j, Dagster) should be abstracted behind interfaces for testability
- All property tests follow the tag format: `Feature: clinical-reasoning-fabric, Property {N}: {title}`
- Tasks 18.x (Axisweave Service API) can run in parallel with tasks 20.x (Clinical Inference Engine) since they have no mutual dependencies
- Frontend tasks (22-24) depend on both API endpoints (server.py) and the underlying services being implemented

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "5.1", "6.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "5.2", "6.2"] },
    { "id": 4, "tasks": ["2.4", "2.5", "6.3"] },
    { "id": 5, "tasks": ["2.6", "4.1"] },
    { "id": 6, "tasks": ["4.2", "4.3"] },
    { "id": 7, "tasks": ["4.4", "8.1"] },
    { "id": 8, "tasks": ["8.2", "9.1"] },
    { "id": 9, "tasks": ["9.2", "10.1"] },
    { "id": 10, "tasks": ["10.2", "11.1"] },
    { "id": 11, "tasks": ["11.2", "13.1"] },
    { "id": 12, "tasks": ["13.2", "14.1"] },
    { "id": 13, "tasks": ["14.2", "15.1"] },
    { "id": 14, "tasks": ["15.2", "16.1"] },
    { "id": 15, "tasks": ["16.2", "18.1", "20.1"] },
    { "id": 16, "tasks": ["18.2", "18.3", "20.2", "20.3"] },
    { "id": 17, "tasks": ["18.4", "18.5", "20.4", "20.5"] },
    { "id": 18, "tasks": ["18.6", "18.7", "20.6", "20.7"] },
    { "id": 19, "tasks": ["18.8", "18.9", "20.8"] },
    { "id": 20, "tasks": ["20.9", "22.1"] },
    { "id": 21, "tasks": ["22.2", "22.3"] },
    { "id": 22, "tasks": ["23.1", "23.2", "23.3"] },
    { "id": 23, "tasks": ["23.4", "23.5", "23.6"] },
    { "id": 24, "tasks": ["23.7", "24.1", "24.2"] },
    { "id": 25, "tasks": ["24.3", "24.4", "24.5"] },
    { "id": 26, "tasks": ["24.6", "26.1"] },
    { "id": 27, "tasks": ["26.2", "26.3"] }
  ]
}
```
