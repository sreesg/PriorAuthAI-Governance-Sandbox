# Requirements Document

## Introduction

The Clinical Reasoning Fabric (CRF) is a unified architecture for clinical AI agents that orchestrates Prior Authorization (PA) decision support. The system integrates three decoupled layers — the Axisweave Retrieval Stack for semantic document retrieval, the Neo4j Causal Ontology Graph for active clinical state management, and the BEACON 7-Layer Safety Harness for controlled, auditable LLM orchestration. The primary application is generating Prior Authorization Evidence Bundles that route to human Medical Directors for final disposition, with automated approvals only when all criteria are unambiguously satisfied.

## Glossary

- **CRF**: Clinical Reasoning Fabric — the unified architecture combining retrieval, graph state, and safety layers
- **Axisweave_Retrieval_Stack**: The semantic retrieval fabric responsible for PDF parsing, chunking, PII scrubbing, cryptographic signing, and hybrid vector search
- **Causal_Ontology_Graph**: The Neo4j graph layer storing active clinical state including member conditions, prescriptions, SDOH factors, and policy rules
- **BEACON_Harness**: The 7-layer safety envelope wrapping LLM interactions with identity control, context restriction, tool orchestration, sandboxed execution, verification, observability, and human gates
- **Briefing_Packet**: A structured JSON object containing the patient clinical state, verified evidence snippets, and provenance metadata assembled before agent reasoning
- **Evidence_Bundle**: The output package containing the agent decision, reasoning lineage trail, and original document signatures for human review
- **Context_Planner**: Layer 2 component that restricts model visibility to only relevant clinical context
- **MCP_Gateway**: Layer 3 component providing a curated catalog of approved tools the agent may invoke
- **OPA_Challenger_Agent**: Layer 5 verification component that independently validates KMS signatures, content hashes, and policy compliance
- **Medical_Director**: The human physician who reviews escalated or ambiguous PA decisions
- **KMS_Signer**: AWS Key Management Service component that signs document hashes for tamper-evident provenance
- **Qdrant_VectorDB**: The vector database storing semantically chunked and signed clinical document embeddings with BM25 sparse indices
- **CDC_Pipeline**: Change Data Capture pipeline using dbt and Dagster to project current state from Snowflake/Iceberg into Neo4j
- **Reciprocal_Rank_Fusion**: The hybrid search scoring method combining dense vector similarity and BM25 lexical matching
- **SDOH_Factor**: Social Determinants of Health data points tracked in the graph (housing, food security, transportation)
- **PII_Scrubber**: Component that removes personally identifiable information from documents before vector storage
- **Axisweave_Service_API**: The versioned REST or gRPC service interface exposing Axisweave document ingestion, retrieval, and provenance verification to external consumers
- **Clinical_Inference_Engine**: The LLM-powered component that derives implied clinical conclusions from contextual clues in clinical notes, producing inferred SDOH factors and clinical implications
- **Inference_Chain**: An ordered sequence of reasoning steps linking a source text observation to a derived clinical conclusion, with confidence scores at each step
- **BEACON_Harness_Visualization**: Frontend component displaying the full 7-layer BEACON harness execution status with progressive disclosure of each layer
- **Axisweave_Context_Panel**: Frontend component showing retrieved evidence chunks with provenance metadata, relevance scores, and KMS signature verification status
- **Evidence_Bundle_Viewer**: Frontend component displaying the full Evidence Bundle with lineage trail linking each conclusion to its source evidence chunk
- **Causal_Graph_Visualization**: Frontend component rendering the member active clinical state from Neo4j as a visual graph with nodes and edges
- **SDOH_Inference_Display**: Frontend component showing inferred SDOH factors with source text, inference chain, and confidence scores
- **Medical_Director_Queue_View**: Frontend view for Medical Directors showing escalated cases with the full artifact package

## Requirements

### Requirement 1: Document Ingestion and Cryptographic Provenance

**User Story:** As a compliance officer, I want all clinical documents to be parsed, scrubbed of PII, hashed, and cryptographically signed upon ingestion, so that the system maintains tamper-evident provenance for every piece of evidence used in PA decisions.

#### Acceptance Criteria

1. WHEN a clinical PDF document is submitted for ingestion, THE Axisweave_Retrieval_Stack SHALL parse the document into structured text using Docling PDF parsing
2. WHEN document text is extracted, THE PII_Scrubber SHALL remove all HIPAA Safe Harbor identifiers including names, geographic data smaller than state, dates (except year), phone numbers, email addresses, SSNs, medical record numbers, health plan numbers, account numbers, device identifiers, URLs, IP addresses, biometric identifiers, photographic images, and any other unique identifying number or code before vector storage
3. WHEN a document is scrubbed, THE Axisweave_Retrieval_Stack SHALL compute a SHA-256 content hash of the cleaned text
4. WHEN a content hash is computed, THE KMS_Signer SHALL sign the hash with an AWS KMS asymmetric key and attach the signature to the document record
5. WHEN a signed document is ready for storage, THE Axisweave_Retrieval_Stack SHALL chunk the text into semantic segments using Chonkie semantic chunking and store each chunk in Qdrant_VectorDB with provenance metadata containing the source document_id, content_hash, KMS_signature, chunk_index, and ingestion_timestamp
6. IF document parsing fails due to corrupted or unsupported format, THEN THE Axisweave_Retrieval_Stack SHALL reject the document with an error indicating the failure reason and log the failure to the audit trail
7. IF the KMS_Signer is unavailable or signing fails, THEN THE Axisweave_Retrieval_Stack SHALL halt ingestion for that document, discard any unsigned chunks, and log the signing failure to the audit trail
8. IF PII scrubbing fails or cannot be completed, THEN THE PII_Scrubber SHALL halt document processing, prevent the document from being stored, and log the scrubbing failure to the audit trail

### Requirement 2: Hybrid Semantic Retrieval

**User Story:** As a clinical AI agent, I want to retrieve the most relevant clinical evidence using both semantic and lexical matching, so that I can construct accurate Briefing Packets with high-recall evidence retrieval.

#### Acceptance Criteria

1. WHEN a retrieval query is submitted, THE Qdrant_VectorDB SHALL execute a dense vector similarity search using the query embedding and return the top 50 candidate chunks ranked by cosine similarity
2. WHEN a retrieval query is submitted, THE Qdrant_VectorDB SHALL execute a BM25 sparse index search using the query terms and return the top 50 candidate chunks ranked by BM25 score
3. WHEN both dense and sparse results are returned, THE Axisweave_Retrieval_Stack SHALL combine scores using Reciprocal_Rank_Fusion to produce a unified ranked result set limited to the top 20 chunks
4. THE Axisweave_Retrieval_Stack SHALL return only chunks whose KMS signatures are verified as valid before inclusion in any Briefing_Packet
5. IF a chunk signature verification fails, THEN THE Axisweave_Retrieval_Stack SHALL exclude the chunk from results and log a tamper alert to the observability layer
6. IF either the dense vector search or the BM25 sparse search fails to return results within 10 seconds, THEN THE Axisweave_Retrieval_Stack SHALL proceed with results from the available search method and log a degraded retrieval warning to the observability layer
7. IF both dense and sparse searches return zero matching chunks after Reciprocal_Rank_Fusion, THEN THE Axisweave_Retrieval_Stack SHALL return an empty result set with a no-evidence-found indicator to the Context_Planner

### Requirement 3: Causal Ontology Graph State Management

**User Story:** As a clinical reasoning agent, I want access to the current active clinical state of a member through a causal graph, so that I can reason over diagnoses, prescriptions, SDOH factors, and policy relationships without querying raw transaction history.

#### Acceptance Criteria

1. THE Causal_Ontology_Graph SHALL store Member, Event, PolicyRule, SDOH_Factor, and EvidenceSource as typed nodes
2. THE Causal_Ontology_Graph SHALL store HAS_CONDITION (Member→Event), IS_PRESCRIBED (Member→Event), TRIGGERED_BY (Event→Event), GOVERNED_BY (Event→PolicyRule), and EVIDENCED_BY (Event→EvidenceSource) as typed directed relationships between nodes
3. WHEN the CDC_Pipeline detects a state change in Snowflake or Iceberg source tables, THE Causal_Ontology_Graph SHALL update the corresponding node or relationship within 5 minutes
4. WHEN a member query is issued, THE Causal_Ontology_Graph SHALL return the active clinical state including current diagnoses, active prescriptions, and linked SDOH factors within 2 seconds
5. THE Causal_Ontology_Graph SHALL represent only active clinical state, where a record is considered active if it has not been marked as resolved, discontinued, or superseded by a subsequent CDC event, and SHALL exclude any record whose status has been set to closed, resolved, or discontinued in the source system

### Requirement 4: Briefing Packet Assembly

**User Story:** As a clinical AI agent, I want a pre-assembled Briefing Packet containing the member profile, relevant evidence snippets, and policy context, so that I can perform bounded reasoning within a controlled information scope.

#### Acceptance Criteria

1. WHEN a PA request is received, THE Context_Planner SHALL query the Causal_Ontology_Graph for the member active clinical state including diagnoses and SDOH factors
2. WHEN member state is retrieved, THE Context_Planner SHALL query the Qdrant_VectorDB for relevant clinical note snippets using the CPT code and diagnosis context, returning a maximum of 20 snippets with a minimum relevance score threshold of 0.5
3. WHEN both graph state and evidence snippets are assembled, THE Context_Planner SHALL package the data into a Briefing_Packet conforming to the defined JSON schema containing request_id, member_id, cpt_code, active_clinical_state, and verified_evidence_snippets with provenance
4. THE Context_Planner SHALL restrict the Briefing_Packet to only information relevant to the specific PA request by filtering to diagnoses, medications, and evidence matching the requested CPT code or its associated clinical condition categories
5. IF the member is not found in the Causal_Ontology_Graph, THEN THE Context_Planner SHALL return an error indicating the member state is unavailable and halt processing
6. THE Context_Planner SHALL complete Briefing_Packet assembly within 30 seconds of receiving the PA request; IF assembly exceeds this timeout, THEN the Context_Planner SHALL halt processing and log a timeout error
7. IF the Qdrant retrieval returns zero evidence snippets for the PA request, THEN THE Context_Planner SHALL assemble the Briefing_Packet with an empty verified_evidence_snippets array and flag the packet with a no-evidence-found indicator for downstream handling

### Requirement 5: BEACON Layer 1 — Identity and Permissions

**User Story:** As a security administrator, I want all agent interactions to enforce PII/PHI masking and role-based access control, so that sensitive patient data is protected throughout the reasoning pipeline.

#### Acceptance Criteria

1. THE BEACON_Harness SHALL replace all PII and PHI fields in trace logs and observability outputs with irreversible masked tokens before they are persisted, such that no original identifiable value is recoverable from persisted outputs
2. WHEN an agent request is initiated, THE BEACON_Harness SHALL authenticate the requesting identity and verify it against the RBAC policy before granting access to clinical data, ensuring no clinical data is returned or processed until verification succeeds
3. IF an identity lacks sufficient permissions for the requested operation, THEN THE BEACON_Harness SHALL deny the request without exposing any clinical data in the response and log an unauthorized access attempt containing the requesting identity, requested operation, timestamp, and the permission that was missing
4. THE BEACON_Harness SHALL associate every agent action with the authenticated identity identifier in the execution trace so that each trace entry is attributable to a specific identity for audit purposes
5. IF authentication fails due to missing, invalid, or expired credentials, THEN THE BEACON_Harness SHALL reject the request without exposing clinical data and log the failed authentication attempt containing the timestamp and reason for failure

### Requirement 6: BEACON Layer 3 — Tool Orchestration via MCP Gateway

**User Story:** As a platform architect, I want the agent to access only pre-approved tools through a controlled gateway, so that the agent cannot invoke arbitrary external services or perform uncontrolled actions.

#### Acceptance Criteria

1. THE MCP_Gateway SHALL maintain a catalog of approved tools where each entry specifies the tool name, permitted input parameter schema, and a description of the tool's function
2. WHEN the agent requests a tool invocation, THE MCP_Gateway SHALL validate that the requested tool name exists in the approved catalog and that the supplied input parameters conform to the tool's permitted parameter schema before execution
3. IF the agent requests a tool not present in the approved catalog, THEN THE MCP_Gateway SHALL reject the invocation and log the unauthorized tool request including the agent identity, requested tool name, and timestamp to the execution trace
4. WHEN a tool invocation is approved, THE MCP_Gateway SHALL execute the call within the sandboxed execution environment defined by Layer 4 subject to a configurable per-tool timeout that defaults to 30 seconds
5. THE MCP_Gateway SHALL record the tool name, input parameters, output result, invocation duration, and success or failure status for every invocation in the execution trace
6. IF an approved tool invocation fails due to timeout, crash, or error response, THEN THE MCP_Gateway SHALL record the failure in the execution trace with the error category and return a structured error indication to the agent without retrying

### Requirement 7: BEACON Layer 5 — Verification Loops

**User Story:** As a compliance officer, I want an independent verification agent to validate that all evidence used in decisions has valid cryptographic provenance and complies with policy rules, so that the system cannot produce decisions based on tampered or unauthorized data.

#### Acceptance Criteria

1. WHEN the agent produces a decision, THE OPA_Challenger_Agent SHALL verify that every evidence snippet referenced in the Evidence_Bundle has a valid KMS signature matching its content hash and SHALL complete all signature verification within 30 seconds of decision receipt
2. WHEN signature verification completes with all signatures valid, THE OPA_Challenger_Agent SHALL evaluate the decision against OPA policy rules defined in rules.rego and produce a verification result of PASS or FAIL within 10 seconds
3. IF any evidence signature is invalid or missing, THEN THE OPA_Challenger_Agent SHALL reject the decision and escalate to the Medical_Director with a tamper alert that identifies each evidence snippet with an invalid or missing signature and its corresponding content hash
4. IF OPA policy evaluation identifies a rule violation, THEN THE OPA_Challenger_Agent SHALL route the decision to the Medical_Director queue for human review with each violated rule identifier and a description of the violation cited
5. THE OPA_Challenger_Agent SHALL operate independently from the primary reasoning agent with no shared mutable state
6. IF the OPA_Challenger_Agent is unable to complete verification due to KMS unavailability, rules.rego loading failure, or internal error, THEN THE OPA_Challenger_Agent SHALL treat the verification as FAIL, halt the decision from proceeding, and escalate to the Medical_Director with an indication that verification could not be completed

### Requirement 8: BEACON Layer 6 — Observability and Immutable Audit Trail

**User Story:** As an auditor, I want every agent action, tool invocation, and decision step recorded in an immutable execution trace, so that any PA decision can be fully reconstructed and audited after the fact.

#### Acceptance Criteria

1. THE BEACON_Harness SHALL record an immutable execution trace for every PA request containing entries categorized as one of: agent_action, tool_invocation, context_retrieval, or decision_step, with each entry assigned a monotonically increasing sequence number to preserve execution order
2. THE BEACON_Harness SHALL include a UTC ISO-8601 timestamp with millisecond precision, request_id correlation, the authenticated identity, and the entry category in every trace entry
3. WHEN a PA decision is produced, THE BEACON_Harness SHALL attach the execution trace containing all entries from request initiation through decision production to the Evidence_Bundle
4. THE BEACON_Harness SHALL store execution traces in append-only storage that does not permit modification or deletion of historical entries and SHALL retain traces for a minimum of 7 years
5. IF trace recording fails, THEN THE BEACON_Harness SHALL halt the PA request processing and return an error indicating trace failure rather than produce an unaudited decision
6. THE BEACON_Harness SHALL support retrieval of the complete execution trace by request_id for audit review within 30 seconds of query submission

### Requirement 9: BEACON Layer 7 — Human Gates and No Automated Denials

**User Story:** As a Medical Director, I want the system to never automatically deny a PA request and always route ambiguous or failed cases to me for review, so that patient care decisions remain under physician oversight.

#### Acceptance Criteria

1. WHEN all clinical necessity criteria evaluate to MET and the OPA_Challenger_Agent verification passes with no challenges issued, THE BEACON_Harness SHALL auto-approve the PA request without requiring Medical_Director review
2. THE BEACON_Harness SHALL prohibit automated denial of any PA request regardless of criteria evaluation outcome; the only automated disposition SHALL be approval
3. WHEN at least one clinical necessity criterion evaluates to NOT_MET or INDETERMINATE, THE BEACON_Harness SHALL escalate the case to the Medical_Director queue with the full Evidence_Bundle within 30 seconds of determination
4. WHEN the OPA_Challenger_Agent issues a formal challenge, THE BEACON_Harness SHALL route the case to the Medical_Director queue with the challenge findings attached
5. WHEN a case is escalated or routed to the Medical_Director, THE BEACON_Harness SHALL include all four artifacts: Briefing_Packet, criteria assessment with per-criterion MET/NOT_MET/INDETERMINATE status, OPA_Challenger_Agent findings, and the complete execution trace
6. IF the Medical_Director queue is unavailable when escalation is attempted, THEN THE BEACON_Harness SHALL hold the PA request in a pending state without issuing any automated decision and retry delivery to the queue at 30-second intervals up to 10 attempts, logging each failed attempt to the audit trail

### Requirement 10: Evidence Bundle Output and Lineage

**User Story:** As a Medical Director, I want every PA decision to include a complete evidence package with the decision rationale, lineage trail, and original document signatures, so that I can verify the basis of any recommendation.

#### Acceptance Criteria

1. WHEN the agent completes reasoning, THE CRF SHALL produce an Evidence_Bundle conforming to the defined output schema containing execution_id, decision, reason, lineage_trail, and original_document_signatures
2. THE Evidence_Bundle SHALL include the lineage_trail as an ordered array of entries, where each entry contains the conclusion statement, the identifier of the evidence snippet or graph query that produced it, and the retrieval timestamp
3. THE Evidence_Bundle SHALL include the original_document_signatures array containing the KMS signature for every source document referenced
4. WHEN an Evidence_Bundle is produced, THE CRF SHALL validate that all required schema fields are present, non-null, and of the correct data type, and that the lineage_trail contains at least one entry and the original_document_signatures array contains at least one signature, before routing to the Medical_Director or auto-approval
5. IF the Evidence_Bundle fails schema validation, THEN THE CRF SHALL halt the decision, log a bundle integrity error identifying the missing or invalid fields, and escalate the PA request to the Medical_Director queue with the validation failure reason attached

### Requirement 11: Bounded Agent Reasoning with Targeted Retrieval

**User Story:** As a clinical AI agent, I want to perform targeted retrieval calls during reasoning to gather additional evidence beyond the initial Briefing Packet, so that I can build complete clinical justifications while remaining bounded by the MCP tool catalog.

#### Acceptance Criteria

1. WHEN the agent requests additional evidence during reasoning, THE CRF SHALL permit the agent to make targeted Qdrant retrieval calls exclusively through the MCP_Gateway using the approved retrieval tool from the tool catalog
2. WHEN the agent retrieves additional evidence, THE CRF SHALL verify the KMS signatures of newly retrieved chunks before incorporating them into reasoning
3. IF a newly retrieved chunk fails KMS signature verification, THEN THE CRF SHALL discard the chunk, exclude it from reasoning, and log a tamper alert to the observability layer
4. WHEN the agent updates the clinical graph during reasoning, THE Causal_Ontology_Graph SHALL record the update with the agent execution_id as provenance
5. THE CRF SHALL limit the total number of retrieval calls per PA request to a configurable maximum with a default of 10 calls, where the configurable range is 1 to 50
6. IF the retrieval call limit is reached, THEN THE CRF SHALL proceed to decision with available evidence and record a retrieval_limit_reached event in the execution trace including the number of calls made and the limit value
7. WHEN the agent completes reasoning that included additional retrieval calls, THE CRF SHALL include all additionally retrieved evidence snippets with their provenance metadata in the Evidence_Bundle lineage_trail

### Requirement 12: CDC Pipeline State Projection

**User Story:** As a data engineer, I want clinical state changes from Snowflake and Iceberg to be projected into the Neo4j graph through a reliable CDC pipeline, so that the Causal Ontology Graph reflects current patient state for real-time agent reasoning.

#### Acceptance Criteria

1. WHEN a record is inserted or updated in Snowflake or Iceberg source tables that map to Member, Event, PolicyRule, SDOH_Factor, or EvidenceSource node types, THE CDC_Pipeline SHALL detect the change within 5 minutes of the source commit timestamp
2. WHEN a change is detected, THE CDC_Pipeline SHALL transform the record into the corresponding Causal_Ontology_Graph node or relationship type (Member, Event, PolicyRule, SDOH_Factor, EvidenceSource, HAS_CONDITION, IS_PRESCRIBED, TRIGGERED_BY, GOVERNED_BY, or EVIDENCED_BY) using dbt models
3. WHEN transformation completes, THE CDC_Pipeline SHALL apply the change to the Causal_Ontology_Graph as an upsert operation that updates the target node or relationship properties without removing or modifying relationships not referenced in the current change event
4. IF a CDC event fails to apply to the graph, THEN THE CDC_Pipeline SHALL retry the operation up to 3 times with exponential backoff starting at a base interval of 5 seconds before alerting the operations team
5. THE CDC_Pipeline SHALL maintain an event checkpoint so that pipeline restarts resume from the last successfully processed event without duplicating graph mutations
6. IF a source record cannot be mapped to a valid graph node or relationship type during transformation, THEN THE CDC_Pipeline SHALL skip the record, log the failure with the source record identifier and reason to the audit trail, and continue processing subsequent events
7. WHEN multiple change events arrive for the same entity, THE CDC_Pipeline SHALL apply them in source-commit-timestamp order to ensure the Causal_Ontology_Graph reflects the latest state


### Requirement 13: Axisweave as Reusable Standalone Service

**User Story:** As a platform architect, I want the Axisweave Retrieval Stack to be a standalone, decoupled microservice with a versioned API, so that any future clinical use case (HEDIS Gap Closure, Fraud Detection, Care Management) can leverage document ingestion, retrieval, and provenance verification independently without coupling to the PA workflow.

#### Acceptance Criteria

1. THE Axisweave_Service_API SHALL expose document ingestion, hybrid retrieval, and provenance verification as independent API operations accessible through a versioned REST or gRPC interface, where each operation is callable independently without requiring prior invocation of the other operations
2. THE Axisweave_Service_API SHALL accept document ingestion requests with use-case agnostic metadata, where the caller specifies a namespace identifier (a non-empty string of 1 to 128 alphanumeric, hyphen, or underscore characters) and a document category (a non-empty string of 1 to 64 characters) without requiring PA-specific fields
3. WHEN a consumer submits a retrieval request, THE Axisweave_Service_API SHALL execute hybrid search scoped to the caller-specified namespace and return results with full provenance metadata (document_id, content_hash, KMS_signature, chunk_index, and ingestion_timestamp) within 10 seconds of request receipt regardless of which use case initiated ingestion
4. THE Axisweave_Retrieval_Stack SHALL support independent deployment and horizontal scaling of ingestion, retrieval, and signing components without requiring co-deployment with the BEACON_Harness or Causal_Ontology_Graph
5. THE Axisweave_Service_API SHALL enforce tenant and namespace isolation such that retrieval queries from one use case return only documents ingested under that use case namespace unless explicit cross-namespace access has been granted through a caller-provided access token or API key that includes the target namespace in its authorized scope
6. THE Axisweave_Service_API SHALL maintain a versioned API contract with semantic versioning where breaking changes increment the major version and non-breaking additions increment the minor version, and SHALL support the prior major version for a minimum of 6 months after a new major version is released
7. WHEN a shared vector index is configured across multiple use-case namespaces, THE Axisweave_Retrieval_Stack SHALL partition retrieval results by namespace metadata and apply namespace-scoped access control before returning results to the caller, ensuring no chunks from unauthorized namespaces are included in the response
8. IF a consumer submits a request without a valid namespace identifier or with a namespace identifier that does not conform to the format of 1 to 128 alphanumeric, hyphen, or underscore characters, THEN THE Axisweave_Service_API SHALL reject the request with an error indicating the invalid or missing namespace and log the rejection to the audit trail
9. IF a consumer submits a request with invalid or missing authentication credentials, THEN THE Axisweave_Service_API SHALL reject the request without exposing any stored document data and log the authentication failure including the timestamp and source identifier to the audit trail

### Requirement 14: Intelligent Clinical Inference in Semantic Retrieval

**User Story:** As a clinical reasoning agent, I want the semantic search layer to derive implied clinical conclusions from contextual clues in clinical notes, so that the system captures SDOH factors, medication adherence risks, and care barriers that are implied but not explicitly stated in clinical documentation.

#### Acceptance Criteria

1. WHEN the Context_Planner assembles a Briefing_Packet and retrieves clinical note snippets from the Qdrant_VectorDB, THE Clinical_Inference_Engine SHALL analyze each retrieved snippet to derive implied clinical conclusions including SDOH factors, medication adherence risks, and care access barriers from the source text within 15 seconds per snippet
2. WHEN the Clinical_Inference_Engine produces an inferred fact, THE Clinical_Inference_Engine SHALL tag the inferred fact with a distinct inference_type label from the set (sdoh_factor, medication_adherence_risk, care_access_barrier) separating it from explicitly stated facts and assign a confidence score between 0.0 and 1.0
3. WHEN an inferred SDOH factor is produced with a confidence score at or above the configured threshold, THE Clinical_Inference_Engine SHALL link the inferred factor to the corresponding node in the Causal_Ontology_Graph with an INFERRED_FROM relationship containing the source text, inference chain, and confidence score
4. THE Clinical_Inference_Engine SHALL support configurable inference depth where shallow depth produces only direct single-hop implications and deep depth produces multi-hop reasoning chains up to a configurable maximum of 3 hops, with the default depth set to shallow
5. WHEN the Clinical_Inference_Engine performs multi-hop inference, THE Clinical_Inference_Engine SHALL produce an Inference_Chain containing each reasoning step with its intermediate conclusion, the evidence supporting that step, and the confidence score at each hop, where the cumulative confidence is the product of individual hop confidence scores
6. THE Clinical_Inference_Engine SHALL derive SDOH factors from the following categories: housing instability, transportation barriers, medication storage limitations, food insecurity, and caregiver availability using LLM-powered clinical reasoning, producing a maximum of 10 inferred facts per clinical note snippet
7. IF the Clinical_Inference_Engine cannot derive an inference with a confidence score above a configurable threshold with a default of 0.3, THEN THE Clinical_Inference_Engine SHALL discard the inference and not include it in downstream reasoning or graph updates
8. WHEN inferred facts are included in a Briefing_Packet, THE Context_Planner SHALL present inferred facts in a distinct section separate from explicitly stated facts, with each inferred fact displaying its confidence score and source inference chain
9. IF the Clinical_Inference_Engine is unavailable or fails to produce a response within 30 seconds for a given snippet, THEN THE Context_Planner SHALL proceed with Briefing_Packet assembly without inferred facts for that snippet, include a degraded_inference indicator in the Briefing_Packet, and log the failure to the observability layer

### Requirement 15: Frontend Application Integration

**User Story:** As a clinical operations user, I want the existing frontend application to display the full BEACON harness execution, Axisweave evidence context, causal graph state, SDOH inferences, and Medical Director escalation queue, so that I have complete visibility into the clinical reasoning process and its supporting evidence.

#### Acceptance Criteria

1. WHEN a PA request is processed through the BEACON harness, THE BEACON_Harness_Visualization SHALL display the full 7-layer execution status with progressive disclosure showing each layer's processing state as pending, active, passed, or failed, updating each layer's displayed state within 2 seconds of the actual state transition occurring
2. THE BEACON_Harness_Visualization SHALL expand the existing 5-step pipeline flow (PHI, Coverage, Evidence, Rego, Notice) to display all 7 BEACON layers (Identity, Context, MCP Gateway, Sandbox, Verification, Observability, Human Gates) while preserving the existing pipeline visual pattern of sequential left-to-right flow steps with arrow connectors and step labels
3. WHEN evidence retrieval completes, THE Axisweave_Context_Panel SHALL display each retrieved evidence chunk (up to 50 chunks) with its provenance metadata including document_id, content_hash, relevance score (displayed as a value from 0.00 to 1.00), and KMS signature verification status (valid or invalid)
4. WHEN the Evidence Bundle is produced, THE Evidence_Bundle_Viewer SHALL display the full lineage trail linking each conclusion statement to its source evidence chunk with retrieval timestamp and a numeric confidence score ranging from 0.00 to 1.00
5. WHEN a member active state is queried from the Causal_Ontology_Graph, THE Causal_Graph_Visualization SHALL render the clinical state as a visual graph with typed nodes (diagnoses, medications, SDOH factors, policy rules) and labeled directed edges representing their relationships, supporting up to 200 nodes in the rendered view
6. WHEN inferred SDOH factors are produced by the Clinical_Inference_Engine, THE SDOH_Inference_Display SHALL show each inferred factor with its source text excerpt (up to 500 characters), complete inference chain, confidence score (0.00 to 1.00), and a visual distinction between explicitly stated and inferred factors using a labeled indicator (such as a tag or icon) that identifies the factor's origin type
7. WHEN a case is escalated to the Medical Director queue, THE Medical_Director_Queue_View SHALL display the escalated case with all four artifacts: Briefing Packet summary, criteria assessment with per-criterion status (met, not met, or not evaluated), OPA Challenger findings, and execution trace summary
8. THE BEACON_Harness_Visualization, Axisweave_Context_Panel, Evidence_Bundle_Viewer, Causal_Graph_Visualization, SDOH_Inference_Display, and Medical_Director_Queue_View SHALL integrate as additional views and cards within the existing frontend layout preserving the vanilla JavaScript module architecture, glass-card CSS pattern, and sidebar navigation structure
9. IF the backend data source for any panel (BEACON_Harness_Visualization, Axisweave_Context_Panel, Evidence_Bundle_Viewer, Causal_Graph_Visualization, SDOH_Inference_Display, or Medical_Director_Queue_View) is unavailable or returns an error, THEN THE respective panel SHALL display a non-blocking error message indicating the data source that failed and retain any previously loaded content without clearing the panel
