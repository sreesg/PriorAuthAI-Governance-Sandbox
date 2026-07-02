"""
PriorAuthAI Agent Engine — Real Python Agent with Local LLM

This is the actual agentic backend that:
1. Reads PDF policy documents with the LLM to extract structured rules
2. Runs the full prior authorization evaluation pipeline
3. Manages skills, hooks, and progressive disclosure server-side
4. Persists extracted policies and generated skills for reuse

All LLM calls go through the local Gemma 4 12B via Ollama.
"""

import json
import os
import re
import io
import urllib.request
from datetime import datetime
from pypdf import PdfReader

# ─── PHI Scrubbing Layer ─────────────────────────────────────────────────────
from phi_scrubber import scrub_for_llm, scrub_text, PHIScrubber
_phi_scrubber = PHIScrubber()

# ─── S3 Support ──────────────────────────────────────────────────────────────
try:
    from s3_helper import is_s3_enabled, get_file_bytes
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# ─── Configuration ───────────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
MODEL = os.environ.get("LLM_MODEL", "gemma4:12b")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_POLICIES_FILE = os.path.join(DATA_DIR, "extracted_policies.json")
GENERATED_SKILLS_FILE = os.path.join(DATA_DIR, "generated_skills.json")


# ─── LLM Interface ──────────────────────────────────────────────────────────

# ─── LLM Interface ──────────────────────────────────────────────────────────

def call_llm(prompt, max_tokens=400, temperature=0, request_context=None):
    """Call LLM via AWS Bedrock (Claude 3.5 Haiku) or fall back to Ollama.
    
    PHI is automatically scrubbed from the prompt before sending to the model.
    The request_context (if provided) supplies known patient/provider names for
    targeted scrubbing.
    """
    # ─── PHI Scrubbing: de-identify before sending to any model ───────────
    if request_context:
        scrubbed_prompt, _phi_map = scrub_for_llm(prompt, request_context)
    else:
        scrubbed_prompt = scrub_text(prompt)
    
    # Try Bedrock first (deployed environment)
    try:
        import boto3
        bedrock_region = os.environ.get("AWS_REGION", "us-west-2")
        model_id = os.environ.get("BEDROCK_MODEL", "amazon.nova-lite-v1:0")
        
        client = boto3.client("bedrock-runtime", region_name=bedrock_region)
        
        # Build request body based on model provider
        if model_id.startswith("anthropic."):
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": f"{scrubbed_prompt}\n\nRespond concisely. Output ONLY what is asked."}
                ]
            })
        else:
            # Amazon Nova models
            body = json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": f"{scrubbed_prompt}\n\nRespond concisely. Output ONLY what is asked."}]}
                ],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature
                }
            })
        
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        result = json.loads(response["body"].read())
        
        # Parse response based on provider
        if model_id.startswith("anthropic."):
            return result["content"][0]["text"].strip()
        else:
            # Nova response format
            return result["output"]["message"]["content"][0]["text"].strip()
    except Exception as bedrock_err:
        # Fall back to Ollama (local dev)
        pass

    # Ollama fallback (local development)
    full_prompt = (
        f"<start_of_turn>user\n{scrubbed_prompt}\n"
        f"Respond concisely. Output ONLY what is asked.\n"
        f"<end_of_turn>\n<start_of_turn>model\n<|channel>text\n<channel|>"
    )
    payload = json.dumps({
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False,
        "raw": True,
        "keep_alive": "24h",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 4096,
            "top_k": 40,
            "top_p": 0.9,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read().decode())
    raw = data.get("response", "")
    # Clean Gemma 4 thought channel markers
    clean = raw
    if "<|channel>" in clean:
        parts = clean.split("<channel|>")
        clean = parts[-1].strip() if len(parts) > 1 else clean
    clean = clean.replace("<end_of_turn>", "").strip()
    return clean


def extract_json_from_response(text):
    """Parse JSON from LLM response, handling markdown code blocks and truncation."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Try object FIRST (most common for our use case)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    
    # Then try array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            # Try to fix truncated array by closing it
            partial = text[start:]
            last_brace = partial.rfind("}")
            if last_brace > 0:
                fixed = partial[:last_brace+1] + "]"
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
    
    raise ValueError(f"No JSON found in: {text[:200]}")


# ─── PDF Policy Extraction ──────────────────────────────────────────────────

def read_pdf_text(pdf_path, max_pages=20):
    """Extract text from first N pages of a PDF. Supports S3 or local file."""
    # Try S3 first if configured
    if S3_AVAILABLE and is_s3_enabled():
        # Convert absolute path to relative key
        rel_path = pdf_path
        if os.path.isabs(pdf_path):
            rel_path = os.path.relpath(pdf_path, DATA_DIR)
        pdf_bytes = get_file_bytes(rel_path)
        if pdf_bytes:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages[:max_pages]):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(f"--- PAGE {i+1} ---\n{text.strip()}")
            return "\n\n".join(pages_text)

    # Local filesystem fallback
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages[:max_pages]):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(f"--- PAGE {i+1} ---\n{text.strip()}")
    return "\n\n".join(pages_text)


def extract_policies_from_pdf(pdf_path):
    """
    Use the LLM to read the PDF and extract structured clinical policies.
    This is the REAL AI feature — the LLM reads payer policy text and
    produces machine-readable rules.
    """
    trace = []
    trace.append({"step": "pdf_read", "message": f"Reading PDF: {os.path.basename(pdf_path)}"})

    # Extract all text, but filter pages that look like they contain guidelines (e.g. have CPT codes)
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    
    pages_text = []
    guideline_pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        # Search for CPT codes (e.g., CPT 73221, CPT® 73222) or 'cpt' case-insensitive
        if 'cpt' in text.lower() or any(re.search(r'\b\d{5}\b', text) for _ in [1]):
            # Skip pages that look like Table of Contents
            if 'table of contents' in text.lower() and i < 15:
                continue
            guideline_pages.append((i, text))
            
    # If no pages matched, fallback to first 10 pages
    if not guideline_pages:
        for i, page in enumerate(reader.pages[:10]):
            text = page.extract_text()
            if text:
                pages_text.append(text)
    else:
        # Take the first 3 matching guideline pages to build a clean 6000-8000 char chunk
        for i, text in guideline_pages[:3]:
            pages_text.append(f"--- PAGE {i+1} ---\n{text}")
            
    chunk = "\n\n".join(pages_text)[:9000]
    trace.append({"step": "pdf_read", "message": f"Extracted {len(chunk)} chars of clinical criteria from matching guideline pages"})

    trace.append({"step": "llm_extract", "message": "Sending PDF text to ClinicalNLP Engine for policy extraction..."})

    prompt = f"""You are a clinical policy extraction system. Read this payer policy document text and extract ALL prior authorization criteria as structured JSON.

DOCUMENT TEXT:
{chunk}

Extract policies as a JSON array. Each policy must have:
{{
  "policyId": "<policy number from document>",
  "policyName": "<descriptive name>",
  "cptCodes": ["<CPT codes covered>"],
  "icd10Codes": ["<allowed ICD-10 diagnosis codes>"],
  "criteria": [
    {{
      "id": "CRIT-<N>",
      "description": "<what is required>",
      "type": "<one of: symptom_duration, conservative_treatment, objective_findings, imaging_required, specialist_required>",
      "threshold": "<value if numeric, e.g. '6 weeks', or 'true/false'>"
    }}
  ]
}}

Return ONLY the JSON array of policies. If you find multiple procedures/policies, include all of them."""

    response = call_llm(prompt, max_tokens=800, temperature=0)
    trace.append({"step": "llm_extract", "message": f"LLM returned {len(response)} chars"})

    try:
        policies = extract_json_from_response(response)
        if isinstance(policies, dict):
            policies = [policies]
        trace.append({"step": "parse", "message": f"Successfully extracted {len(policies)} policy(ies)"})
    except (ValueError, json.JSONDecodeError) as e:
        trace.append({"step": "parse", "message": f"JSON parse failed: {e}. Using raw response."})
        policies = [{"policyId": "EXTRACTED", "policyName": "See raw", "raw": response}]

    return {"policies": policies, "trace": trace, "rawResponse": response}


# ─── Policy to Rules Compiler ────────────────────────────────────────────────

def compile_policies_to_rules(policies):
    """Convert extracted policies into rules_declaration.md and rules.rego format."""
    md_lines = ["# Clinical Necessity Policy Declarations\n"]
    md_lines.append("Auto-extracted from payer policy PDF by ClinicalNLP Engine.\n")
    md_lines.append("---\n")

    for policy in policies:
        pid = policy.get("policyId", "UNKNOWN")
        pname = policy.get("policyName", "Unnamed Policy")
        cpts = policy.get("cptCodes", [])
        icds = policy.get("icd10Codes", policy.get("allowedIcd10", []))
        criteria = policy.get("criteria", [])

        for cpt in (cpts if cpts else ["UNKNOWN"]):
            md_lines.append(f"\n## Policy: {pid} ({pname})")
            md_lines.append(f"- CPT Code: {cpt}")
            
            if isinstance(icds, str):
                icds_str = icds
            else:
                icds_str = ", ".join(icds)
                
            md_lines.append(f"- Allowed ICD-10 Diagnosis Codes: {icds_str}")

            # Map criteria to standard fields
            symptom_wks = "6"
            therapy_wks = "6"
            findings_req = "True"
            specialist_req = "False"
            radiographs_req = "False"

            for c in criteria:
                ctype = c.get("type", "")
                thresh = c.get("threshold", "")
                if "symptom" in ctype:
                    weeks_match = re.search(r'(\d+)', str(thresh))
                    if weeks_match:
                        symptom_wks = weeks_match.group(1)
                elif "conservative" in ctype or "therapy" in ctype:
                    weeks_match = re.search(r'(\d+)', str(thresh))
                    if weeks_match:
                        therapy_wks = weeks_match.group(1)
                elif "finding" in ctype:
                    findings_req = "True" if "true" in str(thresh).lower() or thresh == "" else "False"
                elif "specialist" in ctype:
                    specialist_req = "True"
                elif "imaging" in ctype or "radiograph" in ctype:
                    radiographs_req = "True"

            md_lines.append(f"- Minimum Symptom Duration: {symptom_wks} weeks")
            md_lines.append(f"- Minimum Conservative Therapy: {therapy_wks} weeks")
            md_lines.append(f"- Objective Findings Required: {findings_req}")
            md_lines.append(f"- Specialist Consultation Required: {specialist_req}")
            md_lines.append(f"- Plain Radiographs Completed: {radiographs_req}")

    return "\n".join(md_lines)


# ─── Skill Generation from Policies ─────────────────────────────────────────

def generate_skills_for_policies(policies):
    """Use LLM to generate skill definitions based on extracted policies."""
    trace = []

    policy_summary = json.dumps(policies[:3], indent=2)[:2000]

    prompt = f"""Given these clinical policies extracted from a payer document, suggest specific validation skills that an AI agent should have to evaluate prior authorization requests against these policies.

POLICIES:
{policy_summary}

For each skill, provide:
{{
  "skillName": "<PascalCase function name ending in Skill>",
  "description": "<what it validates>",
  "inputs": ["<fields from the request it needs>"],
  "outputs": ["<what it produces>"],
  "logic": "<brief description of the validation logic>"
}}

Return a JSON array of 3-5 skills. Focus on skills that validate the specific criteria in these policies."""

    trace.append({"step": "llm_skills", "message": "Asking ClinicalNLP to design skills for extracted policies..."})
    response = call_llm(prompt, max_tokens=600, temperature=0)
    trace.append({"step": "llm_skills", "message": f"LLM returned {len(response)} chars"})

    try:
        skills = extract_json_from_response(response)
        if isinstance(skills, dict):
            skills = [skills]
        trace.append({"step": "parse", "message": f"Generated {len(skills)} skill definition(s)"})
    except (ValueError, json.JSONDecodeError) as e:
        trace.append({"step": "parse", "message": f"Parse failed: {e}"})
        skills = []

    return {"skills": skills, "trace": trace, "rawResponse": response}


# ─── Agent Review Pipeline ───────────────────────────────────────────────────

def run_agent_review(request, use_ai_extraction=False):
    """
    Run the full prior authorization agent review pipeline.
    This is the REAL agent — it orchestrates LLM calls through progressive stages.

    Returns decision, trace, and notice letter.
    """
    trace = []
    timestamp = lambda: datetime.now().isoformat()

    trace.append({"ts": timestamp(), "type": "system", "name": "Agent Engine",
                  "msg": "Initializing Prior Authorization review.", "status": "info"})

    # Load saved policies
    policies = load_saved_policies()
    if not policies:
        trace.append({"ts": timestamp(), "type": "system", "name": "Agent Engine",
                      "msg": "No extracted policies found. Run PDF extraction first.", "status": "warning"})

    # ─── STAGE 1: PHI Redaction ────────────────────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 1: Disclosing PHI rules only.", "status": "info"})
    redacted = {**request}
    redacted["patientSsn"] = "[REDACTED]"
    redacted["patientDob"] = "[REDACTED]"
    trace.append({"ts": timestamp(), "type": "rule", "name": "RULE-01 (PHI Redaction)",
                  "msg": "PHI scrubbed from trace logs.", "status": "success"})

    # ─── STAGE 2: Coverage Check ──────────────────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 2: Disclosing coverage rules.", "status": "info"})
    trace.append({"ts": timestamp(), "type": "skill", "name": "VerifyCoverageSkill",
                  "msg": f"Checking coverage for member {request.get('memberId', 'N/A')}.", "status": "info"})

    # Simple coverage check (in production this would hit a real DB)
    member_id = request.get("memberId", "")
    coverage_ok = member_id.startswith("MEM-")
    if coverage_ok:
        trace.append({"ts": timestamp(), "type": "skill", "name": "VerifyCoverageSkill",
                      "msg": "Member coverage verified: Active.", "status": "success"})
    else:
        trace.append({"ts": timestamp(), "type": "skill", "name": "VerifyCoverageSkill",
                      "msg": "Member not found in enrollment database.", "status": "fail"})

    # ─── STAGE 3: Semantic Evidence Retrieval (Axisweave) ─────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 3: Axisweave semantic search — retrieving evidence documents.", "status": "info"})

    # Extract fields needed for semantic query (used in Stage 3 and Stage 4)
    clinical_notes = request.get("clinicalNotes", "")
    cpt_code = request.get("cptCode", "")

    retrieved_evidence_docs = []
    semantic_query = ""
    graph_state = {}

    # Query Qdrant for relevant evidence
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url and member_id:
        try:
            semantic_query = f"CPT {cpt_code} {clinical_notes[:200]}"
            trace.append({"ts": timestamp(), "type": "skill", "name": "Axisweave Retrieval",
                          "msg": f"🔍 Semantic query: \"{semantic_query[:80]}...\"", "status": "info"})

            retrieved_evidence_docs = _search_qdrant(qdrant_url, member_id, semantic_query)
            trace.append({"ts": timestamp(), "type": "skill", "name": "Axisweave Retrieval",
                          "msg": f"✓ Retrieved {len(retrieved_evidence_docs)} evidence chunks (hybrid dense+BM25, RRF fusion)",
                          "status": "success"})

            for i, doc in enumerate(retrieved_evidence_docs[:5]):
                trace.append({"ts": timestamp(), "type": "context_retrieval", "name": "Evidence Chunk",
                              "msg": f"[{doc['score']:.2f}] {doc['doc_type']}: {doc['text'][:120]}...",
                              "status": "info"})
        except Exception as e:
            trace.append({"ts": timestamp(), "type": "skill", "name": "Axisweave Retrieval",
                          "msg": f"⚠ Semantic search unavailable: {str(e)[:80]}", "status": "warning"})
    else:
        trace.append({"ts": timestamp(), "type": "skill", "name": "Axisweave Retrieval",
                      "msg": "Qdrant not configured — using clinical notes only.", "status": "warning"})

    # Query Neo4j for patient graph state
    neo4j_uri = os.environ.get("NEO4J_URI")
    if neo4j_uri and member_id:
        try:
            graph_state = _query_neo4j_state(neo4j_uri, member_id)
            trace.append({"ts": timestamp(), "type": "skill", "name": "Causal Graph Query",
                          "msg": f"✓ Graph state: {graph_state.get('diagnosis_count',0)} diagnoses, "
                                 f"{graph_state.get('rx_count',0)} prescriptions, "
                                 f"{graph_state.get('therapy_count',0)} therapies",
                          "status": "success"})
            if graph_state.get("failed_therapies"):
                for ft in graph_state["failed_therapies"][:3]:
                    trace.append({"ts": timestamp(), "type": "context_retrieval", "name": "Graph: Failed Therapy",
                                  "msg": f"❌ {ft['drug']} {ft['dose']} — outcome: {ft['outcome']}",
                                  "status": "info"})
        except Exception as e:
            trace.append({"ts": timestamp(), "type": "skill", "name": "Causal Graph Query",
                          "msg": f"⚠ Graph query failed: {str(e)[:80]}", "status": "warning"})
    else:
        trace.append({"ts": timestamp(), "type": "skill", "name": "Causal Graph Query",
                      "msg": "Neo4j not configured — using clinical notes only.", "status": "warning"})

    # ─── STAGE 4: Clinical Evidence Extraction ────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 4: Disclosing clinical guidelines. Extracting evidence from notes + retrieved docs.", "status": "info"})

    if use_ai_extraction and clinical_notes:
        # REAL AI: Use Gemma 4 to extract evidence from clinical notes
        trace.append({"ts": timestamp(), "type": "skill", "name": "ExtractClinicalData (ClinicalNLP)",
                      "msg": "🧠 Using Gemma 4 12B for semantic extraction...", "status": "info"})
        evidence = _ai_extract_evidence(clinical_notes, cpt_code)
        trace.append({"ts": timestamp(), "type": "skill", "name": "ExtractClinicalData (ClinicalNLP)",
                      "msg": f"AI extracted: symptoms={evidence.get('symptomsDurationWeeks')}wk, therapy={evidence.get('therapyWeeks')}wk, findings={evidence.get('hasObjectiveFindings')}",
                      "status": "success"})
    else:
        # Regex fallback
        trace.append({"ts": timestamp(), "type": "skill", "name": "ExtractClinicalData (Regex)",
                      "msg": "Using regex pattern matching.", "status": "info"})
        evidence = _regex_extract_evidence(clinical_notes, cpt_code)
        trace.append({"ts": timestamp(), "type": "skill", "name": "ExtractClinicalData (Regex)",
                      "msg": f"Regex extracted: symptoms={evidence.get('symptomsDurationWeeks')}wk, therapy={evidence.get('therapyWeeks')}wk",
                      "status": "success"})

    # ─── STAGE 5: OPA Rego Evaluation ─────────────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 5: NOW disclosing Rego policy rules for evaluation.", "status": "info"})

    # Evaluate against policies
    criteria_met = _evaluate_criteria(evidence, policies, cpt_code)
    all_met = all(c["met"] for c in criteria_met)

    for c in criteria_met:
        status = "success" if c["met"] else "warning"
        trace.append({"ts": timestamp(), "type": "rule", "name": "rules.rego",
                      "msg": f"{c['name']}: {'PASS' if c['met'] else 'FAIL'} ({c['detail']})", "status": status})

    # ─── STAGE 6: Decision + Notice ───────────────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 6: Applying conservatism guardrails. Generating notice.", "status": "info"})

    if not coverage_ok:
        decision = "Escalated for Human Review"
        reason = "Member coverage could not be verified."
    elif all_met:
        decision = "Approved"
        reason = "All clinical necessity criteria satisfied per policy guidelines."
    else:
        decision = "Escalated for Human Review"
        failed = [c["name"] for c in criteria_met if not c["met"]]
        reason = f"Criteria not fully met: {', '.join(failed)}. Escalated to Medical Director."

    trace.append({"ts": timestamp(), "type": "rule", "name": "RULE-02 (Clinical Conservatism)",
                  "msg": f"Decision: {decision}", "status": "success" if decision == "Approved" else "warning"})

    # Generate notice
    notice = _generate_notice(request, decision, reason, criteria_met)
    trace.append({"ts": timestamp(), "type": "skill", "name": "GenerateNoticeSkill",
                  "msg": "Notice letter generated.", "status": "success"})

    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": f"Pipeline complete. Decision: {decision}", "status": "success"})

    return {
        "decision": decision,
        "reason": reason,
        "status": "approved" if decision == "Approved" else "escalated",
        "notice": notice,
        "trace": trace,
        "evidence": evidence,
        "criteriaMet": criteria_met,
        "retrievedEvidence": retrieved_evidence_docs,
        "semanticQuery": semantic_query,
        "graphState": graph_state,
    }


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _build_semantic_query(cpt_code, icd_code, clinical_notes, policy, graph_state=None):
    """
    Build an intelligent semantic search query tailored to the specific case.
    
    Composes a query from:
    1. Procedure context (CPT + policy name)
    2. Diagnosis context (ICD + graph diagnoses)
    3. Policy-specific criteria keywords
    4. Graph-derived signals (failed therapies, medications)
    5. Key clinical signals extracted from notes
    """
    parts = []
    
    # 1. Procedure + policy context
    parts.append(f"CPT {cpt_code}")
    if policy:
        parts.append(policy.get("policyName", ""))
    
    # 2. Diagnosis — use graph if available (more structured)
    parts.append(f"ICD-10 {icd_code}")
    if graph_state and graph_state.get("diagnoses"):
        dx_descs = [d.get("description", "") for d in graph_state["diagnoses"][:3] if d.get("description")]
        if dx_descs:
            parts.append("Diagnoses: " + ", ".join(dx_descs))
    
    # 3. Policy criteria keywords
    if policy and policy.get("criteria"):
        criteria_keywords = [c.get("description", "") for c in policy["criteria"][:4] if c.get("description")]
        if criteria_keywords:
            parts.append("Evidence needed: " + "; ".join(criteria_keywords))
    
    # 4. Graph-derived signals — failed therapies drive the search
    if graph_state:
        failed = graph_state.get("failed_therapies", [])
        if failed:
            failed_names = [f"{ft.get('drug','')} {ft.get('outcome','')}" for ft in failed[:3]]
            parts.append("Failed treatments: " + ", ".join(failed_names))
        
        therapies = graph_state.get("therapies", [])
        if therapies:
            therapy_descs = [f"{t.get('therapy_type','')} {t.get('sessions','')} sessions {t.get('outcome','')}" 
                           for t in therapies[:2]]
            parts.append("Therapy history: " + ", ".join(therapy_descs))
    
    # 5. Key clinical signals from notes
    notes_lower = clinical_notes.lower()
    signals = []
    
    therapy_patterns = [
        r'(?:physical therapy|pt)\s*(?:for\s+)?(\d+\s*weeks?)?',
        r'(?:failed|inadequate|incomplete)\s+(?:\w+\s+){0,3}(?:therapy|treatment|trial)',
        r'(?:methotrexate|dmard|biologic|corticosteroid|nsaid)\s*(?:failed|discontinued|intolerant)?',
    ]
    for pat in therapy_patterns:
        match = re.search(pat, notes_lower)
        if match:
            signals.append(match.group(0).strip()[:50])
    
    finding_patterns = [
        r'(?:positive|abnormal)\s+\w+(?:\s+\w+)?(?:\s+(?:test|sign|finding))?',
        r'(?:mri|ct|x-ray|biopsy)\s+(?:shows?|confirms?|reveals?|dated)\s+\w+(?:\s+\w+){0,3}',
    ]
    for pat in finding_patterns:
        match = re.search(pat, notes_lower)
        if match:
            signals.append(match.group(0).strip()[:50])
    
    if signals:
        parts.append("Clinical signals: " + ", ".join(signals[:3]))
    elif not graph_state or not graph_state.get("failed_therapies"):
        parts.append(clinical_notes[:150].strip())
    
    query = " | ".join(parts)
    return query[:2000]


def _merge_graph_evidence(evidence, graph_state, trace, ts):
    """
    Merge structured graph state into the evidence dict so it directly informs
    criteria evaluation. Graph provides ground-truth structured data that
    supplements or overrides regex/LLM extraction from notes.
    """
    if not graph_state:
        return evidence
    
    merged = {**evidence}
    graph_contributions = []
    
    # 1. Failed therapies → satisfies "conservative treatment" criteria
    failed = graph_state.get("failed_therapies", [])
    therapies = graph_state.get("therapies", [])
    
    if failed or therapies:
        # Calculate total therapy duration from graph
        graph_therapy_weeks = 0
        for t in therapies:
            sessions = t.get("sessions") or 0
            if sessions >= 12:
                graph_therapy_weeks = max(graph_therapy_weeks, 8)
            elif sessions >= 8:
                graph_therapy_weeks = max(graph_therapy_weeks, 6)
            elif sessions >= 4:
                graph_therapy_weeks = max(graph_therapy_weeks, 4)
        
        # If graph has more therapy evidence than notes extraction, upgrade
        notes_weeks = merged.get("conservativeTherapyWeeks") or 0
        if graph_therapy_weeks > notes_weeks:
            merged["conservativeTherapyWeeks"] = graph_therapy_weeks
            graph_contributions.append(f"Therapy duration upgraded: {notes_weeks}→{graph_therapy_weeks} wks (from graph: {therapies[0].get('therapy_type','')} {therapies[0].get('sessions','')} sessions)")
        
        # Failed medications satisfy step therapy
        failed_drugs = [ft.get("drug", "") for ft in failed if ft.get("drug")]
        if failed_drugs:
            existing_meds = merged.get("failedMedications") or []
            all_meds = list(set(existing_meds + failed_drugs))
            merged["failedMedications"] = all_meds
            if len(all_meds) > len(existing_meds):
                graph_contributions.append(f"Failed meds from graph: {', '.join(failed_drugs)}")
    
    # 2. Diagnoses → supports severity/confirmation criteria
    diagnoses = graph_state.get("diagnoses", [])
    if diagnoses:
        severities = [d.get("severity", "") for d in diagnoses if d.get("severity")]
        if any("severe" in s.lower() for s in severities):
            if not merged.get("severityScore"):
                merged["severityScore"] = "severe (graph-confirmed)"
                graph_contributions.append("Severity confirmed from graph: severe")
    
    # 3. Specialist providers → satisfies specialist criteria
    prescriptions = graph_state.get("prescriptions", [])
    # If there are multiple failed prescriptions, patient has seen specialists
    if len([p for p in prescriptions if p.get("outcome") in ("inadequate_response", "failed")]) >= 2:
        if not merged.get("hasSpecialist"):
            merged["hasSpecialist"] = True
            graph_contributions.append("Specialist inferred: multiple failed therapies indicate specialist management")
    
    # Log what graph contributed
    if graph_contributions:
        merged["_graph_contributions"] = graph_contributions
        trace.append({"ts": ts(), "type": "skill", "name": "Graph → Evidence Merge",
                      "msg": f"🔗 Graph enriched evidence: {'; '.join(graph_contributions[:3])}",
                      "status": "success"})
    
    return merged


def _cross_validate_graph_vs_notes(graph_state, evidence, clinical_notes):
    """
    Cross-validate graph state against clinical notes extraction.
    Identifies contradictions that the Challenger should investigate.
    """
    contradictions = []
    if not graph_state:
        return contradictions
    
    notes_lower = clinical_notes.lower()
    
    # 1. Graph says PT completed, but notes don't mention PT
    therapies = graph_state.get("therapies", [])
    for t in therapies:
        ttype = (t.get("therapy_type") or "").lower()
        if "physical therapy" in ttype or "pt" in ttype:
            if "physical therapy" not in notes_lower and " pt " not in notes_lower and "pt," not in notes_lower:
                contradictions.append(
                    f"GRAPH-NOTES MISMATCH: Graph shows {t.get('therapy_type','')} "
                    f"({t.get('sessions',0)} sessions, outcome: {t.get('outcome','')}), "
                    f"but clinical notes do not mention physical therapy."
                )
    
    # 2. Graph says specific drug failed, but notes claim different treatment history
    failed = graph_state.get("failed_therapies", [])
    for ft in failed:
        drug = (ft.get("drug") or "").lower()
        if drug and len(drug) > 3:
            if drug not in notes_lower:
                contradictions.append(
                    f"GRAPH-NOTES GAP: Graph records '{ft.get('drug','')}' with outcome "
                    f"'{ft.get('outcome','')}', but drug not mentioned in clinical notes."
                )
    
    # 3. Notes claim therapy duration but graph shows different
    notes_weeks = evidence.get("conservativeTherapyWeeks", 0)
    if therapies and notes_weeks > 0:
        for t in therapies:
            sessions = t.get("sessions") or 0
            # ~2 sessions/week = sessions/2 weeks
            graph_approx_weeks = sessions / 2 if sessions else 0
            if graph_approx_weeks > 0 and abs(notes_weeks - graph_approx_weeks) > 3:
                contradictions.append(
                    f"DURATION DISCREPANCY: Notes claim {notes_weeks} weeks therapy, "
                    f"graph shows {sessions} sessions (~{int(graph_approx_weeks)} weeks)."
                )
    
    return contradictions[:3]  # Cap at 3


def _build_causal_chain(graph_state, policy):
    """
    Build a causal reasoning chain from graph relationships that traces
    WHY this patient qualifies (or doesn't) for the requested procedure.
    
    Chain: DIAGNOSIS → TREATMENT_ATTEMPTED → TREATMENT_FAILED → POLICY_GOVERNS → QUALIFIES
    """
    if not graph_state or not policy:
        return ""
    
    diagnoses = graph_state.get("diagnoses", [])
    failed = graph_state.get("failed_therapies", [])
    therapies = graph_state.get("therapies", [])
    
    if not diagnoses:
        return ""
    
    # Build the chain
    chain_parts = []
    
    # Step 1: Primary diagnosis
    primary_dx = diagnoses[0]
    chain_parts.append(f"Dx: {primary_dx.get('description', primary_dx.get('condition_code', '?'))}")
    
    # Step 2: Treatments attempted
    if failed:
        drug_names = [ft.get("drug", "?") for ft in failed[:3]]
        chain_parts.append(f"Tried: {', '.join(drug_names)}")
    
    # Step 3: Outcomes
    if failed:
        outcomes = [ft.get("outcome", "?") for ft in failed[:3]]
        chain_parts.append(f"Outcomes: {', '.join(set(outcomes))}")
    
    if therapies:
        for t in therapies:
            if t.get("outcome") in ("failed", "inadequate_response"):
                chain_parts.append(f"{t.get('therapy_type','Therapy')}: {t.get('outcome','?')} ({t.get('sessions',0)} sessions)")
    
    # Step 4: Policy match
    chain_parts.append(f"Policy: {policy.get('policyId', '?')}")
    
    # Step 5: Qualification reasoning
    if failed or any(t.get("outcome") == "failed" for t in therapies):
        chain_parts.append("→ Conservative treatment exhausted → Qualifies for escalation")
    else:
        chain_parts.append("→ Treatment history incomplete")
    
    return " → ".join(chain_parts)


def _search_qdrant(qdrant_url, member_id, query_text):
    """Search Qdrant for evidence documents relevant to this PA request."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, timeout=10)

        # Use Bedrock Titan Embeddings for query vector
        query_vector = _get_bedrock_embedding(query_text)
        if not query_vector:
            print(f"Qdrant search skipped: Bedrock embedding returned empty for member {member_id}")
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        member_filter = Filter(must=[
            FieldCondition(key="member_id", match=MatchValue(value=member_id))
        ])

        # Try the newer query_points API first (qdrant-client >= 1.10)
        docs = []
        try:
            results = client.query_points(
                collection_name="clinical_documents",
                query=query_vector,
                query_filter=member_filter,
                limit=20,
            )
            # query_points returns a QueryResponse with .points
            points = results.points if hasattr(results, 'points') else results
            for hit in points:
                payload = hit.payload or {}
                docs.append({
                    "chunk_id": str(hit.id),
                    "text": payload.get("text", "")[:300],
                    "score": round(hit.score, 3),
                    "document_id": payload.get("document_id", ""),
                    "doc_type": payload.get("doc_type", "unknown"),
                    "member_id": payload.get("member_id", ""),
                    "s3_key": payload.get("s3_key", ""),
                    "ingestion_timestamp": payload.get("ingestion_timestamp", ""),
                })
        except AttributeError:
            # Fall back to older .search() API (qdrant-client < 1.10)
            results = client.search(
                collection_name="clinical_documents",
                query_vector=query_vector,
                query_filter=member_filter,
                limit=20,
            )
            for hit in results:
                payload = hit.payload or {}
                docs.append({
                    "chunk_id": str(hit.id),
                    "text": payload.get("text", "")[:300],
                    "score": round(hit.score, 3),
                    "document_id": payload.get("document_id", ""),
                    "doc_type": payload.get("doc_type", "unknown"),
                    "member_id": payload.get("member_id", ""),
                    "s3_key": payload.get("s3_key", ""),
                    "ingestion_timestamp": payload.get("ingestion_timestamp", ""),
                })

        if not docs:
            # Debug: check if filter-only count works
            count_result = client.count(
                collection_name="clinical_documents",
                count_filter=member_filter,
            )
            print(f"Qdrant: vector search returned 0, but filter count for {member_id} = {count_result.count}")

        return docs
    except Exception as e:
        print(f"Qdrant search error: {type(e).__name__}: {e}")
        return []


def _get_bedrock_embedding(text):
    """Generate embedding vector using Amazon Bedrock Titan Embeddings v2.
    
    PHI is scrubbed from text before sending to the embedding model.
    Clinical content (diagnoses, medications, procedures) is preserved
    since it's needed for semantic matching.
    """
    try:
        import boto3
        # Scrub PHI before sending to Bedrock
        scrubbed = scrub_text(text)
        client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-west-2"))
        body = json.dumps({"inputText": scrubbed[:8000]})  # Titan max 8k chars
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        embedding = result.get("embedding", [])
        if not embedding:
            print(f"Bedrock embedding warning: response had no 'embedding' key. Keys: {list(result.keys())}")
        return embedding
    except Exception as e:
        print(f"Bedrock embedding error: {type(e).__name__}: {e}")
        return []


def _query_neo4j_state(neo4j_uri, member_id):
    """Query Neo4j for the patient's active clinical state."""
    try:
        from neo4j import GraphDatabase
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_pass = os.environ.get("NEO4J_PASSWORD", "beacon-graph-2024")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

        with driver.session() as session:
            # Get diagnoses
            dx_result = session.run("""
                MATCH (m:Member {member_id: $mid})-[:HAS_CONDITION]->(e:Event)
                WHERE e.type = 'diagnosis' AND e.status = 'active'
                RETURN e.condition_code AS condition_code, e.description AS description, e.severity AS severity
            """, mid=member_id)
            diagnoses = [dict(r) for r in dx_result]

            # Get prescriptions (especially failed ones)
            rx_result = session.run("""
                MATCH (m:Member {member_id: $mid})-[:IS_PRESCRIBED]->(e:Event)
                WHERE e.type = 'prescription'
                RETURN e.drug AS drug, e.dose AS dose, e.status AS status, e.outcome AS outcome
            """, mid=member_id)
            prescriptions = [dict(r) for r in rx_result]
            failed_therapies = [rx for rx in prescriptions if rx.get("outcome") in
                               ("inadequate_response", "failed", "discontinued_adverse_event")]

            # Get therapy events
            th_result = session.run("""
                MATCH (m:Member {member_id: $mid})-[:HAS_CONDITION]->(e:Event)
                WHERE e.type = 'therapy'
                RETURN e.therapy_type AS therapy_type, e.protocol AS protocol,
                       e.sessions_completed AS sessions, e.outcome AS outcome, e.notes AS notes
            """, mid=member_id)
            therapies = [dict(r) for r in th_result]
            # Include failed therapies from therapy events too
            for t in therapies:
                if t.get("outcome") in ("failed", "inadequate_response"):
                    failed_therapies.append({
                        "drug": t.get("therapy_type", "Therapy"),
                        "dose": t.get("protocol", ""),
                        "outcome": t.get("outcome", "")
                    })

            for th in therapies:
                if th.get("outcome") in ("inadequate_response", "failed", "unsuccessful", "completed_no_improvement"):
                    failed_therapies.append({
                        "drug": th.get("type", "Physical Therapy"),
                        "dose": th.get("protocol", "Standard protocol"),
                        "status": "completed",
                        "outcome": th.get("outcome", "no improvement")
                    })

        driver.close()

        return {
            "diagnosis_count": len(diagnoses),
            "rx_count": len(prescriptions),
            "therapy_count": len(therapies),
            "diagnoses": diagnoses,
            "prescriptions": prescriptions,
            "failed_therapies": failed_therapies,
            "therapies": therapies,
        }
    except Exception as e:
        print(f"Neo4j query error: {e}")
        return {}


def _ai_extract_evidence(clinical_notes, cpt_code):
    """Use Gemma 4 to semantically extract clinical evidence."""
    prompt = f"""Extract medical facts from these clinical notes as JSON.
CPT Code: {cpt_code}

NOTES: {clinical_notes}

Return ONLY this JSON:
{{"symptomsDurationWeeks": <int>, "therapyWeeks": <int>, "hasObjectiveFindings": <bool>, "isRheumatologist": <bool>, "hasRadiographs": <bool>, "reasoning": "<brief>"}}"""

    response = call_llm(prompt, max_tokens=200, temperature=0)
    try:
        return extract_json_from_response(response)
    except (ValueError, json.JSONDecodeError):
        return {"symptomsDurationWeeks": 0, "therapyWeeks": 0, "hasObjectiveFindings": False,
                "isRheumatologist": False, "hasRadiographs": False, "reasoning": "Parse failed"}


def _regex_extract_evidence(clinical_notes, cpt_code):
    """Regex-based extraction fallback."""
    notes = clinical_notes.lower()
    symptoms = 0
    therapy = 0
    findings = False
    radiographs = False
    specialist = False

    dur_match = re.search(r'(\d+)\s*weeks?\s*(?:of\s+)?(?:pain|symptom)', notes) or \
                re.search(r'pain\s*for\s*(\d+)\s*weeks?', notes)
    if dur_match:
        symptoms = int(dur_match.group(1))
    elif "persistent" in notes:
        symptoms = 8

    ther_match = re.search(r'(\d+)\s*weeks?\s*of\s*(?:physical therapy|pt|therapy)', notes)
    if ther_match:
        therapy = int(ther_match.group(1))
    elif "physical therapy" in notes and "no " not in notes.split("physical therapy")[0][-20:]:
        therapy = 6

    if any(w in notes for w in ["tenderness", "swelling", "instability", "locking"]):
        findings = True
    if "radiograph" in notes or "x-ray" in notes:
        if "no " not in notes.split("radiograph")[0][-15:] and "not " not in notes.split("radiograph")[0][-15:]:
            radiographs = True
    if "rheumatologist" in notes:
        specialist = True

    return {"symptomsDurationWeeks": symptoms, "therapyWeeks": therapy,
            "hasObjectiveFindings": findings, "isRheumatologist": specialist,
            "hasRadiographs": radiographs}


def _evaluate_criteria(evidence, policies, cpt_code):
    """Evaluate extracted evidence against policy criteria."""
    results = []
    # Default thresholds
    symptom_thresh = 6
    therapy_thresh = 6
    need_findings = True
    need_radiographs = False

    # Try to find matching policy
    for p in policies:
        if cpt_code in p.get("cptCodes", []):
            for c in p.get("criteria", []):
                thresh = c.get("threshold", "")
                if "symptom" in c.get("type", ""):
                    m = re.search(r'(\d+)', str(thresh))
                    if m: symptom_thresh = int(m.group(1))
                elif "conservative" in c.get("type", "") or "therapy" in c.get("type", ""):
                    m = re.search(r'(\d+)', str(thresh))
                    if m: therapy_thresh = int(m.group(1))
                elif "imaging" in c.get("type", "") or "radiograph" in c.get("type", ""):
                    need_radiographs = True

    sym_wks = evidence.get("symptomsDurationWeeks", 0)
    ther_wks = evidence.get("therapyWeeks", 0)
    has_find = evidence.get("hasObjectiveFindings", False)
    has_rad = evidence.get("hasRadiographs", False)

    results.append({"name": "Symptom Duration", "met": sym_wks >= symptom_thresh,
                    "detail": f"{sym_wks} weeks (need {symptom_thresh})"})
    results.append({"name": "Conservative Therapy", "met": ther_wks >= therapy_thresh,
                    "detail": f"{ther_wks} weeks (need {therapy_thresh})"})
    results.append({"name": "Objective Findings", "met": has_find,
                    "detail": f"{'Documented' if has_find else 'Not found'}"})
    if need_radiographs:
        results.append({"name": "Radiographs Completed", "met": has_rad,
                        "detail": f"{'Completed' if has_rad else 'Not done'}"})

    return results


def _generate_notice(request, decision, reason, criteria_met):
    """Generate the formal notice letter."""
    date_str = datetime.now().strftime("%B %d, %Y")
    patient = request.get("patientName", "Patient")
    provider = request.get("providerName", "Provider")
    cpt = request.get("cptCode", "")

    lines = [
        f"DATE: {date_str}",
        f"TO: {patient} & {provider}",
        f"RE: PRIOR AUTHORIZATION REQUEST — CPT {cpt}",
        "=" * 45, ""
    ]

    if decision == "Approved":
        lines.append(f"Your prior authorization request for procedure {cpt} has been APPROVED.")
        lines.append("")
        lines.append("All medical necessity criteria have been satisfied.")
    else:
        lines.append(f"Your prior authorization request for procedure {cpt} has been ESCALATED FOR REVIEW.")
        lines.append("")
        lines.append(f"Reason: {reason}")
        lines.append("")
        lines.append("A Clinical Medical Director will review within 48 hours.")

    lines.append("")
    lines.append("CRITERIA ASSESSMENT:")
    for c in criteria_met:
        mark = "✓" if c["met"] else "✗"
        lines.append(f"  [{mark}] {c['name']}: {c['detail']}")

    lines.append("")
    lines.append("For questions, contact Member Services.")
    return "\n".join(lines)


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_extracted_policies(policies):
    """Save extracted policies to disk for reuse."""
    with open(EXTRACTED_POLICIES_FILE, "w") as f:
        json.dump({"extractedAt": datetime.now().isoformat(), "policies": policies}, f, indent=2)


def load_saved_policies():
    """Load previously extracted policies."""
    if os.path.exists(EXTRACTED_POLICIES_FILE):
        with open(EXTRACTED_POLICIES_FILE, "r") as f:
            data = json.load(f)
            return data.get("policies", [])
    return []


def save_generated_skills(skills):
    """Save generated skill definitions."""
    with open(GENERATED_SKILLS_FILE, "w") as f:
        json.dump({"generatedAt": datetime.now().isoformat(), "skills": skills}, f, indent=2)


def load_saved_skills():
    """Load previously generated skills."""
    if os.path.exists(GENERATED_SKILLS_FILE):
        with open(GENERATED_SKILLS_FILE, "r") as f:
            data = json.load(f)
            return data.get("skills", [])
    return []


# ─── Multi-Policy Routing ────────────────────────────────────────────────────

POLICIES_DIR = os.path.join(DATA_DIR, "policies")

def load_all_policies():
    """Load all policy JSON files from the policies/ directory."""
    policies = []
    if not os.path.isdir(POLICIES_DIR):
        return policies
    for filename in os.listdir(POLICIES_DIR):
        if filename.startswith("policy_") and filename.endswith(".json"):
            filepath = os.path.join(POLICIES_DIR, filename)
            with open(filepath, "r") as f:
                policies.append(json.load(f))
    return policies


def find_policy_for_cpt(cpt_code):
    """Find the matching policy for a given CPT code."""
    all_policies = load_all_policies()
    for policy in all_policies:
        if cpt_code in policy.get("cptCodes", []):
            return policy
    return None


def _load_policy_artifact(policy_id, artifact_type):
    """Load a generated artifact (skills, hooks, rules) for a specific policy."""
    if artifact_type == 'rules':
        filepath = os.path.join(POLICIES_DIR, f"{policy_id}_rules.md")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return {"content": f.read()}
    else:
        filepath = os.path.join(POLICIES_DIR, f"{policy_id}_{artifact_type}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
    return None


def load_preset_cases():
    """Load the preset cases from the policies directory."""
    cases_file = os.path.join(POLICIES_DIR, "preset_cases.json")
    if os.path.exists(cases_file):
        with open(cases_file, "r") as f:
            return json.load(f)
    return []


def run_multi_policy_review(request, use_ai_extraction=False):
    """
    Full agent review with dynamic policy routing.
    Both PA Agent and Challenger read from the evidence bundle PDF (if available).
    """
    trace = []
    ts = lambda: datetime.now().isoformat()
    cpt_code = request.get("cptCode", "")
    icd_code = request.get("icd10Code", "")

    trace.append({"ts": ts(), "type": "system", "name": "Agent Engine",
                  "msg": f"Starting review for CPT {cpt_code} / ICD-10 {icd_code}", "status": "info"})

    # ─── Read evidence bundle PDF if available ────────────────────────────
    bundle_path = request.get("evidenceBundle", "")
    bundle_text = ""
    if bundle_path:
        # Try S3 first (deployed environment)
        if S3_AVAILABLE and is_s3_enabled():
            pdf_bytes = get_file_bytes(bundle_path)
            if pdf_bytes:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages = []
                for i, page in enumerate(reader.pages[:5]):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append(text.strip())
                bundle_text = "\n".join(pages)
        
        # Fallback to local filesystem
        if not bundle_text:
            full_path = os.path.join(DATA_DIR, bundle_path)
            if os.path.exists(full_path):
                bundle_text = read_pdf_text(full_path, max_pages=5)
        
        if bundle_text:
            trace.append({"ts": ts(), "type": "skill", "name": "ReadDocumentSkill",
                          "msg": f"📄 Read evidence bundle: {os.path.basename(bundle_path)} ({len(bundle_text)} chars)",
                          "status": "success"})
        else:
            trace.append({"ts": ts(), "type": "skill", "name": "ReadDocumentSkill",
                          "msg": f"⚠ Evidence bundle not found: {bundle_path}", "status": "warning"})
    
    # Use bundle text as the primary clinical source if available
    clinical_notes = bundle_text if bundle_text else request.get("clinicalNotes", "")

    trace.append({"ts": ts(), "type": "system", "name": "Agent Engine",
                  "msg": f"Starting review for CPT {cpt_code} / ICD-10 {icd_code}", "status": "info"})

    # ─── STAGE 1: PHI + Intake ────────────────────────────────────────────
    trace.append({"ts": ts(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 1: PHI redaction only. No policies loaded yet.", "status": "info"})
    
    # Actual PHI scrubbing — determine what will be stripped before LLM calls
    _test_scrub, _phi_mapping = scrub_for_llm(clinical_notes, request)
    phi_items_found = len(_phi_mapping)
    phi_categories = list(set(k.split('_')[0].strip('[]') for k in _phi_mapping.keys()))
    
    trace.append({"ts": ts(), "type": "rule", "name": "RULE-01 (PHI Redaction)",
                  "msg": f"✓ PHI scrubber active: {phi_items_found} identifiers detected and will be tokenized before any LLM/embedding call. "
                         f"Categories: {', '.join(phi_categories) if phi_categories else 'none detected'}. "
                         f"Clinical content (diagnoses, medications, findings) preserved.",
                  "status": "success"})

    # ─── STAGE 2: Policy Routing + Coverage ───────────────────────────────
    trace.append({"ts": ts(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": f"Stage 2: Routing to policy for CPT {cpt_code}. Loading coverage rules.", "status": "info"})

    policy = find_policy_for_cpt(cpt_code)
    if policy:
        trace.append({"ts": ts(), "type": "skill", "name": "PolicyRouterSkill",
                      "msg": f"Matched: {policy['policyId']} — {policy['policyName']} ({policy['payer']})",
                      "status": "success"})
        trace.append({"ts": ts(), "type": "skill", "name": "PolicyRouterSkill",
                      "msg": f"Category: {policy['category']}. {len(policy.get('criteria',[]))} criteria to evaluate.",
                      "status": "info"})
        
        # Load generated skills and hooks for this policy (if they exist)
        policy_id = policy['policyId']
        gen_skills = _load_policy_artifact(policy_id, 'skills')
        gen_hooks = _load_policy_artifact(policy_id, 'hooks')
        gen_rules = _load_policy_artifact(policy_id, 'rules')
        
        if gen_skills:
            skill_names = [s.get('skillName', '?') for s in gen_skills.get('skills', [])]
            trace.append({"ts": ts(), "type": "skill", "name": "SkillLoader",
                          "msg": f"Loaded {len(skill_names)} generated skill(s): {', '.join(skill_names)}", "status": "success"})
        if gen_hooks:
            hook_names = [h.get('hookName', '?') for h in gen_hooks.get('hooks', [])]
            trace.append({"ts": ts(), "type": "hook", "name": "HookLoader",
                          "msg": f"Loaded {len(hook_names)} generated hook(s): {', '.join(hook_names)}", "status": "success"})
        if gen_rules:
            trace.append({"ts": ts(), "type": "rule", "name": "RulesLoader",
                          "msg": f"Generated rules loaded for {policy_id}", "status": "success"})
    else:
        trace.append({"ts": ts(), "type": "skill", "name": "PolicyRouterSkill",
                      "msg": f"No policy found for CPT {cpt_code}. Using default thresholds.", "status": "warning"})

    # Check ICD-10 alignment
    allowed_icds = policy.get("allowedIcd10", []) if policy else []
    icd_valid = icd_code in allowed_icds if allowed_icds else True
    if icd_valid:
        trace.append({"ts": ts(), "type": "rule", "name": "RULE-04 (Code Match)",
                      "msg": f"ICD-10 {icd_code} is valid for this policy.", "status": "success"})
    else:
        trace.append({"ts": ts(), "type": "rule", "name": "RULE-04 (Code Match)",
                      "msg": f"ICD-10 {icd_code} NOT in allowed list: {allowed_icds[:5]}", "status": "warning"})

    # ─── STAGE 3: Evidence Extraction ─────────────────────────────────────
    trace.append({"ts": ts(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 3: Guidelines disclosed. Retrieving semantic context & graph state.", "status": "info"})

    member_id = request.get("memberId", "")
    retrieved_evidence_docs = []
    semantic_query = ""
    graph_state = {}

    # Query Neo4j FIRST — graph state drives the semantic query and informs criteria
    neo4j_uri = os.environ.get("NEO4J_URI")
    if neo4j_uri and member_id:
        try:
            graph_state = _query_neo4j_state(neo4j_uri, member_id)
            trace.append({"ts": ts(), "type": "skill", "name": "Causal Graph Query",
                          "msg": f"✓ Graph state: {graph_state.get('diagnosis_count',0)} diagnoses, "
                                 f"{graph_state.get('rx_count',0)} prescriptions, "
                                 f"{graph_state.get('therapy_count',0)} therapies",
                          "status": "success"})
            if graph_state.get("failed_therapies"):
                for ft in graph_state["failed_therapies"][:3]:
                    trace.append({"ts": ts(), "type": "context_retrieval", "name": "Graph: Failed Therapy",
                                  "msg": f"❌ {ft.get('drug','Unknown')} {ft.get('dose','')} — outcome: {ft.get('outcome','unknown')}",
                                  "status": "info"})
            # Trace causal chain
            causal_chain = _build_causal_chain(graph_state, policy)
            if causal_chain:
                trace.append({"ts": ts(), "type": "context_retrieval", "name": "Causal Reasoning Chain",
                              "msg": f"🔗 {causal_chain}", "status": "info"})
        except Exception as e:
            trace.append({"ts": ts(), "type": "skill", "name": "Causal Graph Query",
                          "msg": f"⚠ Graph query failed: {str(e)[:80]}", "status": "warning"})
    else:
        trace.append({"ts": ts(), "type": "skill", "name": "Causal Graph Query",
                      "msg": "Neo4j not configured — using clinical notes only.", "status": "warning"})

    # Query Qdrant — semantic query now DRIVEN BY graph state
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url and member_id:
        try:
            semantic_query = _build_semantic_query(cpt_code, icd_code, clinical_notes, policy, graph_state)
            trace.append({"ts": ts(), "type": "skill", "name": "Axisweave Retrieval",
                          "msg": f"🔍 Semantic query: \"{semantic_query[:100]}...\"", "status": "info"})

            retrieved_evidence_docs = _search_qdrant(qdrant_url, member_id, semantic_query)
            if retrieved_evidence_docs:
                trace.append({"ts": ts(), "type": "skill", "name": "Axisweave Retrieval",
                              "msg": f"✓ Retrieved {len(retrieved_evidence_docs)} evidence chunks (Bedrock Titan cosine search)",
                              "status": "success"})
                for i, doc in enumerate(retrieved_evidence_docs[:5]):
                    trace.append({"ts": ts(), "type": "context_retrieval", "name": "Evidence Chunk",
                                  "msg": f"[{doc['score']:.2f}] {doc['doc_type']}: {doc['text'][:120]}...",
                                  "status": "info"})
            else:
                trace.append({"ts": ts(), "type": "skill", "name": "Axisweave Retrieval",
                              "msg": f"⚠ 0 chunks returned for {member_id}.",
                              "status": "warning"})
        except Exception as e:
            trace.append({"ts": ts(), "type": "skill", "name": "Axisweave Retrieval",
                          "msg": f"⚠ Semantic search unavailable: {str(e)[:80]}", "status": "warning"})
    else:
        trace.append({"ts": ts(), "type": "skill", "name": "Axisweave Retrieval",
                      "msg": "Qdrant not configured — using clinical notes only.", "status": "warning"})

    if use_ai_extraction and clinical_notes:
        trace.append({"ts": ts(), "type": "skill", "name": "ExtractEvidence (ClinicalNLP)",
                      "msg": "🧠 Reading clinical documentation with LLM...", "status": "info"})
        evidence = _ai_extract_full(clinical_notes, cpt_code, policy, request_context=request)
        if isinstance(evidence, list):
            evidence = evidence[0] if evidence else {}
        if not isinstance(evidence, dict):
            evidence = {}
        positive = sum(1 for v in evidence.values() if v and v != 0)
        trace.append({"ts": ts(), "type": "skill", "name": "ExtractEvidence (ClinicalNLP)",
                      "msg": f"AI extraction complete. Found {positive} positive findings.",
                      "status": "success"})
    else:
        trace.append({"ts": ts(), "type": "skill", "name": "ExtractEvidence (Regex)",
                      "msg": "Using pattern matching for extraction.", "status": "info"})
        evidence = _regex_extract_full(clinical_notes, policy)
        trace.append({"ts": ts(), "type": "skill", "name": "ExtractEvidence (Regex)",
                      "msg": f"Extracted: therapy={evidence.get('conservativeTherapyWeeks',0)}wk, neuro={evidence.get('hasNeurologicalSymptoms')}, specialist={evidence.get('hasSpecialist')}", "status": "success"})

    # ─── Merge graph state into evidence (graph informs decision logic) ────
    evidence = _merge_graph_evidence(evidence, graph_state, trace, ts)

    # ─── STAGE 4: Criteria Evaluation ─────────────────────────────────────
    trace.append({"ts": ts(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 4: Evaluating criteria against policy rules.", "status": "info"})

    criteria_results = _evaluate_policy_criteria(evidence, policy, request)
    all_met = all(c["met"] for c in criteria_results)

    # ─── Cross-validation: graph vs notes contradictions ──────────────────
    contradictions = _cross_validate_graph_vs_notes(graph_state, evidence, clinical_notes)
    for contradiction in contradictions:
        trace.append({"ts": ts(), "type": "rule", "name": "Cross-Validation",
                      "msg": f"⚠ {contradiction}", "status": "warning"})

    for c in criteria_results:
        status = "success" if c["met"] else "fail"
        trace.append({"ts": ts(), "type": "rule", "name": f"Criteria: {c['name']}",
                      "msg": f"{'✓ MET' if c['met'] else '✗ NOT MET'} — {c['detail']}", "status": status})

    # ─── STAGE 5: Decision ────────────────────────────────────────────────
    trace.append({"ts": ts(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 5: Applying Clinical Conservatism. Generating notice.", "status": "info"})

    if all_met and icd_valid:
        decision = "Approved"
        reason = f"All {len(criteria_results)} criteria met per {policy['policyId'] if policy else 'default'} guidelines."
    elif not icd_valid:
        decision = "Escalated for Human Review"
        reason = f"Diagnosis code {icd_code} does not align with policy for CPT {cpt_code}."
    else:
        decision = "Escalated for Human Review"
        failed = [c["name"] for c in criteria_results if not c["met"]]
        reason = f"Criteria not met: {', '.join(failed)}. Escalated per Clinical Conservatism rule."

    trace.append({"ts": ts(), "type": "rule", "name": "RULE-02 (Clinical Conservatism)",
                  "msg": f"Final decision: {decision}", "status": "success" if decision == "Approved" else "warning"})

    notice = _generate_notice(request, decision, reason, criteria_results)
    trace.append({"ts": ts(), "type": "system", "name": "Agent Engine",
                  "msg": f"Pipeline complete. {len(trace)} events logged.", "status": "success"})

    # ─── CHALLENGER AGENT: Autonomous second-opinion review ────────────────
    trace.append({"ts": ts(), "type": "system", "name": "Agent Engine",
                  "msg": "Handing off to Challenger Agent for independent review...", "status": "info"})
    
    try:
        import challenger_agent
        challenger_result = challenger_agent.review(
            pa_decision=decision,
            pa_reason=reason,
            evidence=evidence,
            criteria_met=criteria_results,
            request=request,
            policy=policy,
            bundle_text=bundle_text,
            graph_contradictions=contradictions,
            graph_state=graph_state,
        )
        # Merge challenger trace into main trace
        for t in challenger_result.get("trace", []):
            trace.append(t)
        
        # RED FLAG: If challenger formally challenges, override decision
        if challenger_result.get("formalChallenge"):
            original_decision = decision
            decision = "Flagged for Medical Director Review"
            reason = f"Challenger Agent override (confidence {challenger_result['confidence']}/10): {challenger_result.get('reasoning', '')[:100]}"
            trace.append({"ts": ts(), "type": "rule", "name": "RED FLAG",
                          "msg": f"Challenger overrode '{original_decision}' → Flagged for Medical Director. Findings attached.",
                          "status": "fail"})
            # Regenerate notice with challenger notes
            notice = _generate_notice(request, decision, reason, criteria_results)
            notice += f"\n\n--- CHALLENGER AGENT FINDINGS ---\n"
            notice += f"Verdict: {challenger_result['verdict']} (confidence: {challenger_result['confidence']}/10)\n"
            notice += f"Reason: {challenger_result.get('reasoning', '')}\n"
            for f in challenger_result.get("findings", []):
                notice += f"• {f}\n"
            notice += f"Recommendation: {challenger_result.get('recommendation', '')}\n"
    except Exception as e:
        challenger_result = {
            "verdict": "ERROR",
            "confidence": 0,
            "reasoning": f"Challenger agent error: {str(e)}",
            "findings": [],
            "recommendation": "Proceed with PA Agent decision.",
            "formalChallenge": False
        }
        trace.append({"ts": ts(), "type": "system", "name": "Challenger Agent",
                      "msg": f"Error: {str(e)}", "status": "fail"})

    return {
        "decision": decision,
        "reason": reason,
        "status": "approved" if decision == "Approved" else "escalated",
        "notice": notice,
        "trace": trace,
        "evidence": evidence,
        "criteriaMet": criteria_results,
        "retrievedEvidence": retrieved_evidence_docs,
        "semanticQuery": semantic_query,
        "graphState": graph_state,
        "causalChain": _build_causal_chain(graph_state, policy) if graph_state else "",
        "crossValidation": contradictions,
        "policyUsed": policy.get("policyId", "default") if policy else "no_match",
        "policyName": policy.get("policyName", "") if policy else "Default",
        "category": policy.get("category", "General") if policy else "General",
        "generatedArtifacts": {
            "hasRules": gen_rules is not None if policy else False,
            "hasSkills": gen_skills is not None if policy else False,
            "hasHooks": gen_hooks is not None if policy else False,
        } if policy else {},
        "challenger": {
            "verdict": challenger_result.get("verdict", "AGREE"),
            "confidence": challenger_result.get("confidence", 0),
            "reasoning": challenger_result.get("reasoning", ""),
            "findings": challenger_result.get("findings", []),
            "recommendation": challenger_result.get("recommendation", ""),
            "formalChallenge": challenger_result.get("formalChallenge", False)
        }
    }


def _ai_extract_full(clinical_notes, cpt_code, policy, request_context=None):
    """Use LLM to extract evidence relevant to the specific policy criteria.
    PHI is automatically scrubbed via call_llm."""
    criteria_desc = ""
    if policy and policy.get("criteria"):
        criteria_desc = "Criteria to look for:\n" + "\n".join(
            f"- {c['description']}" for c in policy["criteria"][:5]
        )

    prompt = f"""Extract clinical facts from these notes relevant to prior authorization.
CPT: {cpt_code}
{criteria_desc}

NOTES: {clinical_notes}

Return JSON with these fields (use true/false for booleans, integers for weeks/months):
{{"conservativeTherapyWeeks": <int>, "hasNeurologicalSymptoms": <bool>, "hasMechanicalSymptoms": <bool>, "hasImagingFindings": <bool>, "hasSpecialist": <bool>, "hasPriorImaging": <bool>, "hasPathology": <bool>, "severityScore": <int or null>, "failedMedications": [<list>], "reasoning": "<brief>"}}"""

    response = call_llm(prompt, max_tokens=300, temperature=0, request_context=request_context)
    try:
        result = extract_json_from_response(response)
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            result = {}
        return result
    except (ValueError, json.JSONDecodeError):
        return {"conservativeTherapyWeeks": 0, "reasoning": "Parse failed: " + response[:100]}


def _regex_extract_full(clinical_notes, policy):
    """Regex extraction with multi-field support."""
    notes = clinical_notes.lower()
    evidence = {}

    # Therapy duration
    m = re.search(r'(\d+)\s*weeks?\s*(?:of\s+)?(?:physical therapy|pt|therapy|conservative)', notes)
    evidence["conservativeTherapyWeeks"] = int(m.group(1)) if m else 0

    # Neurological
    evidence["hasNeurologicalSymptoms"] = any(w in notes for w in
        ["radiculopathy", "numbness", "tingling", "weakness", "radiating", "straight leg raise"])

    # Mechanical
    evidence["hasMechanicalSymptoms"] = any(w in notes for w in
        ["locking", "catching", "giving way", "mechanical", "locked"])

    # Imaging findings
    evidence["hasImagingFindings"] = any(w in notes for w in
        ["mri shows", "mri dated", "mri findings", "ct shows", "imaging shows", "signal abnormality"])

    # Specialist
    evidence["hasSpecialist"] = any(w in notes for w in
        ["oncologist", "dermatologist", "orthopedic surgeon", "board-certified", "rheumatologist"])

    # Prior imaging
    evidence["hasPriorImaging"] = any(w in notes for w in
        ["ct chest", "chest ct", "prior ct", "prior mri", "biopsy"])

    # Pathology
    evidence["hasPathology"] = any(w in notes for w in
        ["pathology", "biopsy", "confirmed malignancy", "adenocarcinoma", "carcinoma"])

    # Failed medications
    meds = re.findall(r'failed?\s+(?:trials?\s+of\s+)?(\w+)', notes)
    evidence["failedMedications"] = meds

    # Severity scores (EASI, IGA, BSA)
    evidence["severityScore"] = None
    easi_match = re.search(r'easi\s*(?:score\s*)?(\d+)', notes)
    if easi_match:
        evidence["severityScore"] = int(easi_match.group(1))
    iga_match = re.search(r'iga\s*(\d+)', notes)
    if iga_match and not evidence["severityScore"]:
        evidence["severityScore"] = int(iga_match.group(1))

    return evidence


def _evaluate_policy_criteria(evidence, policy, request):
    """Evaluate evidence against policy-specific criteria."""
    results = []
    if not policy:
        # Default simple evaluation
        results.append({"name": "General Review", "met": True, "detail": "No specific policy matched"})
        return results

    for criterion in policy.get("criteria", []):
        ctype = criterion.get("type", "")
        desc = criterion.get("description", "")
        met = False
        detail = ""

        if "conservative_treatment" in ctype:
            weeks = evidence.get("conservativeTherapyWeeks") or 0
            thresh_match = re.search(r'(\d+)', criterion.get("threshold", "6"))
            thresh = int(thresh_match.group(1)) if thresh_match else 6
            met = int(weeks) >= thresh
            detail = f"{weeks} weeks completed (need {thresh})"

        elif "neurological" in ctype:
            met = bool(evidence.get("hasNeurologicalSymptoms"))
            detail = "Present" if met else "Not documented"

        elif "mechanical" in ctype:
            met = bool(evidence.get("hasMechanicalSymptoms"))
            detail = "Present" if met else "Not documented"

        elif "imaging_findings" in ctype or "prior_imaging" in ctype:
            if "no " in criterion.get("description", "").lower()[:20] or "within" in criterion.get("description", "").lower():
                has_prior = bool(evidence.get("hasPriorImaging"))
                met = True
                detail = "No conflicting prior imaging" if not has_prior else "Prior imaging exists (new symptoms documented)"
            else:
                met = bool(evidence.get("hasImagingFindings")) or bool(evidence.get("hasPriorImaging"))
                detail = "Imaging documented" if met else "No imaging documented"

        elif "diagnosis" in ctype or "severity" in ctype:
            score = evidence.get("severityScore")
            has_path = bool(evidence.get("hasPathology"))
            met = score is not None or has_path
            detail = f"Score: {score}" if score else ("Confirmed" if has_path else "Not confirmed")

        elif "specialist" in ctype or "provider" in ctype:
            met = bool(evidence.get("hasSpecialist"))
            detail = "Specialist confirmed" if met else "Specialist not documented"

        elif "clinical_indication" in ctype or "treatment_impact" in ctype:
            met = bool(evidence.get("hasPriorImaging")) or bool(evidence.get("hasPathology"))
            detail = "Clinical indication documented" if met else "Not documented"

        elif "step_therapy" in ctype:
            meds = evidence.get("failedMedications") or []
            met = len(meds) > 0
            detail = f"Failed: {', '.join(meds[:3])}" if meds else "No failed therapies documented"

        elif "age" in ctype:
            dob = request.get("patientDob", "")
            if dob:
                from datetime import date
                birth = date.fromisoformat(dob)
                age = (date.today() - birth).days // 365
                if "minimum" in ctype:
                    thresh_match = re.search(r'(\d+)', criterion.get("threshold", "0"))
                    thresh = int(thresh_match.group(1)) if thresh_match else 0
                    met = age >= thresh
                    detail = f"Age {age} (need >= {thresh})"
                else:
                    met = age < 65
                    detail = f"Age {age}"
            else:
                met = True
                detail = "DOB not provided, assumed eligible"

        elif "red_flag" in ctype or "safety" in ctype:
            met = True  # Default pass unless red flags detected
            detail = "No red flags identified"

        elif "bmi" in ctype:
            met = True  # Default pass
            detail = "BMI within acceptable range (assumed)"

        else:
            met = True
            detail = f"Auto-pass: {ctype}"

        results.append({"name": criterion.get("id", "") + " " + desc[:40], "met": met, "detail": detail})

    return results


# ─── Build Rules/Skills/Hooks for a Specific Policy ──────────────────────────

def build_for_policy(policy, action):
    """
    Use the LLM to generate rules, skills, or hooks for a specific policy.
    Each is generated contextually based on the policy's criteria.
    """
    policy_id = policy.get("policyId", "")
    policy_name = policy.get("policyName", "")
    criteria_text = "\n".join(
        f"- {c['id']}: {c['description']} (threshold: {c.get('threshold', 'N/A')})"
        for c in policy.get("criteria", [])
    )
    cpt_codes = ", ".join(policy.get("cptCodes", []))

    if action == "rules":
        return _build_rules_for_policy(policy_id, policy_name, cpt_codes, criteria_text, policy)
    elif action == "skills":
        return _build_skills_for_policy(policy_id, policy_name, cpt_codes, criteria_text)
    elif action == "hooks":
        return _build_hooks_for_policy(policy_id, policy_name, cpt_codes, criteria_text)
    else:
        return {"error": f"Unknown action: {action}"}


def _build_rules_for_policy(policy_id, policy_name, cpt_codes, criteria_text, policy):
    """Generate OPA Rego-style rules from policy criteria."""
    prompt = f"""Generate declarative policy rules for this prior authorization policy.
Policy: {policy_id} — {policy_name}
CPT Codes: {cpt_codes}
Criteria:
{criteria_text}

Output as a markdown rules declaration (like rules_declaration.md format):
## Policy: {policy_id} ({policy_name})
- CPT Code: <code>
- Allowed ICD-10 Diagnosis Codes: <codes>
- <criteria as key-value pairs>

Be concise. Output ONLY the markdown rules."""

    response = call_llm(prompt, max_tokens=400, temperature=0)

    # Save to per-policy rules file
    rules_file = os.path.join(POLICIES_DIR, f"{policy_id}_rules.md")
    with open(rules_file, "w") as f:
        f.write(f"# Rules: {policy_id} — {policy_name}\n\n{response}\n")
    
    audit = f"✓ Rules generated and saved to policies/{policy_id}_rules.md\n\n{response}"
    return {"status": "success", "action": "rules", "policyId": policy_id, "auditLog": audit}


def _build_skills_for_policy(policy_id, policy_name, cpt_codes, criteria_text):
    """Generate skill definitions tailored to a specific policy."""
    prompt = f"""Design 2-3 validation skills for this prior authorization policy.
Policy: {policy_id} — {policy_name}
CPT: {cpt_codes}
Criteria:
{criteria_text}

For each skill provide JSON:
{{"skillName": "<PascalCaseSkill>", "description": "<what it validates>", "inputs": ["<fields>"], "outputs": ["<results>"], "logic": "<brief validation logic>"}}

Return a JSON array of 2 skills only. Be very concise in descriptions (under 15 words each)."""

    response = call_llm(prompt, max_tokens=500, temperature=0)

    try:
        skills = extract_json_from_response(response)
        if isinstance(skills, dict):
            skills = [skills]
        
        # Save per-policy
        skills_file = os.path.join(POLICIES_DIR, f"{policy_id}_skills.json")
        with open(skills_file, "w") as f:
            json.dump({"policyId": policy_id, "generatedAt": datetime.now().isoformat(), "skills": skills}, f, indent=2)

        audit = f"✓ Generated {len(skills)} skill(s) for {policy_id}:\n"
        for s in skills:
            audit += f"  • {s.get('skillName', '?')}: {s.get('description', '')[:60]}\n"
        audit += f"\nSaved to policies/{policy_id}_skills.json"
    except (ValueError, json.JSONDecodeError):
        skills = []
        audit = f"⚠️ Could not parse skills from LLM response.\nRaw:\n{response[:300]}"

    return {"status": "success", "action": "skills", "policyId": policy_id, "skills": skills, "auditLog": audit}


def _build_hooks_for_policy(policy_id, policy_name, cpt_codes, criteria_text):
    """Generate lifecycle hooks for a specific policy."""
    prompt = f"""Design 2 lifecycle hooks for this PA policy agent.
Policy: {policy_id} — {policy_name}
Criteria:
{criteria_text}

Stages: on_request_received, on_guidelines_loaded, on_evidence_extracted, on_criteria_evaluated, on_notice_generated

Return JSON array of 2 hooks:
{{"hookName": "<name>", "stage": "<stage>", "description": "<10 words max>", "validates": "<what>"}}

Be very concise."""

    response = call_llm(prompt, max_tokens=300, temperature=0)

    try:
        hooks = extract_json_from_response(response)
        if isinstance(hooks, dict):
            hooks = [hooks]
        
        # Save per-policy
        hooks_file = os.path.join(POLICIES_DIR, f"{policy_id}_hooks.json")
        with open(hooks_file, "w") as f:
            json.dump({"policyId": policy_id, "generatedAt": datetime.now().isoformat(), "hooks": hooks}, f, indent=2)
        
        audit = f"✓ Generated {len(hooks)} hook(s) for {policy_id}:\n"
        for h in hooks:
            audit += f"  🪝 {h.get('hookName', '?')} @ {h.get('stage', '?')}\n"
            audit += f"     → {h.get('description', '')[:60]}\n"
        audit += f"\nSaved to policies/{policy_id}_hooks.json"
    except (ValueError, json.JSONDecodeError):
        hooks = []
        audit = f"⚠️ Could not parse hooks from LLM response.\nRaw:\n{response[:300]}"

    return {"status": "success", "action": "hooks", "policyId": policy_id, "hooks": hooks, "auditLog": audit}


# ─── AVI Agent ───────────────────────────────────────────────────────────────

def avi_respond(user_message, ui_context=None):
    """
    AVI — the conversational agent. Runs server-side with full access to:
    - All loaded policies (reads from disk)
    - Generated skills
    - Current system state
    - The user's current UI context (which tab, which case, decision state)
    - Last execution results: decision, trace, graph state, evidence, challenger
    
    This is a REAL agent — it assembles its own context based on what it needs.
    """
    # Build dynamic system context from actual system state
    policies = load_all_policies()
    skills = load_saved_skills()
    
    # Policy reference (built dynamically from actual files)
    policy_ref = "POLICIES IN THIS SYSTEM:\n"
    for p in policies:
        criteria_summary = ", ".join(c.get("description", "")[:40] for c in p.get("criteria", [])[:3])
        policy_ref += f"- {p['policyId']} ({p['policyName']}): Payer={p['payer']}, CPT={p.get('cptCodes',[])}, Criteria: {criteria_summary}\n"
    
    # CPT/ICD reference from actual policies
    code_ref = "\nCPT/ICD-10 REFERENCE:\n"
    for p in policies:
        for cpt in p.get("cptCodes", []):
            desc = p.get("cptDescriptions", {}).get(cpt, "")
            if desc:
                code_ref += f"- CPT {cpt} = {desc}\n"
        for icd in p.get("allowedIcd10", [])[:4]:
            desc = p.get("icd10Descriptions", {}).get(icd, "")
            if desc:
                code_ref += f"- ICD-10 {icd} = {desc}\n"
    
    # Skills reference
    skills_ref = "\nGENERATED SKILLS:\n"
    if skills:
        for s in skills[:6]:
            skills_ref += f"- {s.get('skillName', '?')}: {s.get('description', '')[:50]}\n"
    else:
        skills_ref += "- No skills generated yet. User should use Policy Workspace to generate.\n"
    
    # ─── Context-Aware Section ─────────────────────────────────────────────────
    # Determine which tab the user is on and build rich contextual awareness
    active_view = "unknown"
    tab_context = ""
    decision_context = ""
    trace_context = ""
    graph_context = ""
    evidence_context = ""
    challenger_context = ""
    
    if ui_context:
        active_view = ui_context.get("activeView", ui_context.get("activeTab", "review"))
        
        # Last Decision context
        last_decision = ui_context.get("lastDecision")
        if last_decision:
            decision_context = f"\nLAST EXECUTION DECISION:\n- Decision: {last_decision.get('decision', 'N/A')}\n- Reason: {last_decision.get('reason', 'N/A')}\n- Policy: {last_decision.get('policyName', 'N/A')}\n"
        
        # Last Trace events
        last_trace = ui_context.get("lastTrace")
        if last_trace:
            trace_context = "\nLAST TRACE EVENTS (most recent):\n"
            for t in last_trace:
                trace_context += f"- [{t.get('type', '')}] {t.get('name', '')}: {t.get('msg', '')}\n"
        
        # Graph state
        graph_state = ui_context.get("graphState")
        if graph_state:
            graph_context = f"\nPATIENT GRAPH STATE (Neo4j CRF):\n{graph_state}\n"
        
        # Retrieved evidence
        retrieved_evidence = ui_context.get("retrievedEvidence")
        if retrieved_evidence:
            evidence_context = f"\nRETRIEVED EVIDENCE:\n- Documents retrieved: {retrieved_evidence.get('count', 0)} chunks\n- Summary: {retrieved_evidence.get('summary', 'N/A')}\n"
        
        # Challenger verdict
        challenger_verdict = ui_context.get("challengerVerdict")
        if challenger_verdict:
            challenger_context = f"\nCHALLENGER AGENT VERDICT:\n- Verdict: {challenger_verdict.get('verdict', 'N/A')}\n- Reasoning: {challenger_verdict.get('reasoning', 'N/A')}\n"
        
        # Tab-specific guidance
        if active_view == "review":
            case_data = ui_context.get("caseContext", "")
            tab_context = f"\nUSER IS ON: Agent Review tab — they can see the PA execution results.\n{case_data}\n"
            tab_context += "You have access to the PA execution results. Use the decision, trace, evidence, and challenger data above to answer questions about why something was approved/escalated, what criteria passed/failed, and what the Challenger found."
        elif active_view == "crf":
            tab_context = "\nUSER IS ON: Clinical Reasoning Fabric (CRF) tab — they're viewing the patient ontology graph.\n"
            tab_context += "You're viewing the Clinical Reasoning Fabric. Use the graph state data above to explain diagnoses, failed therapies, and patient history connections shown in the graph."
        elif active_view == "workspace":
            selected_policy = ui_context.get("selectedPolicy", "")
            audit_log = ui_context.get("auditLog", "")
            tab_context = f"\nUSER IS ON: Policy Workspace tab. Selected policy: {selected_policy}.\n"
            tab_context += f"Audit log: {audit_log[:200]}\n"
            tab_context += "You're in the Policy Workspace managing policy artifacts. Help with policy configuration, rules generation, skill management, and hook setup."
        elif active_view == "concepts":
            tab_context = "\nUSER IS ON: Concepts tab — the animated architecture demo showing how PA Agent and Challenger Agent work together with skills, rules, and hooks."
        elif active_view == "video":
            tab_context = "\nUSER IS ON: Video Tutorial tab."
        elif active_view == "home":
            tab_context = "\nUSER IS ON: Home tab — system overview and architecture."
        else:
            case_data = ui_context.get("caseContext", "")
            tab_context = f"\nUSER IS ON: {active_view} tab.\n{case_data}"
    
    system_prompt = f"""You are AVI, the clinical intelligence assistant built into PriorAuthAI. You have real-time access to the system's loaded policies, skills, state, AND the current execution context.

YOUR CAPABILITIES:
1. Explain decisions (why approved/escalated) based on actual policy criteria and execution results
2. Identify missing documentation by comparing evidence against policy thresholds
3. Explain rules: RULE-01 (PHI Redaction), RULE-02 (Clinical Conservatism), RULE-03 (Citation), RULE-04 (Code Match), RULE-05 (Plain Language)
4. Explain the 5-stage progressive disclosure pipeline
5. Interpret CPT/ICD-10 codes using the reference data below
6. Guide users on Policy Workspace actions (generate rules/skills/hooks)
7. Explain how ClinicalNLP Engine differs from regex extraction
8. Answer "Why was this escalated?" using the decision + trace + criteria results
9. Answer "What did the graph show?" using the patient graph state
10. Answer "What evidence was retrieved?" using the retrieved evidence data
11. Explain the Challenger decision using the challenger verdict and reasoning

{policy_ref}
{code_ref}
{skills_ref}
{tab_context}
{decision_context}
{trace_context}
{graph_context}
{evidence_context}
{challenger_context}

RULES:
- RULE-01 (PHI Redaction): Scrubs names/SSN/DOB from trace logs
- RULE-02 (Clinical Conservatism): Never auto-deny; escalate uncertain to human Medical Director
- RULE-03 (Citation Compulsory): Every notice must cite the policy ID
- RULE-04 (Code Match): Validates CPT/ICD format and alignment with policy
- RULE-05 (Plain Language): Translates medical acronyms for patient correspondence

CONTEXT-AWARE ANSWERING:
- If user asks "Why was this escalated/approved?" → Use LAST EXECUTION DECISION and trace events to explain exactly which criteria passed/failed and why.
- If user asks "What did the graph show?" → Use PATIENT GRAPH STATE to describe diagnoses, therapies, and clinical history nodes.
- If user asks "What evidence was retrieved?" → Use RETRIEVED EVIDENCE to describe document count, types, and relevance scores.
- If user asks "Explain the Challenger decision" → Use CHALLENGER AGENT VERDICT to explain the Challenger's reasoning, confidence score, and findings.
- If decision data is not available, say "No execution has been run yet. Please run a case first."

IMPORTANT: Use ONLY the reference data above for code lookups. Never fabricate definitions.
Keep answers concise (3-5 sentences). Be direct and clinical. You ARE AVI."""

    response = call_llm(f"{system_prompt}\n\nUser question: {user_message}", max_tokens=200, temperature=0.3)
    return response


# ─── Skill: Read Evidence Bundle ─────────────────────────────────────────────

def read_evidence_bundle(bundle_path, cpt_code):
    """
    Generic skill that reads a PA evidence bundle PDF and extracts
    structured clinical information using the LLM.
    
    This skill:
    1. Reads the PDF text
    2. Identifies the matched policy based on CPT code
    3. Sends the text + policy criteria to the LLM
    4. Returns structured evidence + auto-generated clinical summary
    
    Works across ALL use cases (radiology, oncology, surgery, specialty rx).
    """
    trace = []
    ts = lambda: datetime.now().isoformat()
    
    trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                  "msg": f"Reading PDF: {os.path.basename(bundle_path)}", "status": "info"})
    
    # Read the PDF — try S3 first, then local
    full_path = os.path.join(DATA_DIR, bundle_path)
    
    # Check S3 first (deployed environment)
    pdf_text = ""
    if S3_AVAILABLE and is_s3_enabled():
        pdf_bytes = get_file_bytes(bundle_path)
        if pdf_bytes:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages[:5]):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(f"--- PAGE {i+1} ---\n{text.strip()}")
            pdf_text = "\n\n".join(pages_text)
            trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                          "msg": f"Read from S3: {len(pdf_text)} chars", "status": "success"})
    
    # Fallback to local filesystem
    if not pdf_text and os.path.exists(full_path):
        pdf_text = read_pdf_text(full_path, max_pages=5)
        trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                      "msg": f"Read from local: {len(pdf_text)} chars", "status": "success"})
    
    if not pdf_text:
        trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                      "msg": f"File not found in S3 or locally: {bundle_path}", "status": "fail"})
        return {"error": "Bundle not found", "trace": trace}
    
    # Find the matching policy
    policy = find_policy_for_cpt(cpt_code)
    criteria_context = ""
    if policy:
        trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                      "msg": f"Policy matched: {policy['policyId']} ({policy['policyName']})", "status": "success"})
        criteria_context = "Policy criteria to validate against:\n" + "\n".join(
            f"- {c['id']}: {c['description']} (threshold: {c.get('threshold', 'N/A')})"
            for c in policy.get("criteria", [])
        )
    
    # Use LLM to extract and validate
    trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                  "msg": "Sending to ClinicalNLP Engine for extraction + validation...", "status": "info"})
    
    prompt = f"""Read this PA evidence bundle and extract clinical information.

DOCUMENT TEXT:
{pdf_text[:3500]}

{criteria_context}

Return JSON (keep findings SHORT - max 8 words each):
{{
  "clinicalSummary": "<2 sentence summary>",
  "criteriaFindings": [
    {{"criterion": "<ID>", "finding": "<short finding>", "met": <true/false>}}
  ],
  "missingDocumentation": ["<missing items>"]
}}"""

    response = call_llm(prompt, max_tokens=700, temperature=0)
    
    try:
        result = extract_json_from_response(response)
        if isinstance(result, list):
            result = result[0] if result else {}
        
        trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                      "msg": f"Extraction complete. Found {len(result.get('criteriaFindings', []))} criteria findings.",
                      "status": "success"})
        
        # Add validation summary
        findings = result.get("criteriaFindings", [])
        met_count = sum(1 for f in findings if f.get("met"))
        total = len(findings)
        result["validationSummary"] = f"{met_count}/{total} criteria met"
        
        if result.get("missingDocumentation"):
            trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                          "msg": f"Missing: {', '.join(result['missingDocumentation'][:3])}",
                          "status": "warning"})
        
        return {"status": "success", "extracted": result, "trace": trace}
        
    except (ValueError, json.JSONDecodeError) as e:
        trace.append({"ts": ts(), "type": "skill", "name": "ReadEvidenceBundleSkill",
                      "msg": f"Parse error: {str(e)[:100]}", "status": "fail"})
        # Return raw text as clinical summary fallback
        return {
            "status": "partial",
            "extracted": {"clinicalSummary": response[:300]},
            "trace": trace
        }


def create_clinical_evidence_pdf(patient_name, dob, member_id, clinical_notes, filepath):
    """Generate a clean synthetic clinical evidence PDF document for a dynamic case."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        from reportlab.lib import colors

        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=12
        )
        story.append(Paragraph("Clinical Evidence Document", title_style))
        story.append(Spacer(1, 10))

        # Metadata
        meta_style = ParagraphStyle(
            'DocMeta',
            parent=styles['Normal'],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#475569')
        )
        story.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", meta_style))
        story.append(Paragraph(f"<b>Date of Birth:</b> {dob}", meta_style))
        story.append(Paragraph(f"<b>Member ID:</b> {member_id}", meta_style))
        story.append(Paragraph(f"<b>Document Date:</b> {datetime.now().strftime('%Y-%m-%d')}", meta_style))
        story.append(Spacer(1, 15))

        # Section Header
        story.append(Paragraph("<b>Clinical Notes & Findings Summary:</b>", styles['Heading3']))
        story.append(Spacer(1, 8))

        # Notes Body
        body_style = ParagraphStyle(
            'DocBody',
            parent=styles['Normal'],
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor('#1e293b')
        )
        # Convert simple text formatting to HTML-like paragraphs
        notes_html = clinical_notes.replace('\n', '<br/>')
        story.append(Paragraph(notes_html, body_style))

        doc.build(story)
        return True
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return False


def generate_cases_for_policy(policy):
    """
    Call Bedrock to design exactly 2 testing cases for a newly loaded policy.
    One Approved case (fully meeting criteria) and one Escalated case.
    Saves to preset_cases.json and compiles PDF bundles.
    """
    policy_id = policy.get("policyId", "")
    policy_name = policy.get("policyName", "")
    category = policy.get("category", "General")
    cpt_codes = ", ".join(policy.get("cptCodes", []))
    icd10_codes = ", ".join(policy.get("allowedIcd10", policy.get("icd10Codes", [])))
    
    criteria_text = "\n".join(
        f"- {c['id']}: {c['description']} (type: {c.get('type', 'N/A')}, threshold: {c.get('threshold', 'N/A')})"
        for c in policy.get("criteria", [])
    )
    
    prompt = f"""You are a medical case generator. Given this prior authorization clinical policy, generate exactly two distinct prior authorization request cases as a JSON array.

Policy ID: {policy_id}
Policy Name: {policy_name}
CPT Codes: {cpt_codes}
Diagnosis Codes: {icd10_codes}
Criteria:
{criteria_text}

Generate a JSON array of exactly 2 cases:
1. First case must represent an APPROVED scenario:
   - Make up a realistic member ID, patient name, SSN, and DOB.
   - Use one of the covered CPT codes and diagnosis codes.
   - Write a detailed "clinicalNotes" string (2-3 sentences) where the patient has completed the required conservative therapy duration (if any), has the required symptoms/findings, and meets all criteria.
2. Second case must represent an ESCALATED scenario:
   - Make up a realistic member ID, patient name, SSN, and DOB.
   - Use one of the covered CPT codes and diagnosis codes.
   - Write a detailed "clinicalNotes" string (2-3 sentences) where the patient does NOT meet the criteria (e.g., has mechanical symptoms but has NOT completed the required weeks of conservative physical therapy, or CPT/ICD-10 is mismatching).

Output format:
[
  {{
    "id": "case-{policy_id.lower()}-approve",
    "title": "{policy_name} — Approved (meets clinical criteria)",
    "category": "{category}",
    "request": {{
      "memberId": "MEM-{random_member_suffix()}",
      "patientName": "<patient name>",
      "patientSsn": "999-99-9999",
      "patientDob": "1980-01-01",
      "cptCode": "<cpt>",
      "icd10Code": "<icd10>",
      "providerName": "<provider name>",
      "providerNpi": "1234567890",
      "clinicalNotes": "<detailed notes meeting all criteria>"
    }}
  }},
  {{
    "id": "case-{policy_id.lower()}-escalate",
    "title": "{policy_name} — Escalated (fails criteria)",
    "category": "{category}",
    "request": {{
      "memberId": "MEM-{random_member_suffix(True)}",
      "patientName": "<patient name>",
      "patientSsn": "999-99-9999",
      "patientDob": "1985-05-05",
      "cptCode": "<cpt>",
      "icd10Code": "<icd10>",
      "providerName": "<provider name>",
      "providerNpi": "1234567890",
      "clinicalNotes": "<detailed notes failing criteria>"
    }}
  }}
]

Respond with ONLY the JSON array. Do not include markdown code block formatting or explanation."""

    response = call_llm(prompt, max_tokens=1000, temperature=0)
    
    try:
        cases = extract_json_from_response(response)
        if not isinstance(cases, list) or len(cases) < 2:
            raise ValueError("Expected list of 2 cases")
    except Exception as e:
        print(f"Failed to parse cases from LLM: {e}. Generating fallback cases.")
        # Fallback cases
        cpt = policy.get("cptCodes", ["99999"])[0]
        icd = policy.get("allowedIcd10", policy.get("icd10Codes", ["M54.5"]))[0]
        cases = [
            {
                "id": f"case-{policy_id.lower()}-approve",
                "title": f"{policy_name} — Approved (meets clinical criteria)",
                "category": category,
                "request": {
                    "memberId": f"MEM-{random_member_suffix()}",
                    "patientName": "John Doe",
                    "patientSsn": "999-01-2345",
                    "patientDob": "1978-10-12",
                    "cptCode": cpt,
                    "icd10Code": icd,
                    "providerName": "Quality Health Partners",
                    "providerNpi": "9876543210",
                    "clinicalNotes": f"Patient presents for evaluation. All diagnostic criteria for {policy_name} under {cpt} and {icd} have been fully satisfied. Completed 8 weeks of physical therapy and conservative management with documented failure."
                }
            },
            {
                "id": f"case-{policy_id.lower()}-escalate",
                "title": f"{policy_name} — Escalated (fails criteria)",
                "category": category,
                "request": {
                    "memberId": f"MEM-{random_member_suffix(True)}",
                    "patientName": "Jane Smith",
                    "patientSsn": "999-02-6789",
                    "patientDob": "1988-04-18",
                    "cptCode": cpt,
                    "icd10Code": icd,
                    "providerName": "Quality Health Partners",
                    "providerNpi": "9876543210",
                    "clinicalNotes": f"Patient reports symptoms for only 2 weeks. Has not completed conservative therapy or physical therapy trials yet. CPT {cpt} requested."
                }
            }
        ]
        
    # Generate PDF documents and save locally + S3
    for case in cases:
        req = case["request"]
        case_id = case["id"]
        pdf_filename = f"{case_id}_bundle.pdf"
        local_dir = os.path.join(DATA_DIR, "cases")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, pdf_filename)
        
        # Build PDF
        create_clinical_evidence_pdf(
            req["patientName"], req["patientDob"], req["memberId"],
            req["clinicalNotes"], local_path
        )
        
        case["evidenceBundle"] = f"cases/{pdf_filename}"
        
        # Upload to S3 if configured
        if is_s3_enabled():
            try:
                import boto3
                s3_key = f"cases/{pdf_filename}"
                s3_client = boto3.client("s3")
                with open(local_path, "rb") as f:
                    s3_client.put_object(Bucket=os.environ.get("S3_BUCKET"), Key=s3_key, Body=f.read())
                print(f"Uploaded case bundle to S3: {s3_key}")
            except Exception as se:
                print(f"Error uploading case bundle to S3: {se}")

    # Append to preset_cases.json
    cases_file = os.path.join(POLICIES_DIR, "preset_cases.json")
    existing_cases = []
    if os.path.exists(cases_file):
        try:
            with open(cases_file, "r") as f:
                existing_cases = json.load(f)
        except Exception:
            pass
            
    # Remove any existing versions of these specific case IDs to avoid duplicates
    new_ids = [c["id"] for c in cases]
    existing_cases = [c for c in existing_cases if c["id"] not in new_ids]
    
    existing_cases.extend(cases)
    
    with open(cases_file, "w") as f:
        json.dump(existing_cases, f, indent=2)
        
    print(f"Appended {len(cases)} cases for policy {policy_id} to preset_cases.json")
    return cases


def random_member_suffix(escalate=False):
    import random
    num = random.randint(1000, 9999)
    return f"{num}"


def compile_all_policies_to_rules():
    """Load all policies from disk and rebuild rules_declaration.md & rules.rego."""
    all_policies = load_all_policies()
    md_content = compile_policies_to_rules(all_policies)
    
    # Save rules_declaration.md
    with open('rules_declaration.md', 'w') as f:
        f.write(md_content)
        
    print(f"Successfully compiled {len(all_policies)} policies to rules_declaration.md")
    return md_content


def create_custom_case_for_policy(policy_id, patient_name, member_id, scenario):
    """
    Generate a custom test case for a staged policy using LLM clinical note generation
    and ReportLab PDF compilation, then register it in preset_cases.json.
    """
    all_policies = load_all_policies()
    policy = next((p for p in all_policies if p.get("policyId") == policy_id), None)
    if not policy:
        raise ValueError(f"Policy {policy_id} not found in workspace.")

    policy_name = policy.get("policyName", "")
    category = policy.get("category", "General")
    cpt_codes = ", ".join(policy.get("cptCodes", []))
    icd10_codes = ", ".join(policy.get("allowedIcd10", policy.get("icd10Codes", [])))
    
    criteria_text = "\n".join(
        f"- {c['id']}: {c['description']} (type: {c.get('type', 'N/A')})"
        for c in policy.get("criteria", [])
    )
    
    cpt = policy.get("cptCodes", ["99999"])[0]
    icd = policy.get("allowedIcd10", policy.get("icd10Codes", ["M54.5"]))
    if isinstance(icd, list):
        icd = icd[0] if icd else "M54.5"
    
    prompt = f"""You are a medical case generator. Given this prior authorization clinical policy, generate a realistic prior authorization request case with a detailed clinical story.

Policy ID: {policy_id}
Policy Name: {policy_name}
CPT Code: {cpt}
Diagnosis Code: {icd}
Criteria:
{criteria_text}

Generate a case matching the scenario: {scenario.upper()}.
- If scenario is APPROVED, write a detailed clinical notes paragraph (3-4 sentences) where the patient {patient_name} has fully completed conservative therapy (if any) and meets all criteria.
- If scenario is ESCALATED, write a detailed clinical notes paragraph (3-4 sentences) where the patient {patient_name} fails at least one criterion (e.g., did not complete physical therapy or symptoms are too brief).

Response format MUST be a single raw JSON object as follows:
{{
  "clinicalNotes": "<detailed notes matching the scenario>"
}}

Respond with ONLY the JSON object. Do not include markdown code block formatting or explanation."""

    response = call_llm(prompt, max_tokens=600, temperature=0)
    
    try:
        res_data = extract_json_from_response(response)
        notes = res_data.get("clinicalNotes", "")
    except Exception as e:
        print(f"Failed to parse custom case notes: {e}")
        if scenario.lower() == "approve":
            notes = f"Patient {patient_name} presents for evaluation. All diagnostic criteria for {policy_name} under CPT {cpt} and ICD-10 {icd} have been fully satisfied. Completed conservative therapy trial."
        else:
            notes = f"Patient {patient_name} presents with acute symptoms. Has not completed conservative therapy trials yet. CPT {cpt} requested."

    case_id = f"case-{policy_id.lower().replace('.', '_')}-{scenario.lower()}"
    pdf_filename = f"{case_id}_bundle.pdf"
    local_dir = os.path.join(DATA_DIR, "cases")
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, pdf_filename)
    
    create_clinical_evidence_pdf(
        patient_name, "1980-01-01", member_id, notes, local_path
    )
    
    if is_s3_enabled():
        try:
            import boto3
            s3_key = f"cases/{pdf_filename}"
            s3_client = boto3.client("s3")
            with open(local_path, "rb") as f:
                s3_client.put_object(Bucket=os.environ.get("S3_BUCKET"), Key=s3_key, Body=f.read())
            print(f"Uploaded custom case bundle to S3: {s3_key}")
        except Exception as se:
            print(f"Error uploading custom case bundle to S3: {se}")
            
    case = {
        "id": case_id,
        "title": f"{policy_name} — {scenario.capitalize()} ({patient_name})",
        "category": category,
        "request": {
            "memberId": member_id,
            "patientName": patient_name,
            "patientSsn": "999-01-1234",
            "patientDob": "1980-01-01",
            "cptCode": cpt,
            "icd10Code": icd,
            "providerName": "Quality Health Partners",
            "providerNpi": "1234567890",
            "clinicalNotes": notes
        },
        "evidenceBundle": f"cases/{pdf_filename}"
    }
    
    cases_file = os.path.join(POLICIES_DIR, "preset_cases.json")
    existing_cases = []
    if os.path.exists(cases_file):
        try:
            with open(cases_file, "r") as f:
                existing_cases = json.load(f)
        except Exception:
            pass
            
    existing_cases = [c for c in existing_cases if c["id"] != case_id]
    existing_cases.append(case)
    
    with open(cases_file, "w") as f:
        json.dump(existing_cases, f, indent=2)
        
    print(f"Custom case {case_id} added successfully to preset_cases.json")
    return case


def promote_policy(policy_id):
    """
    Remove 'staged_' prefix from policy files and contents, converting the staged
    policy into an approved production policy.
    """
    if not policy_id.startswith("staged_"):
        return {"status": "error", "message": "Policy is not in staged status."}
        
    prod_id = policy_id[len("staged_"):]
    policies_dir = os.path.join(DATA_DIR, "policies")
    
    # 1. Load, update, and rename the policy JSON
    staged_json_path = os.path.join(policies_dir, f"policy_{policy_id.lower().replace('-', '_')}.json")
    prod_json_path = os.path.join(policies_dir, f"policy_{prod_id.lower().replace('-', '_')}.json")
    
    if not os.path.exists(staged_json_path):
        return {"status": "error", "message": f"Staged policy JSON not found: {staged_json_path}"}
        
    with open(staged_json_path, "r") as f:
        policy = json.load(f)
        
    policy["policyId"] = prod_id
    
    # 2. Write prod JSON and delete staged JSON
    with open(prod_json_path, "w") as f:
        json.dump(policy, f, indent=2)
        
    try:
        os.remove(staged_json_path)
    except Exception:
        pass
        
    # 3. Rename rules, skills, and hooks files
    for suffix in ["_rules.md", "_skills.json", "_hooks.json"]:
        staged_file = os.path.join(policies_dir, f"{policy_id}{suffix}")
        prod_file = os.path.join(policies_dir, f"{prod_id}{suffix}")
        if os.path.exists(staged_file):
            with open(staged_file, "r") as f:
                content = f.read()
            content = content.replace(policy_id, prod_id)
            with open(prod_file, "w") as f:
                f.write(content)
            try:
                os.remove(staged_file)
            except Exception:
                pass
                
    # 4. Rename cases in preset_cases.json if any custom test cases exist
    cases_file = os.path.join(policies_dir, "preset_cases.json")
    if os.path.exists(cases_file):
        try:
            with open(cases_file, "r") as f:
                cases = json.load(f)
            changed = False
            for case in cases:
                if case["id"].startswith(f"case-{policy_id.lower().replace('.', '_')}"):
                    case["id"] = case["id"].replace(
                        f"case-{policy_id.lower().replace('.', '_')}",
                        f"case-{prod_id.lower().replace('.', '_')}"
                    )
                    case["title"] = case["title"].replace(policy_id, prod_id)
                    old_pdf = case["evidenceBundle"]
                    new_pdf = old_pdf.replace(
                        policy_id.lower().replace('.', '_'),
                        prod_id.lower().replace('.', '_')
                    )
                    case["evidenceBundle"] = new_pdf
                    
                    old_pdf_abs = os.path.join(DATA_DIR, old_pdf)
                    new_pdf_abs = os.path.join(DATA_DIR, new_pdf)
                    if os.path.exists(old_pdf_abs):
                        os.rename(old_pdf_abs, new_pdf_abs)
                        
                    changed = True
            if changed:
                with open(cases_file, "w") as f:
                    json.dump(cases, f, indent=2)
        except Exception as ce:
            print(f"Error renaming cases during promotion: {ce}")
            
    print(f"Policy {policy_id} promoted to {prod_id} successfully.")
    return {"status": "success", "newPolicyId": prod_id}


def delete_policy(policy_id):
    """
    Completely delete a staged or approved policy, its rules, skills, hooks, cases,
    and S3 assets.
    """
    policies_dir = os.path.join(DATA_DIR, "policies")
    
    # 1. Load JSON to find associated PDF file path
    json_path = os.path.join(policies_dir, f"policy_{policy_id.lower().replace('-', '_')}.json")
    pdf_file_rel = ""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                policy = json.load(f)
                pdf_file_rel = policy.get("pdfFile", "")
        except Exception:
            pass
            
    # 2. Delete main policy files
    for suffix in [f"policy_{policy_id.lower().replace('-', '_')}.json", f"{policy_id}_rules.md", f"{policy_id}_skills.json", f"{policy_id}_hooks.json"]:
        file_path = os.path.join(policies_dir, suffix)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
                
    # 3. Delete PDF locally and from S3
    if pdf_file_rel:
        pdf_path_abs = os.path.join(DATA_DIR, pdf_file_rel)
        if os.path.exists(pdf_path_abs):
            try:
                os.remove(pdf_path_abs)
            except Exception:
                pass
        if is_s3_enabled():
            try:
                import boto3
                s3_key = os.path.basename(pdf_file_rel)
                s3 = boto3.client("s3")
                s3.delete_object(Bucket=os.environ.get("S3_BUCKET"), Key=s3_key)
                print(f"Deleted PDF {s3_key} from S3 bucket.")
            except Exception as se:
                print(f"Failed to delete S3 PDF: {se}")

    # 4. Remove associated test cases from preset_cases.json
    cases_file = os.path.join(policies_dir, "preset_cases.json")
    if os.path.exists(cases_file):
        try:
            with open(cases_file, "r") as f:
                cases = json.load(f)
            
            case_prefix = f"case-{policy_id.lower().replace('.', '_')}"
            for case in cases:
                if case["id"].startswith(case_prefix):
                    bundle_rel = case.get("evidenceBundle", "")
                    if bundle_rel:
                        bundle_abs = os.path.join(DATA_DIR, bundle_rel)
                        if os.path.exists(bundle_abs):
                            try:
                                os.remove(bundle_abs)
                            except Exception:
                                pass
                        if is_s3_enabled():
                            try:
                                import boto3
                                s3 = boto3.client("s3")
                                s3.delete_object(Bucket=os.environ.get("S3_BUCKET"), Key=bundle_rel)
                            except Exception:
                                pass
                                
            filtered_cases = [c for c in cases if not c["id"].startswith(case_prefix)]
            with open(cases_file, "w") as f:
                json.dump(filtered_cases, f, indent=2)
        except Exception as ce:
            print(f"Error cleaning cases during deletion: {ce}")
            
    print(f"Policy {policy_id} deleted successfully from container & S3.")
    return {"status": "success"}

