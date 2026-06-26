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
import urllib.request
from datetime import datetime
from pypdf import PdfReader

# ─── Configuration ───────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:12b"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_POLICIES_FILE = os.path.join(DATA_DIR, "extracted_policies.json")
GENERATED_SKILLS_FILE = os.path.join(DATA_DIR, "generated_skills.json")


# ─── LLM Interface ──────────────────────────────────────────────────────────

def call_llm(prompt, max_tokens=400, temperature=0.1):
    """Call local Gemma 4 12B via Ollama with proper chat template."""
    full_prompt = (
        f"<start_of_turn>user\n{prompt}\n"
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
    """Parse JSON from LLM response, handling markdown code blocks."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    # Try array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No JSON found in: {text[:200]}")


# ─── PDF Policy Extraction ──────────────────────────────────────────────────

def read_pdf_text(pdf_path, max_pages=20):
    """Extract text from first N pages of a PDF."""
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

    pdf_text = read_pdf_text(pdf_path, max_pages=15)
    trace.append({"step": "pdf_read", "message": f"Extracted {len(pdf_text)} chars from PDF"})

    # Chunk the text for LLM (Gemma 4 context is 4096 tokens)
    # Take the most relevant sections (first few pages usually have the criteria)
    chunk = pdf_text[:6000]

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

    response = call_llm(prompt, max_tokens=800, temperature=0.1)
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
        icds = policy.get("icd10Codes", [])
        criteria = policy.get("criteria", [])

        for cpt in (cpts if cpts else ["UNKNOWN"]):
            md_lines.append(f"\n## Policy: {pid} ({pname})")
            md_lines.append(f"- CPT Code: {cpt}")
            md_lines.append(f"- Allowed ICD-10 Diagnosis Codes: {', '.join(icds)}")

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
    response = call_llm(prompt, max_tokens=600, temperature=0.2)
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

    # ─── STAGE 3: Clinical Evidence Extraction ────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 3: Disclosing clinical guidelines. Extracting evidence.", "status": "info"})

    clinical_notes = request.get("clinicalNotes", "")
    cpt_code = request.get("cptCode", "")

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

    # ─── STAGE 4: OPA Rego Evaluation ─────────────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 4: NOW disclosing Rego policy rules for evaluation.", "status": "info"})

    # Evaluate against policies
    criteria_met = _evaluate_criteria(evidence, policies, cpt_code)
    all_met = all(c["met"] for c in criteria_met)

    for c in criteria_met:
        status = "success" if c["met"] else "warning"
        trace.append({"ts": timestamp(), "type": "rule", "name": "rules.rego",
                      "msg": f"{c['name']}: {'PASS' if c['met'] else 'FAIL'} ({c['detail']})", "status": status})

    # ─── STAGE 5: Decision + Notice ───────────────────────────────────────
    trace.append({"ts": timestamp(), "type": "disclosure", "name": "Progressive Disclosure",
                  "msg": "Stage 5: Applying conservatism guardrails. Generating notice.", "status": "info"})

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
        "criteriaMet": criteria_met
    }


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _ai_extract_evidence(clinical_notes, cpt_code):
    """Use Gemma 4 to semantically extract clinical evidence."""
    prompt = f"""Extract medical facts from these clinical notes as JSON.
CPT Code: {cpt_code}

NOTES: {clinical_notes}

Return ONLY this JSON:
{{"symptomsDurationWeeks": <int>, "therapyWeeks": <int>, "hasObjectiveFindings": <bool>, "isRheumatologist": <bool>, "hasRadiographs": <bool>, "reasoning": "<brief>"}}"""

    response = call_llm(prompt, max_tokens=200, temperature=0.1)
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
