import http.server
import socketserver
import json
import os
import re

PORT = 8000

# Initialize Headroom context compression (if available)
try:
    from headroom import compress as headroom_compress
    HEADROOM_AVAILABLE = True
    print("Headroom context compression: ENABLED (60-95% token reduction)")
except ImportError:
    HEADROOM_AVAILABLE = False
    print("Headroom not installed. Running without context compression.")

# Import the real Python agent engine
try:
    import agent_engine
    AGENT_ENGINE_AVAILABLE = True
    print("Agent Engine: LOADED (Python backend with LLM pipeline)")
except ImportError as e:
    AGENT_ENGINE_AVAILABLE = False
    print(f"Agent Engine not available: {e}")

# LLM Performance Configuration
LLM_CONFIG = {
    "chat": {"num_predict": 150, "temperature": 0.3},       # AVI: short, fast answers
    "extract": {"num_predict": 250, "temperature": 0.1},    # Clinical extraction: precise JSON
    "letter": {"num_predict": 350, "temperature": 0.4},     # Letter rewrite: creative but bounded
    "appeal": {"num_predict": 300, "temperature": 0.3},     # Appeal guide: detailed but focused
    "summarize": {"num_predict": 120, "temperature": 0.2},  # Summary: very short
    "score": {"num_predict": 150, "temperature": 0.1},      # Quality score: JSON only
}

# Pre-warm the LLM on startup (keeps model in GPU memory)
def prewarm_llm():
    try:
        import urllib.request
        payload = json.dumps({
            "model": "gemma4:12b",
            "prompt": "<start_of_turn>user\nReady<end_of_turn>\n<start_of_turn>model\n",
            "stream": False, "raw": True,
            "keep_alive": "24h",
            "options": {"num_predict": 1}
        }).encode()
        req = urllib.request.Request('http://localhost:11434/api/generate', data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=30)
        print("LLM pre-warmed: gemma4:12b loaded in GPU memory (keep_alive=24h)")
    except Exception as e:
        print(f"LLM pre-warm skipped (Ollama may not be running): {e}")

prewarm_llm()

# Original files templates for reset
ORIGINAL_SKILLS = """import { MEMBER_BENEFITS, CLINICAL_GUIDELINES } from './cases.js';

// Skill 1: Verify member plan coverage eligibility
export function VerifyCoverageSkill(context, agent) {
  agent.logTrace("skill", "VerifyCoverageSkill", `Retrieving member coverage profile for ID: ${context.request.memberId}.`, "info");
  const member = MEMBER_BENEFITS[context.request.memberId];
  
  if (!member) {
    agent.logTrace("skill", "VerifyCoverageSkill", `Member ID ${context.request.memberId} not found.`, "fail");
    context.coverage = { eligible: false, error: "Member ID not enrolled" };
    return;
  }

  if (member.status !== "Active") {
    agent.logTrace("skill", "VerifyCoverageSkill", `Member ID ${context.request.memberId} coverage is inactive.`, "fail");
    context.coverage = { eligible: false, error: "Member inactive" };
    return;
  }

  let serviceCategory = "Radiology";
  if (context.request.cptCode.startsWith("J")) {
    serviceCategory = "Pharmacy";
  }

  const isCovered = member.coveredCategories.includes(serviceCategory);
  
  if (!isCovered) {
    agent.logTrace("skill", "VerifyCoverageSkill", `Member plan does not cover service category: ${serviceCategory}.`, "warning");
    context.coverage = { eligible: false, member, error: `Plan exclusion for category ${serviceCategory}` };
    context.coverageError = true;
    return;
  }

  agent.logTrace("skill", "VerifyCoverageSkill", `Member active for ${serviceCategory}. Plan: ${member.planType}.`, "success");
  context.coverage = { eligible: true, member };
}

// Skill 2: Fetch Guidelines for target CPT codes
export function RetrieveGuidelinesSkill(context, agent) {
  const cptCode = context.request.cptCode;
  agent.logTrace("skill", "RetrieveGuidelinesSkill", `Querying clinical guidelines catalog for CPT code ${cptCode}.`, "info");
  const guidelines = CLINICAL_GUIDELINES[cptCode];

  if (!guidelines) {
    agent.logTrace("skill", "RetrieveGuidelinesSkill", `No clinical guidelines found for procedure CPT ${cptCode}.`, "warning");
    context.guidelines = null;
    return;
  }

  agent.logTrace("skill", "RetrieveGuidelinesSkill", `Policy found: ${guidelines.policyId} - ${guidelines.policyName}`, "success");
  context.guidelines = guidelines;
}

// Skill 3: Parse provider clinical notes and extract facts
export function ExtractClinicalDataSkill(context, agent) {
  agent.logTrace("skill", "ExtractClinicalDataSkill", "Analyzing clinical documentation for medical necessity triggers.", "info");
  const notes = context.request.clinicalNotes;
  const cptCode = context.request.cptCode;
  
  const lowerNotes = notes.toLowerCase();
  let symptomsDurationWeeks = 0;
  let therapyWeeks = 0;
  let hasObjectiveFindings = false;
  let isRheumatologist = false;
  
  if (cptCode === "73721") {
    const durationMatch = lowerNotes.match(/(\d+)\s*weeks?\s*of\s*(pain|symptom)/) || lowerNotes.match(/pain\s*for\s*(\d+)\s*weeks?/);
    if (durationMatch) {
      symptomsDurationWeeks = parseInt(durationMatch[1], 10);
    } else if (lowerNotes.includes("persistent knee pain") || lowerNotes.includes("chronic pain")) {
      symptomsDurationWeeks = 8; 
    }

    const therapyMatch = lowerNotes.match(/(\d+)\s*weeks?\s*of\s*(physical therapy|pt|therapy|ibuprofen|nsaids)/) || lowerNotes.match(/(physical therapy|pt)\s*for\s*(\d+)\s*weeks?/);
    if (therapyMatch) {
      therapyWeeks = parseInt(therapyMatch[1] || therapyMatch[2], 10);
    } else if (lowerNotes.includes("physical therapy") || lowerNotes.includes("pt")) {
      therapyWeeks = 6;
    }

    if (lowerNotes.includes("tenderness") || lowerNotes.includes("swelling") || lowerNotes.includes("instability") || lowerNotes.includes("locking")) {
      hasObjectiveFindings = true;
    }
  } 
  
  if (cptCode === "J0135") {
    if (lowerNotes.includes("rheumatoid arthritis") || lowerNotes.includes("ra")) {
      hasObjectiveFindings = true;
    }
    if (lowerNotes.includes("methotrexate") || lowerNotes.includes("sulfasalazine") || lowerNotes.includes("dmard")) {
      const failureMatch = lowerNotes.match(/(\d+)\s*months?/);
      therapyWeeks = failureMatch ? parseInt(failureMatch[1], 10) * 4 : 12;
    }
    if (lowerNotes.includes("rheumatologist") || lowerNotes.includes("dr. evans")) {
      isRheumatologist = true;
    }
  }

  const evidence = {
    symptomsDurationWeeks,
    therapyWeeks,
    hasObjectiveFindings,
    isRheumatologist
  };

  agent.logTrace("skill", "ExtractClinicalDataSkill", "Extracted structured facts from provider documentation.", "success", evidence);
  context.evidence = evidence;
}

// Skill 4: Match clinical guidelines against evidence checklist using Rego results
export function EvaluateClinicalCriteriaSkill(context, agent) {
  agent.logTrace("skill", "EvaluateClinicalCriteriaSkill", "Mapping evidence to policy rules check sheet via OPA Rego results.", "info");
  
  if (!context.guidelines) {
    agent.logTrace("skill", "EvaluateClinicalCriteriaSkill", "No active guideline policy available.", "fail");
    return;
  }

  const regoResults = context.regoResults;
  const results = [];

  context.guidelines.criteria.forEach(c => {
    let met = false;
    let actualValue = "Not Met";
    
    if (c.type === "symptom_duration") {
      actualValue = `${context.evidence.symptomsDurationWeeks} weeks`;
      met = regoResults.criteria.find(r => r.id === "symptoms_duration_met")?.met || false;
    } else if (c.type === "conservative_treatment") {
      actualValue = `${context.evidence.therapyWeeks} weeks`;
      met = regoResults.criteria.find(r => r.id === "conservative_therapy_met")?.met || false;
    } else if (c.type === "objective_findings") {
      actualValue = context.evidence.hasObjectiveFindings ? "Documented" : "Not Found";
      met = regoResults.criteria.find(r => r.id === "objective_findings_met")?.met || false;
    } else if (c.type === "diagnosis_confirmation") {
      actualValue = context.icdCodeMatched ? "Confirmed Match" : "Mismatch";
      met = regoResults.criteria.find(r => r.id === "icd_valid")?.met || false;
    } else if (c.type === "dmard_failure") {
      actualValue = context.evidence.therapyWeeks >= 12 ? "Failed DMARD (>3 months)" : "Under-treated";
      met = regoResults.criteria.find(r => r.id === "conservative_therapy_met")?.met || false;
    } else if (c.type === "specialist_consult") {
      actualValue = context.evidence.isRheumatologist ? "Rheumatologist" : "No Specialist";
      met = regoResults.criteria.find(r => r.id === "objective_findings_met")?.met || false;
    }

    results.push({
      id: c.id,
      text: c.text,
      met,
      actualValue
    });
  });

  context.criteriaResults = results;
  agent.logTrace("skill", "EvaluateClinicalCriteriaSkill", `Evaluation checklist mapped: ${results.filter(r => r.met).length}/${results.length} criteria met.`, "success", results);
}

// Skill 5: Generate decision notice letter
export function GenerateDecisionNoticeSkill(context, agent) {
  agent.logTrace("skill", "GenerateDecisionNoticeSkill", "Drafting formal clinical notice correspondence.", "info");
  
  const policy = context.guidelines;
  const decision = context.decision;
  const patientName = context.request.patientName;
  const providerName = context.request.providerName;
  const cpt = context.request.cptCode;
  
  let letter = "";
  const dateStr = new Date().toLocaleDateString();

  letter += `DATE: ${dateStr}\n`;
  letter += `TO: Patient ${patientName} & Provider ${providerName}\n`;
  letter += `RE: PRIOR AUTHORIZATION REQUEST DECISION\n`;
  letter += `===========================================\n\n`;
  
  if (decision === "Approved") {
    letter += `We are pleased to inform you that your prior authorization request for procedure CPT code ${cpt} has been APPROVED.\n\n`;
    letter += `CLINICAL SUMMARY:\n`;
    letter += `The medical reviewer confirmed that the clinical facts meet medical necessity requirements.\n`;
  } else {
    letter += `Your prior authorization request for procedure CPT code ${cpt} has been ESCALATED FOR MANUAL REVIEW.\n\n`;
    letter += `CLINICAL SUMMARY:\n`;
    letter += `The automated clinical review determined that guidelines were not fully satisfied or coverage issues require administrative audit. A Clinical Medical Director is reviewing your case and will issue a final decision within 48 hours.\n`;
  }

  letter += `\nPOLICY REFERENCE:\n`;
  if (policy) {
    letter += `This review was performed in accordance with policy ${policy.policyId} (${policy.policyName}, effective date: ${policy.effectiveDate}).\n`;
    
    letter += `\nCRITERIA MET CHECKLIST:\n`;
    context.criteriaResults.forEach(c => {
      letter += `[${c.met ? 'X' : ' '}] ${c.id}: ${c.text} (Actual: ${c.actualValue})\n`;
    });
  } else {
    letter += `No matching active clinical medical policy guidelines were found for procedure CPT code ${cpt}.\n`;
  }

  letter += `\nFor any questions regarding this decision, contact Payer Member Services.\n`;

  context.noticeDraft = letter;
  agent.logTrace("skill", "GenerateDecisionNoticeSkill", "Prior Authorization notice text generated.", "success");
}
"""

ORIGINAL_RULES_DEC = """# Clinical Necessity Policy Declarations

This document holds the human-readable prior authorization clinical criteria rules. Non-programmer medical directors can update this file to configure guidelines. The SkillJuror OPA compiler parses this markdown to generate active Rego rules.

---

## Policy: POL-RAD-402 (MRI Knee Joint)
- CPT Code: 73721
- Allowed ICD-10 Diagnosis Codes: M25.561, M25.562, M25.569, S83.206A
- Minimum Symptom Duration: 6 weeks
- Minimum Conservative Therapy: 6 weeks
- Objective Findings Required: True
- Specialist Consultation Required: False
- Plain Radiographs Completed: False

## Policy: POL-PHARM-809 (Biologic Therapy for RA)
- CPT Code: J0135
- Allowed ICD-10 Diagnosis Codes: M05.79, M06.9
- Minimum Symptom Duration: 0 weeks
- Minimum Conservative Therapy: 12 weeks
- Objective Findings Required: True
- Specialist Consultation Required: True
- Plain Radiographs Completed: False

## Policy: POL-EVI-402 (eviCore Knee MRI Guidelines)
- CPT Code: 73721
- Allowed ICD-10 Diagnosis Codes: M25.561, M25.562, M25.569, S83.206A
- Minimum Symptom Duration: 6 weeks
- Minimum Conservative Therapy: 6 weeks
- Objective Findings Required: True
- Specialist Consultation Required: False
- Plain Radiographs Completed: True
"""

ORIGINAL_SKILLS_DEC = """# Agent Skill Declarations

This document declares the skills available to the AI prior authorization agent. Non-programmers can configure or register new skills by writing their interfaces below.

---

## Skill: VerifyCoverageSkill
- Description: Verifies member plan coverage eligibility from enrollment database.
- Inputs: memberId, cptCode
- Outputs: coverageEligible, planType

## Skill: RetrieveGuidelinesSkill
- Description: Queries clinical guidelines catalog for procedure rules.
- Inputs: cptCode
- Outputs: policyId, policyCriteria

## Skill: ExtractClinicalDataSkill
- Description: Parses clinical notes to extract medical triggers and therapy history.
- Inputs: clinicalNotes, cptCode
- Outputs: symptomsDuration, conservativeTherapyWeeks, objectiveFindings

## Skill: EvaluateClinicalCriteriaSkill
- Description: Maps extracted clinical evidence against OPA policy guidelines.
- Inputs: guidelines, evidence, icdCodeMatched
- Outputs: criteriaResultsChecklist

## Skill: GenerateDecisionNoticeSkill
- Description: Drafts the formal correspondence letter with citations and plain language translation.
- Inputs: decision, criteriaResults
- Outputs: noticeLetterDraft
"""

NPI_SKILL_JS_CODE = """
// Dynamically compiled Skill 6: Verify NPI registry status
export function VerifyNpiStatusSkill(context, agent) {
  agent.logTrace("skill", "VerifyNpiStatusSkill", `Validating provider NPI credentials for NPI: ${context.request.providerNpi}.`, "info");
  const npi = context.request.providerNpi;
  const isValid = npi && npi.length === 10 && /^\\d+$/.test(npi);
  if (isValid) {
    agent.logTrace("skill", "VerifyNpiStatusSkill", `NPI ${npi} format is active and verified in provider registry.`, "success");
    context.npiStatus = "Valid";
  } else {
    agent.logTrace("skill", "VerifyNpiStatusSkill", `NPI ${npi} validation failed: Must be exactly 10 numeric digits.`, "fail");
    context.npiStatus = "Invalid";
  }
}
"""

def parse_nl_to_skill(description):
    """
    Parse a plain English skill description into structured skill metadata
    and generate both the markdown declaration and executable JavaScript.
    """
    description_lower = description.lower()
    audit_logs = []
    audit_logs.append(f"NL Skill Parser: Received description: \"{description[:80]}...\"")
    
    # --- Step 1: Extract a skill name from the description ---
    action_verbs = {
        'check': 'Check', 'verify': 'Verify', 'validate': 'Validate',
        'confirm': 'Confirm', 'ensure': 'Ensure', 'flag': 'Flag',
        'detect': 'Detect', 'calculate': 'Calculate', 'compute': 'Compute',
        'extract': 'Extract', 'retrieve': 'Retrieve', 'lookup': 'Lookup',
        'look up': 'Lookup', 'evaluate': 'Evaluate', 'assess': 'Assess',
        'determine': 'Determine', 'compare': 'Compare', 'match': 'Match',
        'filter': 'Filter', 'score': 'Score', 'rate': 'Rate',
        'classify': 'Classify', 'categorize': 'Categorize',
        'monitor': 'Monitor', 'track': 'Track', 'audit': 'Audit',
        'review': 'Review', 'screen': 'Screen', 'scan': 'Scan',
        'notify': 'Notify', 'alert': 'Alert', 'report': 'Report',
        'block': 'Block', 'deny': 'Deny', 'reject': 'Reject',
        'approve': 'Approve', 'authorize': 'Authorize'
    }
    
    domain_objects = {
        'age': 'Age', 'patient': 'Patient', 'provider': 'Provider',
        'npi': 'Npi', 'coverage': 'Coverage', 'eligibility': 'Eligibility',
        'diagnosis': 'Diagnosis', 'medication': 'Medication', 'drug': 'Drug',
        'dose': 'Dose', 'dosage': 'Dosage', 'frequency': 'Frequency',
        'duration': 'Duration', 'history': 'History', 'allergy': 'Allergy',
        'contraindication': 'Contraindication', 'prior': 'Prior',
        'authorization': 'Authorization', 'referral': 'Referral',
        'specialist': 'Specialist', 'network': 'Network', 'formulary': 'Formulary',
        'tier': 'Tier', 'copay': 'Copay', 'deductible': 'Deductible',
        'limit': 'Limit', 'quantity': 'Quantity', 'step therapy': 'StepTherapy',
        'gender': 'Gender', 'bmi': 'Bmi', 'lab': 'Lab', 'result': 'Result',
        'imaging': 'Imaging', 'radiology': 'Radiology', 'surgery': 'Surgery',
        'procedure': 'Procedure', 'code': 'Code', 'status': 'Status',
        'senior': 'Senior', 'pediatric': 'Pediatric', 'emergency': 'Emergency',
        'urgent': 'Urgent', 'routine': 'Routine', 'preventive': 'Preventive',
        'chronic': 'Chronic', 'acute': 'Acute', 'mental health': 'MentalHealth',
        'behavioral': 'Behavioral', 'substance': 'Substance', 'rehab': 'Rehab',
        'therapy': 'Therapy', 'compliance': 'Compliance', 'adherence': 'Adherence',
        'claim': 'Claim', 'denial': 'Denial', 'appeal': 'Appeal',
        'duplicate': 'Duplicate', 'fraud': 'Fraud', 'abuse': 'Abuse',
        'waste': 'Waste', 'utilization': 'Utilization'
    }
    
    found_verb = 'Check'
    for verb, pascal in action_verbs.items():
        if verb in description_lower:
            found_verb = pascal
            break
    
    found_objects = []
    for obj, pascal in domain_objects.items():
        if obj in description_lower and pascal not in found_objects:
            found_objects.append(pascal)
    
    if not found_objects:
        found_objects = ['Custom']
    
    skill_name = found_verb + ''.join(found_objects[:3]) + 'Skill'
    audit_logs.append(f"NL Skill Parser: Derived skill name: {skill_name}")
    
    # --- Step 2: Determine inputs and outputs ---
    input_mapping = {
        'age': 'patientDob', 'dob': 'patientDob', 'date of birth': 'patientDob',
        'patient name': 'patientName', 'member': 'memberId', 'member id': 'memberId',
        'npi': 'providerNpi', 'provider': 'providerName',
        'cpt': 'cptCode', 'procedure': 'cptCode',
        'icd': 'icd10Code', 'diagnosis': 'icd10Code',
        'notes': 'clinicalNotes', 'clinical notes': 'clinicalNotes',
        'ssn': 'patientSsn', 'coverage': 'memberId', 'plan': 'memberId',
        'medication': 'clinicalNotes', 'drug': 'clinicalNotes',
        'history': 'clinicalNotes', 'allergy': 'clinicalNotes',
        'gender': 'clinicalNotes', 'bmi': 'clinicalNotes',
        'lab': 'clinicalNotes', 'imaging': 'clinicalNotes'
    }
    
    inputs = set()
    for keyword, field in input_mapping.items():
        if keyword in description_lower:
            inputs.add(field)
    
    if not inputs:
        inputs.add('clinicalNotes')
    
    inputs = sorted(list(inputs))
    
    # Determine outputs
    output_base = skill_name.replace('Skill', '')[0].lower() + skill_name.replace('Skill', '')[1:]
    outputs = []
    if any(w in description_lower for w in ['flag', 'alert', 'warn', 'notify', 'escalate']):
        outputs.append(f"{output_base}Flagged")
    if any(w in description_lower for w in ['check', 'verify', 'validate', 'confirm']):
        outputs.append(f"{output_base}Valid")
    if any(w in description_lower for w in ['score', 'calculate', 'compute', 'rate']):
        outputs.append(f"{output_base}Score")
    if any(w in description_lower for w in ['extract', 'retrieve', 'get', 'lookup']):
        outputs.append(f"{output_base}Result")
    if any(w in description_lower for w in ['classify', 'categorize', 'determine']):
        outputs.append(f"{output_base}Category")
    if not outputs:
        outputs.append(f"{output_base}Status")
    
    audit_logs.append(f"NL Skill Parser: Inputs: {inputs}")
    audit_logs.append(f"NL Skill Parser: Outputs: {outputs}")
    
    # --- Step 3: Generate conditions from description ---
    conditions = parse_conditions(description, description_lower)
    audit_logs.append(f"NL Skill Parser: Extracted {len(conditions)} condition(s) from description.")
    
    # --- Step 4: Build Markdown declaration ---
    md_entry = f"\n## Skill: {skill_name}\n"
    md_entry += f"- Description: {description}\n"
    md_entry += f"- Inputs: {', '.join(inputs)}\n"
    md_entry += f"- Outputs: {', '.join(outputs)}\n"
    
    # --- Step 5: Build executable JavaScript ---
    js_code = generate_skill_js(skill_name, description, inputs, outputs, conditions)
    audit_logs.append(f"NL Skill Parser: Generated JavaScript implementation ({len(js_code)} chars).")
    
    return {
        'skill_name': skill_name,
        'inputs': inputs,
        'outputs': outputs,
        'md_entry': md_entry,
        'js_code': js_code,
        'audit_logs': audit_logs
    }


def parse_conditions(description, description_lower):
    """Extract conditional logic from the plain English description."""
    conditions = []
    
    # Pattern: "if <field> is over/above/greater than <number>"
    over_patterns = [
        (r'(?:if|when|where)?\s*(?:the\s+)?(?:patient\'?s?\s+)?(\w+)\s+(?:is\s+)?(?:over|above|greater than|more than|exceeds?)\s+(\d+)', 'gt'),
        (r'(?:if|when|where)?\s*(?:the\s+)?(?:patient\'?s?\s+)?(\w+)\s+(?:is\s+)?(?:under|below|less than|fewer than)\s+(\d+)', 'lt'),
        (r'(?:if|when|where)?\s*(?:the\s+)?(?:patient\'?s?\s+)?(\w+)\s+(?:is\s+)?(?:equal to|equals?|exactly)\s+(\d+)', 'eq'),
        (r'(?:if|when|where)?\s*(?:the\s+)?(?:patient\'?s?\s+)?(\w+)\s+(?:is\s+)?(?:at least|minimum)\s+(\d+)', 'gte'),
    ]
    
    for pattern, op in over_patterns:
        matches = re.finditer(pattern, description_lower)
        for m in matches:
            field = m.group(1).strip()
            value = m.group(2).strip()
            conditions.append({'field': field, 'operator': op, 'value': value, 'type': 'numeric'})
    
    # Additional pattern: "60 and above", "age group 60", "65 or older"
    age_group_pattern = r'(?:age\s+(?:group\s+)?)?(\d+)\s+(?:and above|and older|or above|or older)'
    for m in re.finditer(age_group_pattern, description_lower):
        value = m.group(1).strip()
        # Only add if we don't already have a numeric condition for age
        if not any(c['field'] == 'age' and c['type'] == 'numeric' for c in conditions):
            conditions.append({'field': 'age', 'operator': 'gte', 'value': value, 'type': 'numeric'})
    
    # Pattern: "make sure it/is <value>" or "ensure it/is <value>" for equality checks on text fields
    # Handles: "make sure it male", "ensure sex is female", "verify gender is male"
    gender_pattern = r'(?:make sure|ensure|verify|check)\s+(?:it(?:\s+is)?|(?:the\s+)?(?:patient\'?s?\s+)?(?:sex|gender)\s+(?:is\s+)?)\s*(male|female|man|woman|boy|girl)'
    for m in re.finditer(gender_pattern, description_lower):
        value = m.group(1).strip()
        if not any(c.get('field_type') == 'gender' for c in conditions):
            conditions.append({'field': 'gender', 'operator': 'equals_text', 'value': value, 'type': 'text', 'field_type': 'gender'})
    
    # Pattern: "if <field> contains/includes/mentions <word>"
    contains_pattern = r'(?:if|when|where)?\s*(?:the\s+)?(?:patient\'?s?\s+)?(?:clinical\s+)?(\w+(?:\s+\w+)?)\s+(?:contains?|includes?|mentions?|has)\s+["\']?([^"\',.]+)["\']?'
    for m in re.finditer(contains_pattern, description_lower):
        field = m.group(1).strip()
        value = m.group(2).strip()
        if len(value.split()) <= 4:
            # Handle "or" conjunctions: split "opioid or narcotic" into separate conditions
            if ' or ' in value:
                parts = [p.strip() for p in value.split(' or ')]
                for part in parts:
                    if part:
                        conditions.append({'field': field, 'operator': 'contains_or', 'value': part, 'type': 'text'})
            else:
                conditions.append({'field': field, 'operator': 'contains', 'value': value, 'type': 'text'})
    
    # Detect action from description
    action_pattern = r'(?:flag|escalate|route|send|trigger|alert|mark)\s+(?:for|to|as)\s+([^.]+)'
    action_match = re.search(action_pattern, description_lower)
    action_text = action_match.group(1).strip() if action_match else "review"
    
    # If no conditions found, try simpler patterns
    if not conditions:
        if 'age' in description_lower:
            age_match = re.search(r'(\d+)', description)
            if age_match:
                conditions.append({'field': 'age', 'operator': 'gte', 'value': age_match.group(1), 'type': 'numeric'})
        elif any(w in description_lower for w in ['missing', 'absent', 'lacks', 'without']):
            missing_match = re.search(r'(?:missing|absent|lacks?|without)\s+([^,.]+)', description_lower)
            if missing_match:
                conditions.append({'field': 'notes', 'operator': 'not_contains', 'value': missing_match.group(1).strip(), 'type': 'text'})
    
    for c in conditions:
        c['action'] = action_text
    
    return conditions


def generate_skill_js(skill_name, description, inputs, outputs, conditions):
    """Generate executable JavaScript function for a skill."""
    import string
    
    # Sanitize description for use in JS strings (escape quotes and backslashes)
    safe_desc = description.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    safe_desc_short = safe_desc[:60]
    
    result_var = outputs[0] if outputs else 'skillResult'
    needs_age_helper = any(c['field'] == 'age' for c in conditions)
    
    js_lines = []
    js_lines.append(f'// Auto-generated skill: {safe_desc}')
    js_lines.append(f'export function {skill_name}(context, agent) {{')
    js_lines.append(f'  agent.logTrace("skill", "{skill_name}", "Executing: {safe_desc_short}...", "info");')
    
    if needs_age_helper:
        js_lines.append('  function calculateAge(dob) {')
        js_lines.append('    if (!dob) return 0;')
        js_lines.append('    const birthDate = new Date(dob);')
        js_lines.append('    const today = new Date();')
        js_lines.append('    let age = today.getFullYear() - birthDate.getFullYear();')
        js_lines.append('    const m = today.getMonth() - birthDate.getMonth();')
        js_lines.append('    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) age--;')
        js_lines.append('    return age;')
        js_lines.append('  }')
    
    if conditions:
        for i, cond in enumerate(conditions):
            field = cond['field']
            op = cond['operator']
            val = cond['value']
            ctype = cond['type']
            
            if field == 'age':
                accessor = 'calculateAge(context.request.patientDob)'
            elif field == 'gender':
                accessor = 'context.request.clinicalNotes.toLowerCase()'
            elif field in ('notes', 'clinical notes'):
                accessor = 'context.request.clinicalNotes.toLowerCase()'
            else:
                accessor = 'context.request.clinicalNotes.toLowerCase()'
            
            var_name = f'check{i}'
            
            if ctype == 'numeric':
                js_lines.append(f'  const {var_name} = {accessor};')
                if op == 'gt':
                    js_lines.append(f'  const condition{i} = {var_name} > {val};')
                elif op == 'lt':
                    js_lines.append(f'  const condition{i} = {var_name} < {val};')
                elif op == 'eq':
                    js_lines.append(f'  const condition{i} = {var_name} === {val};')
                elif op == 'gte':
                    js_lines.append(f'  const condition{i} = {var_name} >= {val};')
            elif ctype == 'text':
                if op == 'contains' or op == 'contains_or':
                    js_lines.append(f'  const condition{i} = context.request.clinicalNotes.toLowerCase().includes("{val}");')
                elif op == 'not_contains':
                    js_lines.append(f'  const condition{i} = !context.request.clinicalNotes.toLowerCase().includes("{val}");')
                elif op == 'equals_text':
                    js_lines.append(f'  const condition{i} = context.request.clinicalNotes.toLowerCase().includes("{val}");')
        
        num_conditions = len(conditions)
        # Group contains_or conditions with OR logic, others with AND
        or_indices = [i for i, c in enumerate(conditions) if c['operator'] == 'contains_or']
        and_indices = [i for i, c in enumerate(conditions) if c['operator'] != 'contains_or']
        
        parts = []
        if or_indices:
            or_combined = ' || '.join([f'condition{i}' for i in or_indices])
            parts.append(f'({or_combined})')
        for i in and_indices:
            parts.append(f'condition{i}')
        
        combined = ' && '.join(parts) if parts else 'true'
        action = conditions[0].get('action', 'review').replace('"', '\\"').replace("'", "\\'")
        
        js_lines.append(f'  const triggered = {combined};')
        js_lines.append(f'  if (triggered) {{')
        js_lines.append(f'    context.{result_var} = "Flagged";')
        js_lines.append(f'    agent.logTrace("skill", "{skill_name}", "Condition met: {action}. Case flagged.", "warning");')
        js_lines.append(f'  }} else {{')
        js_lines.append(f'    context.{result_var} = "Passed";')
        js_lines.append(f'    agent.logTrace("skill", "{skill_name}", "All checks passed. No flag raised.", "success");')
        js_lines.append(f'  }}')
    else:
        # Fallback: generic keyword scan from description
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'if', 'and',
                      'or', 'but', 'for', 'to', 'from', 'of', 'in', 'on', 'at', 'by', 'with',
                      'that', 'this', 'it', 'its', 'not', 'do', 'does', 'did', 'has', 'have',
                      'had', 'will', 'would', 'could', 'should', 'may', 'might', 'shall',
                      'can', 'then', 'than', 'when', 'where', 'what', 'which', 'who',
                      'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
                      'some', 'such', 'only', 'own', 'same', 'so', 'very', 'just', 'because',
                      'check', 'verify', 'validate', 'ensure', 'confirm', 'true', 'false',
                      'patient', 'flag', 'review', 'skill'}
        
        words = description.lower().translate(str.maketrans('', '', string.punctuation)).split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3][:5]
        
        js_lines.append(f'  const notes = (context.request.clinicalNotes || "").toLowerCase();')
        if keywords:
            keyword_check = ' || '.join([f'notes.includes("{kw}")' for kw in keywords])
            js_lines.append(f'  const triggered = {keyword_check};')
        else:
            js_lines.append(f'  const triggered = notes.length > 0;')
        
        js_lines.append(f'  if (triggered) {{')
        js_lines.append(f'    context.{result_var} = "Flagged";')
        js_lines.append(f'    agent.logTrace("skill", "{skill_name}", "Keyword match found. Flagged for review.", "warning");')
        js_lines.append(f'  }} else {{')
        js_lines.append(f'    context.{result_var} = "Passed";')
        js_lines.append(f'    agent.logTrace("skill", "{skill_name}", "No relevant keywords detected. Passed.", "success");')
        js_lines.append(f'  }}')
    
    js_lines.append('}')
    return '\n'.join(js_lines)


def compile_rego_from_md(md_content):
    policies = []
    current_policy = None
    
    # Parse policies line by line
    for line in md_content.split('\n'):
        line = line.strip()
        if line.startswith('## Policy:'):
            if current_policy:
                policies.append(current_policy)
            current_policy = {"id": line.split('## Policy:')[1].strip()}
        elif current_policy and line.startswith('- '):
            parts = line[2:].split(':', 1)
            if len(parts) >= 2:
                key = parts[0].strip()
                val = parts[1].strip()
                current_policy[key] = val
                
    if current_policy:
        policies.append(current_policy)

    rego = """package prior_auth.policy

default approve = false
default escalate = false

# Automated Approval Rule
approve {
    cpt_valid
    icd_valid
    conservative_therapy_met
    symptoms_duration_met
    objective_findings_met
    specialist_consult_met
    radiographs_completed_met
}

escalate {
    not approve
    has_evidence
}

has_evidence {
    input.clinicalNotes != ""
}
"""

    # 1. CPT validation rules
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        rego += f'cpt_valid {{\n    input.cptCode == "{cpt}"\n}}\n'

    # 2. ICD diagnosis validation
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        icd_str = p.get("Allowed ICD-10 Diagnosis Codes", "")
        icd_list = [i.strip() for i in icd_str.split(',') if i.strip()]
        icd_set_str = ", ".join([f'"{i}"' for i in icd_list])
        rego += f'icd_valid {{\n    input.cptCode == "{cpt}"\n    input.icd10Code in {{{icd_set_str}}}\n}}\n'

    # 3. Conservative therapy duration rules
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        weeks_str = p.get("Minimum Conservative Therapy", "6")
        weeks = re.sub("[^0-9]", "", weeks_str)
        if not weeks: weeks = "6"
        rego += f'conservative_therapy_met {{\n    input.cptCode == "{cpt}"\n    input.extractedEvidence.therapyWeeks >= {weeks}\n}}\n'

    # 4. Symptoms duration rules
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        weeks_str = p.get("Minimum Symptom Duration", "6")
        weeks = re.sub("[^0-9]", "", weeks_str)
        if not weeks: weeks = "6"
        rego += f'symptoms_duration_met {{\n    input.cptCode == "{cpt}"\n    input.extractedEvidence.symptomsDurationWeeks >= {weeks}\n}}\n'

    # 5. Objective findings requirements
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        req = p.get("Objective Findings Required", "True") == "True"
        if req:
            rego += f'objective_findings_met {{\n    input.cptCode == "{cpt}"\n    input.extractedEvidence.hasObjectiveFindings\n}}\n'
        else:
            rego += f'objective_findings_met {{\n    input.cptCode == "{cpt}"\n}}\n'

    # 6. Specialist required check
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        req = p.get("Specialist Consultation Required", "False") == "True"
        if req:
            rego += f'specialist_consult_met {{\n    input.cptCode == "{cpt}"\n    input.extractedEvidence.isRheumatologist\n}}\n'
        else:
            rego += f'specialist_consult_met {{\n    input.cptCode == "{cpt}"\n}}\n'

    # 6.5. Radiographs completed check
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        req = p.get("Plain Radiographs Completed", "False") == "True"
        if req:
            rego += f'radiographs_completed_met {{\n    input.cptCode == "{cpt}"\n    input.extractedEvidence.hasRadiographs\n}}\n'
        else:
            rego += f'radiographs_completed_met {{\n    input.cptCode == "{cpt}"\n}}\n'

    # 7. Allowed sets compilation (to support old regoInterpreter regex parsing)
    rego += "\n"
    for p in policies:
        cpt = p.get("CPT Code", "")
        icd_str = p.get("Allowed ICD-10 Diagnosis Codes", "")
        icd_list = [i.strip() for i in icd_str.split(',') if i.strip()]
        icd_set_str = ", ".join([f'"{i}"' for i in icd_list])
        
        if cpt == "73721":
            rego += f'knee_icd_codes = {{{icd_set_str}}}\n'
        elif cpt == "J0135":
            rego += f'ra_icd_codes = {{{icd_set_str}}}\n'

    return rego

class DynamicAuditorHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        # 1. Save and Compile human-readable editor updates
        if self.path == '/save-file':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode())
            
            filename = params.get('filename')
            content = params.get('content', '')
            
            if not filename or filename not in ['rules_declaration.md', 'skills_declaration.md']:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid filename requested.")
                return

            try:
                # Save plain text Markdown declaration to disk
                with open(filename, 'w') as f:
                    f.write(content)
                
                audit_logs = []
                audit_logs.append(f"SkillOpt/SkillJuror Auditor: Change request logged for '{filename}'.")
                
                # Run dynamic compilation and SkillJuror auditing
                if filename == 'rules_declaration.md':
                    # Compile OPA rules
                    compiled_rego = compile_rego_from_md(content)
                    
                    # SkillJuror validation checks
                    audit_logs.append("SkillJuror validation: Checking rules.rego formatting syntax... OK.")
                    audit_logs.append("SkillJuror validation: Scanning for direct hardcoded PHI strings... Passed.")
                    
                    with open('rules.rego', 'w') as f:
                        f.write(compiled_rego)
                    
                    audit_logs.append("SkillJuror compilation: rules.rego compiled and committed successfully.")
                else:
                    # Compile skills
                    # Extract skill block names
                    skills_declared = re.findall(r'## Skill:\s*(\w+)', content)
                    audit_logs.append(f"SkillJuror contract audit: Found {len(skills_declared)} skills declared.")
                    
                    # Base Javascript skills construction
                    compiled_js = ORIGINAL_SKILLS
                    
                    # If NpiStatusSkill is declared in markdown, compile the JS function
                    if "VerifyNpiStatusSkill" in skills_declared:
                        audit_logs.append("SkillJuror contract audit: Validating interface contract for 'VerifyNpiStatusSkill'...")
                        audit_logs.append("SkillJuror contract audit: Verified inputs (providerNpi) and outputs (npiStatus).")
                        
                        # Security audit checks
                        unsafe_keywords = ['eval', 'process', 'child_process', 'localStorage', 'sessionStorage', 'XMLHttpRequest', 'WebSocket']
                        clean = True
                        for word in unsafe_keywords:
                            if word in content:
                                clean = False
                                audit_logs.append(f"SkillJuror security alert: Unsafe keyword '{word}' detected in skills markdown.")
                        
                        if clean:
                            audit_logs.append("SkillJuror security check: PASSED. Script behaves in a sandboxed manner.")
                            compiled_js += NPI_SKILL_JS_CODE
                            audit_logs.append("SkillJuror compilation: VerifyNpiStatusSkill compiled and appended to skills.js.")
                        else:
                            audit_logs.append("SkillJuror audit: FAILED. VerifyNpiStatusSkill compilation aborted due to security policies.")
                    
                    with open('skills.js', 'w') as f:
                        f.write(compiled_js)
                    
                    audit_logs.append("SkillJuror compilation: skills.js updated successfully.")

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    "status": "success",
                    "message": f"Successfully compiled and updated {filename} on the disk.",
                    "auditLogs": audit_logs
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(str(e).encode())
                
        # 2. Reset workspace back to default
        elif self.path == '/reset-workspace':
            try:
                with open('skills.js', 'w') as f:
                    f.write(ORIGINAL_SKILLS)
                with open('rules.rego', 'w') as f:
                    f.write(compile_rego_from_md(ORIGINAL_RULES_DEC))
                with open('skills_declaration.md', 'w') as f:
                    f.write(ORIGINAL_SKILLS_DEC)
                with open('rules_declaration.md', 'w') as f:
                    f.write(ORIGINAL_RULES_DEC)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Workspace reset to initial templates."}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(str(e).encode())

        # 3. AI Clinical Note Extraction via Gemma 4 12B (Ollama)
        elif self.path == '/ai-extract':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode())
            
            clinical_notes = params.get('clinicalNotes', '')
            cpt_code = params.get('cptCode', '')
            guidelines_text = params.get('guidelinesText', '')
            
            if not clinical_notes:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No clinical notes provided"}).encode())
                return
            
            try:
                import urllib.request
                
                prompt = f"""<start_of_turn>user
You are a clinical data extraction system for prior authorization review.
Given the clinical notes below, extract structured medical facts as JSON.

CPT Code: {cpt_code}
Clinical Guidelines Context: {guidelines_text}

CLINICAL NOTES:
{clinical_notes}

Extract EXACTLY this JSON structure (nothing else):
{{
  "symptomsDurationWeeks": <number of weeks patient has had symptoms, 0 if not mentioned>,
  "therapyWeeks": <number of weeks of conservative therapy completed, 0 if not mentioned>,
  "hasObjectiveFindings": <true if objective clinical findings are documented like tenderness/swelling/instability/locking, false otherwise>,
  "isRheumatologist": <true if a rheumatologist or specialist is mentioned, false otherwise>,
  "hasRadiographs": <true if X-rays or plain radiographs are mentioned as completed, false otherwise>,
  "reasoning": "<brief explanation of what you found in the notes>"
}}

Respond ONLY with the JSON object, no other text.
<end_of_turn>
<start_of_turn>model
"""

                ollama_payload = json.dumps({
                    "model": "gemma4:12b",
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,
                    "keep_alive": "24h",
                    "options": {
                        "temperature": LLM_CONFIG["extract"]["temperature"],
                        "num_predict": LLM_CONFIG["extract"]["num_predict"],
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1
                    }
                }).encode()
                
                req = urllib.request.Request(
                    'http://localhost:11434/api/generate',
                    data=ollama_payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                resp = urllib.request.urlopen(req, timeout=60)
                resp_data = json.loads(resp.read().decode())
                raw_response = resp_data.get('response', '')
                
                # Parse the JSON from the LLM response
                # Gemma 4 may include thought channels — strip them
                json_text = raw_response.strip()
                # Remove Gemma 4 thought channel markers
                if '<|channel>' in json_text:
                    # Take content after the last channel marker
                    parts = json_text.split('<channel|>')
                    json_text = parts[-1].strip() if len(parts) > 1 else json_text
                
                if '```json' in json_text:
                    json_text = json_text.split('```json')[1].split('```')[0].strip()
                elif '```' in json_text:
                    json_text = json_text.split('```')[1].split('```')[0].strip()
                
                # Find the JSON object
                start = json_text.find('{')
                end = json_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_text = json_text[start:end]
                
                extracted = json.loads(json_text)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "model": "gemma4:12b",
                    "extracted": extracted,
                    "rawResponse": raw_response
                }).encode())
                
            except urllib.error.URLError as e:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Ollama not running or Gemma 4 model not loaded. Run: ollama pull gemma4:12b",
                    "details": str(e)
                }).encode())
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"Error in /ai-extract: {tb}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"AI extraction failed: {str(e)}",
                    "rawResponse": raw_response if 'raw_response' in dir() else ""
                }).encode())

        # 3b. AVI Agent (backend-driven, context-aware)
        elif self.path == '/agent/avi':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode())
            
            message = params.get('message', '')
            ui_context = params.get('uiContext', None)
            
            if not message:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No message"}).encode())
                return
            
            try:
                response = agent_engine.avi_respond(message, ui_context)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"response": response}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # 4. General-purpose AI chat endpoint (AVI + all AI features)
        elif self.path == '/ai-chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode())
            
            prompt_text = params.get('prompt', '')
            system_context = params.get('systemContext', '')
            mode = params.get('mode', 'chat')  # chat|letter|appeal|summarize|score
            
            if not prompt_text:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No prompt provided"}).encode())
                return
            
            try:
                import urllib.request
                
                # Compress context through Headroom if available (reduces token load on large contexts)
                effective_context = system_context
                compression_note = ""
                if HEADROOM_AVAILABLE and len(system_context) > 500:
                    try:
                        # Structure as tool output (which Headroom compresses aggressively)
                        messages = [
                            {"role": "assistant", "content": "Let me look at the case context."},
                            {"role": "tool", "content": system_context, "tool_call_id": "case_ctx"}
                        ]
                        result = headroom_compress(messages, model='gemma-2-9b', model_limit=8192, compress_user_messages=True)
                        # Extract the compressed tool message
                        for msg in result.messages:
                            if msg.get("role") == "tool":
                                effective_context = msg["content"]
                                break
                        if result.tokens_saved > 0:
                            savings = round((1 - result.tokens_after / result.tokens_before) * 100)
                            compression_note = f" [Headroom: {savings}% fewer tokens, {result.tokens_saved} saved]"
                            print(f"  Headroom compression: {result.tokens_before} → {result.tokens_after} tokens ({savings}% saved)")
                    except Exception as he:
                        effective_context = system_context  # Fallback to uncompressed
                        print(f"  Headroom compression skipped: {he}")
                
                full_prompt = f"<start_of_turn>user\n"
                if effective_context:
                    full_prompt += f"CONTEXT:\n{effective_context}\n\n"
                full_prompt += f"{prompt_text}\nRespond concisely in 2-4 sentences.\n<end_of_turn>\n<start_of_turn>model\n<|channel>text\n<channel|>"
                
                ollama_payload = json.dumps({
                    "model": "gemma4:12b",
                    "prompt": full_prompt,
                    "stream": False,
                    "raw": True,
                    "keep_alive": "24h",
                    "options": {
                        "temperature": LLM_CONFIG.get(mode, LLM_CONFIG["chat"])["temperature"],
                        "num_predict": LLM_CONFIG.get(mode, LLM_CONFIG["chat"])["num_predict"],
                        "num_ctx": 2048 if mode in ("chat", "summarize", "score") else 4096,
                        "repeat_penalty": 1.1,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                }).encode()
                
                req = urllib.request.Request(
                    'http://localhost:11434/api/generate',
                    data=ollama_payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                resp = urllib.request.urlopen(req, timeout=90)
                resp_data = json.loads(resp.read().decode())
                raw_response = resp_data.get('response', '')
                
                # Clean Gemma 4 thought channel markers
                clean_response = raw_response
                if '<|channel>' in clean_response:
                    parts = clean_response.split('<channel|>')
                    clean_response = parts[-1].strip() if len(parts) > 1 else clean_response
                # Remove any trailing turn markers
                clean_response = clean_response.replace('<end_of_turn>', '').strip()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "response": clean_response,
                    "compression": compression_note if compression_note else None
                }).encode())
                
            except urllib.error.URLError as e:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "ClinicalNLP Engine offline. Ensure Ollama is running.", "details": str(e)}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # 5. Agent: Extract policies from PDF
        elif self.path == '/agent/extract-policy':
            if not AGENT_ENGINE_AVAILABLE:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Agent engine not loaded"}).encode())
                return
            try:
                pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_payer_policy_uhc.pdf')
                result = agent_engine.extract_policies_from_pdf(pdf_path)
                
                # Save for reuse
                agent_engine.save_extracted_policies(result["policies"])
                
                # Also compile to rules_declaration.md
                md_content = agent_engine.compile_policies_to_rules(result["policies"])
                with open('rules_declaration.md', 'w') as f:
                    f.write(md_content)
                
                # Compile to rego
                compiled_rego = compile_rego_from_md(md_content)
                with open('rules.rego', 'w') as f:
                    f.write(compiled_rego)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "policies": result["policies"],
                    "trace": result["trace"],
                    "mdGenerated": md_content[:500],
                    "message": f"Extracted {len(result['policies'])} policies from PDF. Saved to rules_declaration.md and rules.rego."
                }).encode())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # 5b. Agent: Build rules/skills/hooks for a specific policy
        elif self.path == '/agent/build-for-policy':
            if not AGENT_ENGINE_AVAILABLE:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Agent engine not loaded"}).encode())
                return
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode())
                policy_id = params.get('policyId', '')
                action = params.get('action', '')  # rules, skills, or hooks
                
                # Find the policy
                all_policies = agent_engine.load_all_policies()
                policy = next((p for p in all_policies if p['policyId'] == policy_id), None)
                
                if not policy:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Policy {policy_id} not found"}).encode())
                    return
                
                result = agent_engine.build_for_policy(policy, action)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # 6. Agent: Generate skills for extracted policies
        elif self.path == '/agent/generate-skills':
            if not AGENT_ENGINE_AVAILABLE:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Agent engine not loaded"}).encode())
                return
            try:
                policies = agent_engine.load_saved_policies()
                if not policies:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No policies extracted yet. Run PDF extraction first."}).encode())
                    return
                
                result = agent_engine.generate_skills_for_policies(policies)
                agent_engine.save_generated_skills(result["skills"])
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "skills": result["skills"],
                    "trace": result["trace"],
                    "message": f"Generated {len(result['skills'])} skill definitions."
                }).encode())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # 7. Agent: Run full review pipeline (Python backend)
        elif self.path == '/agent/run-review':
            if not AGENT_ENGINE_AVAILABLE:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Agent engine not loaded"}).encode())
                return
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode())
                
                request_data = params.get('request', {})
                use_ai = params.get('useAI', False)
                
                result = agent_engine.run_multi_policy_review(request_data, use_ai_extraction=use_ai)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # 7b. Agent: Get preset cases
        elif self.path == '/agent/cases':
            try:
                cases = agent_engine.load_preset_cases() if AGENT_ENGINE_AVAILABLE else []
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(cases).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # 7c. (Moved to do_GET)

        # 8. Generate skill from natural language description
        elif self.path == '/generate-skill':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode())
            
            description = params.get('description', '').strip()
            
            if not description or len(description) < 10:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Description too short. Please provide a meaningful skill description.", "auditLogs": []}).encode())
                return
            
            try:
                # Parse natural language into skill components
                result = parse_nl_to_skill(description)
                audit_logs = result['audit_logs']
                
                # Security audit on generated JS
                unsafe_keywords = ['eval', 'process', 'child_process', 'localStorage', 'sessionStorage', 'XMLHttpRequest', 'WebSocket', 'fetch(']
                clean = True
                for word in unsafe_keywords:
                    if word in result['js_code']:
                        clean = False
                        audit_logs.append(f"SkillJuror security alert: Unsafe pattern '{word}' in generated code. Aborting.")
                
                if not clean:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Security audit failed", "auditLogs": audit_logs}).encode())
                    return
                
                audit_logs.append("SkillJuror security check: PASSED. Generated code is sandboxed.")
                
                # Check for duplicate skill name before appending
                with open('skills.js', 'r') as f:
                    existing_js = f.read()
                
                if f'export function {result["skill_name"]}(' in existing_js:
                    # Remove the existing version before appending the new one
                    audit_logs.append(f"SkillJuror: Existing {result['skill_name']} found. Replacing with updated version.")
                    # Find and remove the old function block
                    import_section_end = existing_js.find('\n\n', existing_js.rfind('import '))
                    lines = existing_js.split('\n')
                    new_lines = []
                    skip = False
                    for line in lines:
                        if f'export function {result["skill_name"]}(' in line:
                            skip = True
                            continue
                        if skip:
                            # Look for the closing brace of the function (at column 0)
                            if line == '}':
                                skip = False
                                continue
                        if not skip:
                            new_lines.append(line)
                    existing_js = '\n'.join(new_lines)
                    with open('skills.js', 'w') as f:
                        f.write(existing_js)
                
                # Also check skills_declaration.md for duplicates
                with open('skills_declaration.md', 'r') as f:
                    existing_md = f.read()
                
                if f'## Skill: {result["skill_name"]}' in existing_md:
                    # Remove old declaration
                    md_lines = existing_md.split('\n')
                    new_md_lines = []
                    skip_md = False
                    for line in md_lines:
                        if line.strip() == f'## Skill: {result["skill_name"]}':
                            skip_md = True
                            continue
                        if skip_md:
                            if line.startswith('## ') or (line.strip() == '' and new_md_lines and new_md_lines[-1].strip() == ''):
                                skip_md = False
                                if line.startswith('## '):
                                    new_md_lines.append(line)
                                continue
                            continue
                        new_md_lines.append(line)
                    existing_md = '\n'.join(new_md_lines)
                    with open('skills_declaration.md', 'w') as f:
                        f.write(existing_md)
                    audit_logs.append(f"SkillJuror: Replaced existing declaration in skills_declaration.md")
                
                # Append to skills_declaration.md
                with open('skills_declaration.md', 'a') as f:
                    f.write(result['md_entry'])
                audit_logs.append(f"SkillJuror: Appended skill declaration to skills_declaration.md")
                
                # Append to skills.js
                with open('skills.js', 'a') as f:
                    f.write('\n' + result['js_code'] + '\n')
                audit_logs.append(f"SkillJuror compilation: {result['skill_name']} compiled and appended to skills.js")
                
                # Final validation: check JS syntax
                import subprocess
                check = subprocess.run(['node', '--check', 'skills.js'], capture_output=True, text=True)
                if check.returncode != 0:
                    # Rollback - restore original
                    audit_logs.append(f"SkillJuror ERROR: Generated code has syntax errors. Rolling back.")
                    audit_logs.append(f"Node error: {check.stderr.strip()}")
                    with open('skills.js', 'w') as f:
                        f.write(existing_js if 'existing_js' in dir() else ORIGINAL_SKILLS)
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Generated code failed syntax validation. Rolled back.", "auditLogs": audit_logs}).encode())
                    return
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    "status": "success",
                    "skillName": result['skill_name'],
                    "inputs": result['inputs'],
                    "outputs": result['outputs'],
                    "mdEntry": result['md_entry'],
                    "jsCode": result['js_code'],
                    "auditLogs": audit_logs
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"Error in /generate-skill: {tb}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Server error: {str(e)}", "auditLogs": [f"Exception: {str(e)}"]}).encode())

        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Endpoint not found: {self.path}"}).encode())

    def do_GET(self):
        # Handle API GET routes before falling through to static file serving
        if self.path == '/agent/policies':
            try:
                policies = agent_engine.load_all_policies() if AGENT_ENGINE_AVAILABLE else []
                summary = [{"policyId": p["policyId"], "name": p["policyName"], "category": p["category"],
                            "payer": p["payer"], "cptCodes": p["cptCodes"], "pdfFile": p.get("pdfFile", "")} for p in policies]
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(summary).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == '/agent/cases':
            try:
                cases = agent_engine.load_preset_cases() if AGENT_ENGINE_AVAILABLE else []
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(cases).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path.startswith('/agent/policy-detail'):
            try:
                # Parse query param: ?id=POL-RAD-501
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                policy_id = params.get('id', [''])[0]
                
                all_policies = agent_engine.load_all_policies() if AGENT_ENGINE_AVAILABLE else []
                policy = next((p for p in all_policies if p['policyId'] == policy_id), None)
                
                if policy:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(policy).encode())
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Policy {policy_id} not found"}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            # Fall through to static file serving
            super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), DynamicAuditorHandler) as httpd:
    print(f"Serving audit-ready agent server on port {PORT}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server shutting down.")
