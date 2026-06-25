import { MEMBER_BENEFITS, CLINICAL_GUIDELINES } from './cases.js';

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

  letter += `DATE: ${dateStr}
`;
  letter += `TO: Patient ${patientName} & Provider ${providerName}
`;
  letter += `RE: PRIOR AUTHORIZATION REQUEST DECISION
`;
  letter += `===========================================

`;
  
  if (decision === "Approved") {
    letter += `We are pleased to inform you that your prior authorization request for procedure CPT code ${cpt} has been APPROVED.

`;
    letter += `CLINICAL SUMMARY:
`;
    letter += `The medical reviewer confirmed that the clinical facts meet medical necessity requirements.
`;
  } else {
    letter += `Your prior authorization request for procedure CPT code ${cpt} has been ESCALATED FOR MANUAL REVIEW.

`;
    letter += `CLINICAL SUMMARY:
`;
    letter += `The automated clinical review determined that guidelines were not fully satisfied or coverage issues require administrative audit. A Clinical Medical Director is reviewing your case and will issue a final decision within 48 hours.
`;
  }

  letter += `
POLICY REFERENCE:
`;
  if (policy) {
    letter += `This review was performed in accordance with policy ${policy.policyId} (${policy.policyName}, effective date: ${policy.effectiveDate}).
`;
    
    letter += `
CRITERIA MET CHECKLIST:
`;
    context.criteriaResults.forEach(c => {
      letter += `[${c.met ? 'X' : ' '}] ${c.id}: ${c.text} (Actual: ${c.actualValue})
`;
    });
  } else {
    letter += `No matching active clinical medical policy guidelines were found for procedure CPT code ${cpt}.
`;
  }

  letter += `
For any questions regarding this decision, contact Payer Member Services.
`;

  context.noticeDraft = letter;
  agent.logTrace("skill", "GenerateDecisionNoticeSkill", "Prior Authorization notice text generated.", "success");
}

// Auto-generated skill: Check the patient sex and make sure it male and in the age group 60 and above
export function CheckAgePatientSkill(context, agent) {
  agent.logTrace("skill", "CheckAgePatientSkill", "Executing: Check the patient sex and make sure it male and in the age g...", "info");
  function calculateAge(dob) {
    if (!dob) return 0;
    const birthDate = new Date(dob);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) age--;
    return age;
  }
  const check0 = calculateAge(context.request.patientDob);
  const condition0 = check0 >= 60;
  const condition1 = context.request.clinicalNotes.toLowerCase().includes("male");
  const triggered = condition0 && condition1;
  if (triggered) {
    context.checkAgePatientValid = "Flagged";
    agent.logTrace("skill", "CheckAgePatientSkill", "Condition met: review. Case flagged.", "warning");
  } else {
    context.checkAgePatientValid = "Passed";
    agent.logTrace("skill", "CheckAgePatientSkill", "All checks passed. No flag raised.", "success");
  }
}
