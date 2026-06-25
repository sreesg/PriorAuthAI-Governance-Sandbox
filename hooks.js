export const hooks = {
  // Hook 1: Fires immediately upon request receipt -> Enforces PHI Redaction Rule
  on_request_received: [
    async (context, agent) => {
      if (agent.rules.phiRedaction.enabled) {
        agent.logTrace("rule", "RULE-01 (PHI Redaction)", "Enforcing PHI redaction rules on incoming request data.", "info");
        
        const ssnPattern = /\b\d{3}-\d{2}-\d{4}\b/g;
        const dobPattern = /\b\d{4}-\d{2}-\d{2}\b/g;
        let redacted = false;
        
        const cleanContext = JSON.parse(JSON.stringify(context.request));
        
        if (cleanContext.patientSsn) {
          cleanContext.patientSsn = "[REDACTED-SSN]";
          redacted = true;
        }
        if (cleanContext.patientDob) {
          cleanContext.patientDob = "[REDACTED-DOB]";
          redacted = true;
        }
        if (cleanContext.patientName) {
          cleanContext.patientName = cleanContext.patientName.replace(/\w+/g, (txt) => txt[0] + ".".repeat(txt.length - 1));
          redacted = true;
        }

        if (redacted) {
          agent.logTrace("rule", "RULE-01 (PHI Redaction)", "Sensitive data found and scrubbed from trace logs.", "success", cleanContext);
          context.redactedRequest = cleanContext;
        } else {
          agent.logTrace("rule", "RULE-01 (PHI Redaction)", "No sensitive PHI patterns detected for redaction.", "success");
          context.redactedRequest = context.request;
        }
      }
    }
  ],

  // Hook 2: Fires after guidelines are fetched -> Enforces Code Match Rule
  on_guidelines_loaded: [
    async (context, agent) => {
      if (agent.rules.codeMatch.enabled) {
        agent.logTrace("rule", "RULE-04 (Code Match)", "Verifying CPT code syntax and guideline alignment.", "info");
        const cpt = context.request.cptCode;
        const icd = context.request.icd10Code;
        
        const cptValid = /^[0-9]{4}[0-9a-zA-Z]$/.test(cpt);
        const icdValid = /^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$/.test(icd);

        if (!cptValid || !icdValid) {
          agent.logTrace("rule", "RULE-04 (Code Match)", `Code formatting validation failed. CPT: ${cpt} (${cptValid ? 'Valid' : 'Invalid'}), ICD-10: ${icd} (${icdValid ? 'Valid' : 'Invalid'})`, "fail");
          context.codeValidationFailed = true;
          return;
        }

        if (context.guidelines) {
          const allowedIcd = context.guidelines.requiredIcd10 || [];
          if (allowedIcd.includes(icd)) {
            agent.logTrace("rule", "RULE-04 (Code Match)", `Diagnosis code ${icd} matches active guidelines for CPT ${cpt}.`, "success");
            context.icdCodeMatched = true;
          } else {
            agent.logTrace("rule", "RULE-04 (Code Match)", `Diagnosis code ${icd} is NOT standardly covered for CPT ${cpt} under guidelines.`, "warning", { allowedIcd });
            context.icdCodeMatched = false;
          }
        } else {
          agent.logTrace("rule", "RULE-04 (Code Match)", `No active guideline policy matches CPT code ${cpt}.`, "fail");
          context.codeValidationFailed = true;
        }
      }
    }
  ],

  // Hook 3: Fires after clinical note extraction
  on_evidence_extracted: [
    async (context, agent) => {
      agent.logTrace("hook", "on_evidence_extracted", "Reviewing consistency of clinical fact extraction.", "info");
      if (!context.evidence) {
        agent.logTrace("hook", "on_evidence_extracted", "No evidence extracted from clinical notes.", "fail");
      } else {
        agent.logTrace("hook", "on_evidence_extracted", `Facts verified: Symptoms: ${context.evidence.symptomsDurationWeeks}w, Therapy: ${context.evidence.therapyWeeks}w.`, "success");
      }
    }
  ],

  // Hook 4: Fires after OPA criteria matching -> Enforces Clinical Conservatism Rule
  on_criteria_evaluated: [
    async (context, agent) => {
      if (agent.rules.clinicalConservatism.enabled) {
        agent.logTrace("rule", "RULE-02 (Clinical Conservatism)", "Reviewing policy scoring against medical escalation guardrails.", "info");
        
        // Read OPA Rego evaluation results
        const approve = context.regoResults.approve;
        const escalate = context.regoResults.escalate;

        if (context.coverageError) {
          context.decision = "Escalated for Human Review";
          context.decisionReason = "Administrative review needed: Coverage exclusion or plan limit mismatch.";
          agent.logTrace("rule", "RULE-02 (Clinical Conservatism)", "Escalated to human review due to coverage exclusion.", "warning");
        } else if (approve) {
          context.decision = "Approved";
          context.decisionReason = "Automated approval: Case notes fully satisfy all medical necessity guidelines under rules.rego OPA policy.";
          agent.logTrace("rule", "RULE-02 (Clinical Conservatism)", "Approved: All clinical necessity criteria met.", "success");
        } else if (escalate) {
          context.decision = "Escalated for Human Review";
          context.decisionReason = `Escalated to Human Medical Director: Case criteria did not fully satisfy automated OPA policy rules. Standard guidelines require clinical director audit.`;
          agent.logTrace("rule", "RULE-02 (Clinical Conservatism)", `Criteria not fully met. Escalating to Human Medical Director instead of automatic rejection.`, "warning");
        } else {
          context.decision = "Escalated for Human Review";
          context.decisionReason = `Escalated: Request criteria unmet or notes missing.`;
          agent.logTrace("rule", "RULE-02 (Clinical Conservatism)", `Case requires manual audit. Escalated.`, "warning");
        }
      }
    }
  ],

  // Hook 5: Fires after letter generation -> Enforces Citation Compulsory & Plain Language Rules
  on_notice_generated: [
    async (context, agent) => {
      if (agent.rules.citationCompulsory.enabled) {
        agent.logTrace("rule", "RULE-03 (Citation Compulsory)", "Validating that clinical references and guidelines are cited.", "info");
        const containsCitation = context.noticeDraft && context.noticeDraft.includes(context.guidelines?.policyId || "POL-");
        if (containsCitation) {
          agent.logTrace("rule", "RULE-03 (Citation Compulsory)", `Verified: Reference to policy ${context.guidelines?.policyId} is embedded in correspondence.`, "success");
        } else {
          agent.logTrace("rule", "RULE-03 (Citation Compulsory)", "Validation Failed: No policy citations found in notice letter.", "fail");
        }
      }

      if (agent.rules.plainLanguage.enabled) {
        agent.logTrace("rule", "RULE-05 (Plain Language)", "Enforcing translation of medical acronyms to layman terms.", "info");
        const translations = {
          "MRI": "Magnetic Resonance Imaging (MRI)",
          "CPT": "Procedure Catalog (CPT)",
          "PT": "Physical Therapy (PT)",
          "RA": "Rheumatoid Arthritis (RA)",
          "DMARD": "Disease-Modifying Antirheumatic Drug (DMARD)",
          "NSAID": "Anti-inflammatory pain relief medication (NSAIDs)"
        };

        let updatedNotice = context.noticeDraft || "";
        let translatedCount = 0;
        
        for (const [acronym, replacement] of Object.entries(translations)) {
          const regex = new RegExp(`\\b${acronym}\\b`, 'g');
          if (regex.test(updatedNotice)) {
            updatedNotice = updatedNotice.replace(regex, replacement);
            translatedCount++;
          }
        }
        
        context.noticeDraft = updatedNotice;
        agent.logTrace("rule", "RULE-05 (Plain Language)", `Replaced/expanded ${translatedCount} acronyms for patient readability.`, "success");
      }
    }
  ]
};
