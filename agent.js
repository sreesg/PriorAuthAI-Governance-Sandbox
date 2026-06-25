import { hooks as defaultHooks } from './hooks.js';
import * as StaticSkills from './skills.js';
import { RegoInterpreter } from './regoInterpreter.js';

/**
 * PriorAuthAgent — Implements Progressive Disclosure at the Execution Level
 * 
 * Progressive Disclosure is not just a UI pattern here. The agent itself receives
 * rules, policies, and context INCREMENTALLY — only what is needed at each pipeline
 * stage. This prevents the agent from being overwhelmed with irrelevant context and
 * mirrors how a human clinical reviewer works: you don't read the entire policy manual
 * before looking at a case — you pull up the specific guideline when you need it.
 * 
 * Architecture:
 * - Stage 1 (Intake): Agent receives ONLY the raw request + PHI rules
 * - Stage 2 (Coverage): Agent receives ONLY member eligibility data + coverage rules
 * - Stage 3 (Evidence): Agent receives ONLY the clinical guidelines relevant to THIS CPT code
 * - Stage 4 (Evaluation): Agent receives ONLY the Rego rules applicable to the matched policy
 * - Stage 5 (Decision): Agent receives ONLY the conservatism guardrails + notice generation rules
 * 
 * At no point does the agent hold the full policy corpus. Each stage discloses only
 * what's needed for the current decision gate.
 */
export class PriorAuthAgent {
  constructor() {
    // Initialize lifecycle hooks registry
    this.hooks = {
      on_request_received: [...defaultHooks.on_request_received],
      on_guidelines_loaded: [...defaultHooks.on_guidelines_loaded],
      on_evidence_extracted: [...defaultHooks.on_evidence_extracted],
      on_criteria_evaluated: [...defaultHooks.on_criteria_evaluated],
      on_notice_generated: [...defaultHooks.on_notice_generated]
    };

    // Load guardrails policies — these are the RULE definitions (metadata only).
    // The actual enforcement logic lives in hooks and is disclosed per-stage.
    this.rules = {
      phiRedaction: {
        id: "RULE-01",
        name: "PHI Redaction Rule",
        description: "Redact Patient Names, SSNs, and DOBs from logging payloads before external storage or API calls.",
        enabled: true,
        stage: 1  // Progressive: Only active during intake stage
      },
      clinicalConservatism: {
        id: "RULE-02",
        name: "Clinical Conservatism Rule",
        description: "Partially met or ambiguous guidelines must escalate to a Human Medical Director instead of automated denial.",
        enabled: true,
        stage: 4  // Progressive: Only active during decision stage
      },
      citationCompulsory: {
        id: "RULE-03",
        name: "Citation Compulsory Rule",
        description: "Every decision must cite the clinical policy number and match the specific criteria ID.",
        enabled: true,
        stage: 5  // Progressive: Only active during notice stage
      },
      codeMatch: {
        id: "RULE-04",
        name: "Code Match Rule",
        description: "Validate standard formatting of CPT and ICD-10 codes, and confirm they map to active policies.",
        enabled: true,
        stage: 2  // Progressive: Only active during coverage/validation stage
      },
      plainLanguage: {
        id: "RULE-05",
        name: "Plain Language Rule",
        description: "Notices for patients must translate medical acronyms (e.g. CPT, MRI, PT) to plain language.",
        enabled: true,
        stage: 5  // Progressive: Only active during notice stage
      }
    };

    this.skills = {};
    this.trace = [];
    this.regoInterpreter = new RegoInterpreter();
    
    // Load skills statically on boot
    this.loadStaticSkills();
  }

  // Load standard skills defined in skills.js
  loadStaticSkills() {
    Object.keys(StaticSkills).forEach(name => {
      this.skills[name] = StaticSkills[name];
    });
  }

  // Scan and reload skills.js to discover dynamically added exports
  async reloadSkills() {
    this.logTrace("system", "Agent Engine", "Scanning skills.js for updated capabilities...", "info");
    
    const cacheBuster = `./skills.js?update=${Date.now()}`;
    try {
      const updatedModule = await import(cacheBuster);
      
      let newCount = 0;
      Object.keys(updatedModule).forEach(name => {
        if (!this.skills[name]) {
          newCount++;
        }
        this.skills[name] = updatedModule[name];
      });

      if (newCount > 0) {
        this.logTrace("system", "Agent Engine", `Dynamic discovery completed. Found and registered ${newCount} new skill(s).`, "success");
      } else {
        this.logTrace("system", "Agent Engine", "Dynamic discovery completed. No new skills found.", "info");
      }
      return Object.keys(this.skills);
    } catch (e) {
      this.logTrace("system", "Agent Engine", `Dynamic discovery scan failed: ${e.message}. Falling back to default list.`, "fail");
      this.loadStaticSkills();
      return Object.keys(this.skills);
    }
  }

  clearTrace() {
    this.trace = [];
  }

  logTrace(type, name, message, status = "success", details = null) {
    this.trace.push({
      timestamp: new Date().toISOString(),
      type,
      name,
      message,
      status,
      details
    });
  }

  async triggerHook(hookName, data) {
    this.logTrace("hook", hookName, `Triggered ${hookName} hook lifecycle step.`, "info");
    if (this.hooks[hookName]) {
      for (const callback of this.hooks[hookName]) {
        await callback(data, this);
      }
    }
  }

  /**
   * Progressive Disclosure Helper: Get only the rules active at a given stage.
   * The agent never sees rules that aren't relevant to the current pipeline gate.
   */
  getActiveRulesForStage(stage) {
    const active = {};
    Object.entries(this.rules).forEach(([key, rule]) => {
      if (rule.stage === stage && rule.enabled) {
        active[key] = rule;
      }
    });
    return active;
  }

  /**
   * Main execution pipeline — implements Progressive Disclosure at the agent level.
   * 
   * Each stage:
   * 1. Discloses ONLY the context needed for that stage
   * 2. Executes only the skills relevant to that stage
   * 3. Applies only the rules scoped to that stage
   * 4. Gates progression — later stages don't execute if early stages fail
   * 
   * The Rego rules are NOT loaded until Stage 4 (after evidence extraction).
   * Guidelines are NOT fetched until after coverage is confirmed.
   * Decision rules are NOT consulted until all evidence is gathered.
   */
  async run(request, regoSourceText, aiEvidence = null) {
    this.clearTrace();
    this.logTrace("system", "Agent Engine", "Initializing Prior Authorization review cycle.", "info");
    this.logTrace("system", "Progressive Disclosure", "Agent context is empty. Stages will disclose rules and policies incrementally.", "info");
    
    if (aiEvidence) {
      this.logTrace("system", "Agent Engine", "🧠 ClinicalNLP Engine evidence provided. Using semantic extraction for clinical evaluation.", "success");
    }

    // Context starts minimal — only the raw request. No policies, no rules loaded yet.
    const context = {
      request: request,
      redactedRequest: null,
      coverage: null,
      guidelines: null,       // NOT loaded yet — disclosed at Stage 3
      evidence: null,         // NOT extracted yet — disclosed at Stage 3
      regoResults: null,      // NOT evaluated yet — disclosed at Stage 4
      criteriaResults: [],
      icdCodeMatched: false,
      codeValidationFailed: false,
      coverageError: false,
      decision: "Pending",
      decisionReason: "",
      noticeDraft: ""
    };

    try {
      // ═══════════════════════════════════════════════════════════════════
      // STAGE 1: INTAKE — Disclose only PHI handling rules
      // Agent context: raw request only. No clinical policies visible yet.
      // ═══════════════════════════════════════════════════════════════════
      const stage1Rules = this.getActiveRulesForStage(1);
      this.logTrace("system", "Progressive Disclosure",
        `Stage 1 (Intake): Disclosing ${Object.keys(stage1Rules).length} rule(s): [${Object.values(stage1Rules).map(r => r.id).join(', ')}]. No clinical policies loaded yet.`,
        "info", { disclosedRules: Object.keys(stage1Rules), policiesLoaded: false });

      await this.triggerHook("on_request_received", context);

      // ═══════════════════════════════════════════════════════════════════
      // STAGE 2: COVERAGE & VALIDATION — Disclose eligibility context + code rules
      // Agent context: redacted request + member profile. Guidelines NOT yet visible.
      // ═══════════════════════════════════════════════════════════════════
      const stage2Rules = this.getActiveRulesForStage(2);
      this.logTrace("system", "Progressive Disclosure",
        `Stage 2 (Coverage): Disclosing ${Object.keys(stage2Rules).length} rule(s): [${Object.values(stage2Rules).map(r => r.id).join(', ')}]. Fetching member eligibility only.`,
        "info", { disclosedRules: Object.keys(stage2Rules), guidelinesLoaded: false });

      // Skill: Verify member coverage (discloses member profile)
      if (this.skills.VerifyCoverageSkill) {
        this.skills.VerifyCoverageSkill(context, this);
      } else {
        this.logTrace("system", "Agent Engine", "Required VerifyCoverageSkill is missing.", "fail");
      }

      // Skill: Retrieve guidelines for THIS specific CPT code only
      // Progressive: Agent only sees the single policy relevant to this procedure
      if (this.skills.RetrieveGuidelinesSkill) {
        this.skills.RetrieveGuidelinesSkill(context, this);
      } else {
        this.logTrace("system", "Agent Engine", "Required RetrieveGuidelinesSkill is missing.", "fail");
      }

      // Optional dynamic skill: NPI validation (if registered)
      if (this.skills.VerifyNpiStatusSkill) {
        this.skills.VerifyNpiStatusSkill(context, this);
      } else {
        this.logTrace("system", "Agent Engine", "Optional VerifyNpiStatusSkill not registered.", "info");
      }

      // Hook: Code Match rule fires AFTER guidelines are disclosed (needs them for validation)
      await this.triggerHook("on_guidelines_loaded", context);

      // GATE CHECK: If code validation failed, stop here. Don't disclose further context.
      if (context.codeValidationFailed) {
        this.logTrace("system", "Progressive Disclosure",
          "Gate 2 FAILED. Halting pipeline — no further policies or evidence will be disclosed.",
          "warning");
        context.decision = "Escalated for Human Review";
        context.decisionReason = "Escalated: Invalid or unsupported procedure code (CPT) format.";
        if (this.skills.GenerateDecisionNoticeSkill) {
          this.skills.GenerateDecisionNoticeSkill(context, this);
        }
        await this.triggerHook("on_notice_generated", context);
        return {
          status: "escalated",
          decision: context.decision,
          reason: context.decisionReason,
          trace: this.trace,
          notice: context.noticeDraft
        };
      }

      // ═══════════════════════════════════════════════════════════════════
      // STAGE 3: EVIDENCE EXTRACTION — Disclose clinical notes parsing context
      // Agent context: now has guidelines for THIS procedure. Can extract evidence.
      // Progressive: Rego evaluation rules still NOT loaded.
      // ═══════════════════════════════════════════════════════════════════
      this.logTrace("system", "Progressive Disclosure",
        `Stage 3 (Evidence): Guidelines for CPT ${context.request.cptCode} now disclosed. Extracting clinical evidence. Rego rules NOT yet loaded.`,
        "info", { guidelinesLoaded: !!context.guidelines, regoLoaded: false });

      // Skill: Parse clinical notes — use AI evidence if provided, otherwise regex extraction
      if (aiEvidence) {
        // Use ClinicalNLP Engine extracted evidence (semantic AI)
        context.evidence = {
          symptomsDurationWeeks: aiEvidence.symptomsDurationWeeks || 0,
          therapyWeeks: aiEvidence.therapyWeeks || 0,
          hasObjectiveFindings: aiEvidence.hasObjectiveFindings || false,
          isRheumatologist: aiEvidence.isRheumatologist || false,
          hasRadiographs: aiEvidence.hasRadiographs || false
        };
        this.logTrace("skill", "ExtractClinicalDataSkill (ClinicalNLP)",
          "🧠 Semantic extraction via ClinicalNLP Engine — understands negation, paraphrasing, and context.", "success", {
            ...context.evidence,
            reasoning: aiEvidence.reasoning || "Semantic extraction",
            engine: "ClinicalNLP"
          });
      } else if (this.skills.ExtractClinicalDataSkill) {
        this.skills.ExtractClinicalDataSkill(context, this);
      } else {
        this.logTrace("system", "Agent Engine", "Required ExtractClinicalDataSkill is missing.", "fail");
      }

      await this.triggerHook("on_evidence_extracted", context);

      // ═══════════════════════════════════════════════════════════════════
      // STAGE 4: OPA REGO EVALUATION — NOW disclose the declarative policy rules
      // Agent context: evidence is extracted. NOW we load and evaluate Rego rules.
      // Progressive: This is the first time the full policy logic is consulted.
      // The agent needed evidence FIRST before it could meaningfully evaluate rules.
      // ═══════════════════════════════════════════════════════════════════
      const stage4Rules = this.getActiveRulesForStage(4);
      this.logTrace("system", "Progressive Disclosure",
        `Stage 4 (Evaluation): NOW disclosing OPA Rego policy rules for evaluation. ${Object.keys(stage4Rules).length} governance rule(s) active: [${Object.values(stage4Rules).map(r => r.id).join(', ')}].`,
        "info", { disclosedRules: Object.keys(stage4Rules), regoLoaded: true, evidenceAvailable: !!context.evidence });

      // NOW load the Rego rules — not before. The interpreter only receives
      // the policy text at the moment it's needed for evaluation.
      this.regoInterpreter.load(regoSourceText);
      this.logTrace("system", "OPA Rego Evaluator", "Policy rules disclosed and loaded into evaluator. Beginning criteria matching.", "info");

      const regoInput = {
        cptCode: context.request.cptCode,
        icd10Code: context.request.icd10Code,
        extractedEvidence: context.evidence,
        clinicalNotes: context.request.clinicalNotes
      };
      
      const regoResults = this.regoInterpreter.evaluate(regoInput);
      context.regoResults = regoResults;

      // Pipe OPA Rego evaluation trace into agent timeline
      regoResults.evalLogs.forEach(logLine => {
        this.logTrace("rule", "rules.rego", logLine, "info");
      });

      // Skill: Map evidence against the disclosed criteria checklist
      if (this.skills.EvaluateClinicalCriteriaSkill) {
        this.skills.EvaluateClinicalCriteriaSkill(context, this);
      }

      // Run dynamically registered skills (beyond core pipeline)
      const coreSkills = new Set([
        'VerifyCoverageSkill', 'RetrieveGuidelinesSkill', 'ExtractClinicalDataSkill',
        'EvaluateClinicalCriteriaSkill', 'GenerateDecisionNoticeSkill', 'VerifyNpiStatusSkill'
      ]);
      
      for (const [skillName, skillFn] of Object.entries(this.skills)) {
        if (!coreSkills.has(skillName)) {
          this.logTrace("system", "Agent Engine", `Invoking dynamically registered skill: ${skillName}.`, "info");
          try {
            skillFn(context, this);
          } catch (e) {
            this.logTrace("system", "Agent Engine", `Error running ${skillName}: ${e.message}`, "fail");
          }
        }
      }

      // ═══════════════════════════════════════════════════════════════════
      // STAGE 5: DECISION & NOTICE — Disclose governance and output rules
      // Agent context: All evidence evaluated. NOW apply conservatism guardrails
      // and notice generation rules. These were intentionally withheld until now
      // so the evaluation stage couldn't be biased by knowing the output format.
      // ═══════════════════════════════════════════════════════════════════
      const stage5Rules = this.getActiveRulesForStage(5);
      this.logTrace("system", "Progressive Disclosure",
        `Stage 5 (Decision & Notice): Disclosing ${Object.keys(stage4Rules).length + Object.keys(stage5Rules).length} final rule(s): [${[...Object.values(stage4Rules), ...Object.values(stage5Rules)].map(r => r.id).join(', ')}]. Applying clinical conservatism and generating notice.`,
        "info", { disclosedRules: [...Object.keys(stage4Rules), ...Object.keys(stage5Rules)], allStagesComplete: true });

      // Hook: Clinical Conservatism — decides Approved vs Escalated
      await this.triggerHook("on_criteria_evaluated", context);

      // Skill: Generate the decision notice letter
      if (this.skills.GenerateDecisionNoticeSkill) {
        this.skills.GenerateDecisionNoticeSkill(context, this);
      }

      // Hook: Citation Compulsory + Plain Language enforcement
      await this.triggerHook("on_notice_generated", context);

      // ═══════════════════════════════════════════════════════════════════
      // COMPLETE — Log final progressive disclosure summary
      // ═══════════════════════════════════════════════════════════════════
      const totalRulesUsed = Object.values(this.rules).filter(r => r.enabled).length;
      const totalSkillsUsed = Object.keys(this.skills).length;
      this.logTrace("system", "Progressive Disclosure",
        `Pipeline complete. Context was disclosed across 5 stages: ${totalRulesUsed} rules activated incrementally, ${totalSkillsUsed} skills invoked on-demand. Decision: ${context.decision}.`,
        "success");

      this.logTrace("system", "Agent Engine", `Execution finished with decision: ${context.decision}.`, "info");
      return {
        status: context.decision === "Approved" ? "approved" : "escalated",
        decision: context.decision,
        reason: context.decisionReason,
        trace: this.trace,
        notice: context.noticeDraft
      };

    } catch (e) {
      this.logTrace("system", "Agent Engine", `Fatal unhandled runtime exception: ${e.message}`, "fail");
      return {
        status: "error",
        decision: "Escalated for Human Review",
        reason: `Runtime Exception: ${e.message}`,
        trace: this.trace,
        notice: `ERROR: Automated prior authorization failed due to a system error. Re-routing case to human administrative reviews.`
      };
    }
  }
}
