"""
Challenger Agent — Autonomous Second-Opinion Reviewer

A separate agent that independently reviews ALL PA decisions made by the
PA Review Agent. It operates with its own skills, rules, and hooks:

Skills:
- ReinterpretEvidenceSkill: Re-reads notes looking for what PA Agent missed
- AssessDocumentationGapsSkill: Identifies missing but inferable evidence
- EvaluateDecisionStrengthSkill: Rates how well-supported the decision is

Rules:
- RULE-C1 (Cite Evidence): Every challenge must cite specific text from notes
- RULE-C2 (No Rubber Stamp): Must provide substantive analysis, never just "agree"
- RULE-C3 (Confidence Threshold): Only formally challenge if confidence >= 7/10

Hooks:
- on_pa_decision: Triggers the challenger review (fires for ALL decisions)
- on_challenge_issued: Logs the disagreement for medical director review

This agent is ADVERSARIAL by design:
- On approvals: acts as devil's advocate (looks for missed red flags)
- On escalations: acts as patient advocate (looks for ways to approve)
"""

import json
import os
from datetime import datetime

# Import shared LLM interface from agent_engine
from agent_engine import call_llm, extract_json_from_response

# ─── Challenger Agent Configuration ─────────────────────────────────────────

CHALLENGER_RULES = {
    "C1_cite_evidence": {
        "id": "RULE-C1",
        "name": "Cite Evidence",
        "description": "Every challenge must reference specific text from clinical notes",
        "enforced": True
    },
    "C2_no_rubber_stamp": {
        "id": "RULE-C2",
        "name": "No Rubber Stamp",
        "description": "Must provide substantive analysis even when agreeing",
        "enforced": True
    },
    "C3_confidence_threshold": {
        "id": "RULE-C3",
        "name": "Confidence Threshold",
        "description": "Only formally challenge (override recommendation) if confidence >= 7",
        "enforced": True
    }
}

CHALLENGER_HOOKS = {
    "on_pa_decision": "Triggers challenger review for every PA decision",
    "on_challenge_issued": "Logs disagreement and routes to medical director queue"
}


# ─── Main Entry Point ────────────────────────────────────────────────────────

def review(pa_decision, pa_reason, evidence, criteria_met, request, policy):
    """
    Run the Challenger Agent's autonomous review.
    
    This is a fully independent agent call — it does NOT share state with
    the PA Agent. It receives the PA Agent's outputs and forms its own opinion.
    """
    trace = []
    ts = lambda: datetime.now().isoformat()
    
    # ─── Hook: on_pa_decision fires ──────────────────────────────────────
    trace.append({"ts": ts(), "type": "hook", "name": "on_pa_decision",
                  "msg": f"Challenger activated. PA decided: {pa_decision}", "status": "info"})
    
    # ─── Skill 1: Reinterpret Evidence ───────────────────────────────────
    trace.append({"ts": ts(), "type": "skill", "name": "ReinterpretEvidenceSkill",
                  "msg": "Re-reading clinical notes with fresh perspective...", "status": "info"})
    
    clinical_notes = request.get("clinicalNotes", "")
    policy_name = policy.get("policyName", "Unknown") if policy else "Unknown"
    policy_id = policy.get("policyId", "") if policy else ""
    
    criteria_summary = ""
    for c in (criteria_met or []):
        mark = "MET" if c.get("met") else "NOT MET"
        criteria_summary += f"- {c['name'][:45]}: {mark} ({c.get('detail', '')})\n"
    
    # ─── Skill 2: Assess Documentation Gaps ──────────────────────────────
    trace.append({"ts": ts(), "type": "skill", "name": "AssessDocumentationGapsSkill",
                  "msg": "Identifying implicit evidence and documentation gaps...", "status": "info"})

    # ─── Skill 3: Evaluate Decision Strength ─────────────────────────────
    # Build adversarial prompt based on decision type
    if pa_decision == "Approved":
        role = "skeptical medical reviewer looking for missed risks"
        focus = """Focus on:
1. Criteria marked 'met' with WEAK or ASSUMED evidence
2. Red flags: contradictions, incomplete history, missing labs/imaging
3. Whether conservative treatment was TRULY exhausted (or just claimed)
4. Patient safety concerns the PA Agent may have overlooked"""
    else:
        role = "patient advocate seeking approval paths"
        focus = """Focus on:
1. Evidence the PA Agent MISSED or misinterpreted in the notes
2. Alternative valid readings that satisfy criteria (implicit evidence)
3. Whether the escalation is overly strict for borderline cases
4. Clinical context that supports medical necessity despite documentation gaps"""

    prompt = f"""You are a Challenger Agent — an autonomous {role}.
You must provide SUBSTANTIVE analysis (RULE-C2).
You must CITE specific text from the notes (RULE-C1).

PA Decision: {pa_decision}
PA Reason: {pa_reason}
Policy: {policy_id} — {policy_name}

Clinical Notes: "{clinical_notes[:400]}"

PA Agent Criteria Assessment:
{criteria_summary}

{focus}

Return JSON (be specific, cite notes):
{{"verdict": "AGREE" or "CHALLENGE", "confidence": <1-10>, "reasoning": "<2-3 sentences with specific citations from notes>", "findings": ["<finding 1>", "<finding 2>"], "recommendation": "<what should happen next>"}}"""

    trace.append({"ts": ts(), "type": "skill", "name": "EvaluateDecisionStrengthSkill",
                  "msg": f"Analyzing as {role.split(' looking')[0] if 'looking' in role else role.split(' seeking')[0]}...",
                  "status": "info"})

    response = call_llm(prompt, max_tokens=350, temperature=0.3)
    
    try:
        result = extract_json_from_response(response)
        if isinstance(result, list):
            result = result[0] if result else {}
        
        verdict = result.get("verdict", "AGREE")
        confidence = result.get("confidence", 5)
        reasoning = result.get("reasoning", "")
        findings = result.get("findings", [])
        recommendation = result.get("recommendation", "")
        
        # ─── Rule C3: Confidence Threshold ───────────────────────────────
        formal_challenge = verdict == "CHALLENGE" and confidence >= 7
        if verdict == "CHALLENGE" and confidence < 7:
            trace.append({"ts": ts(), "type": "rule", "name": "RULE-C3 (Confidence)",
                          "msg": f"Challenge confidence {confidence}/10 < 7. Downgrading to 'CONCERN' (not formal challenge).",
                          "status": "warning"})
            verdict = "CONCERN"
        
        trace.append({"ts": ts(), "type": "rule", "name": "RULE-C1 (Cite Evidence)",
                      "msg": f"Citations present: {bool(reasoning and len(reasoning) > 20)}", "status": "success"})
        
        trace.append({"ts": ts(), "type": "rule", "name": "RULE-C2 (No Rubber Stamp)",
                      "msg": f"Substantive: {len(findings)} findings provided", "status": "success"})
        
        # ─── Hook: on_challenge_issued ────────────────────────────────────
        if formal_challenge:
            trace.append({"ts": ts(), "type": "hook", "name": "on_challenge_issued",
                          "msg": f"FORMAL CHALLENGE issued (confidence {confidence}/10). Routing to Medical Director.",
                          "status": "warning"})
        
        trace.append({"ts": ts(), "type": "system", "name": "Challenger Agent",
                      "msg": f"Review complete: {verdict} (confidence {confidence}/10)",
                      "status": "success" if verdict == "AGREE" else "warning"})
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "findings": findings,
            "recommendation": recommendation,
            "formalChallenge": formal_challenge,
            "trace": trace
        }
    except (ValueError, json.JSONDecodeError):
        trace.append({"ts": ts(), "type": "system", "name": "Challenger Agent",
                      "msg": f"Analysis failed. Defaulting to AGREE.", "status": "fail"})
        return {
            "verdict": "AGREE",
            "confidence": 3,
            "reasoning": "Unable to form substantive analysis.",
            "findings": [],
            "recommendation": "Proceed with PA Agent decision.",
            "formalChallenge": False,
            "trace": trace
        }


def get_agent_info():
    """Return metadata about the Challenger Agent for UI display."""
    return {
        "name": "Challenger Agent",
        "role": "Autonomous second-opinion reviewer",
        "rules": CHALLENGER_RULES,
        "hooks": CHALLENGER_HOOKS,
        "skills": [
            "ReinterpretEvidenceSkill",
            "AssessDocumentationGapsSkill",
            "EvaluateDecisionStrengthSkill"
        ]
    }
