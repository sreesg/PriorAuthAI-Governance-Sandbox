/**
 * CRF Workflow Explainer — Animated UI showing how PA leverages
 * Axisweave semantic search, Causal Ontology Graph, and BEACON Harness.
 *
 * This creates a full-page animated walkthrough explaining:
 *   1. PA Request arrives
 *   2. BEACON L1 authenticates
 *   3. Axisweave retrieves evidence (semantic + BM25 hybrid)
 *   4. Causal Graph provides active clinical state
 *   5. Context Planner assembles Briefing Packet
 *   6. Clinical Reasoning Agent evaluates criteria
 *   7. OPA Challenger verifies (KMS signatures + policy rules)
 *   8. Human Gate routes (Auto-Approve or Escalate to MD)
 *   9. Evidence Bundle produced with lineage trail
 */

const WORKFLOW_STEPS = [
  {
    id: "pa-request",
    icon: "📋",
    title: "PA Request Received",
    subtitle: "Prior Authorization request enters the system",
    description: "A clinical provider submits a Prior Authorization request with member ID, CPT code, ICD-10 diagnosis, and clinical notes. The request triggers the BEACON safety harness pipeline.",
    components: ["Provider Portal", "Member ID", "CPT Code", "Clinical Notes"],
    animation: "fadeInDown",
    color: "#6366f1",
  },
  {
    id: "beacon-l1",
    icon: "🔐",
    title: "BEACON L1 — Identity & RBAC",
    subtitle: "Authentication and role-based access control",
    description: "The BEACON harness authenticates the requesting identity, verifies RBAC permissions, and creates a trace context. Every subsequent action is attributed to this identity for audit purposes.",
    components: ["API Key Validation", "RBAC Policy Check", "Trace Context Created", "PHI Masking Active"],
    animation: "fadeInLeft",
    color: "#8b5cf6",
  },
  {
    id: "axisweave",
    icon: "🧬",
    title: "Axisweave Semantic Retrieval",
    subtitle: "Hybrid dense + BM25 search with KMS verification",
    description: "Axisweave executes a hybrid retrieval combining dense vector cosine similarity (semantic meaning) and BM25 sparse index (keyword matching) via Reciprocal Rank Fusion. Only evidence with valid KMS cryptographic signatures is included — tampered documents are excluded with alerts.",
    components: [
      "Dense Vector Search (Top 50)",
      "BM25 Sparse Search (Top 50)",
      "Reciprocal Rank Fusion (k=60)",
      "KMS Signature Verification",
      "Tamper Alert Generation",
    ],
    animation: "fadeInRight",
    color: "#06b6d4",
    badge: "AXISWEAVE",
  },
  {
    id: "causal-graph",
    icon: "🕸️",
    title: "Causal Ontology Graph",
    subtitle: "Neo4j active clinical state — diagnoses, Rx, SDOH, policies",
    description: "The Causal Ontology Graph (Neo4j) provides the member's current active clinical state: diagnoses, prescriptions, completed therapies, SDOH factors, and governing policy rules. Only active records are returned — resolved, discontinued, or superseded entries are excluded.",
    components: [
      "Active Diagnoses (ICD-10)",
      "Current Prescriptions",
      "Failed Therapies (w/ outcomes)",
      "SDOH Factors (explicit + inferred)",
      "Governing Policy Rules",
      "Provider Relationships",
    ],
    animation: "fadeInLeft",
    color: "#10b981",
    badge: "NEO4J",
  },
  {
    id: "context-planner",
    icon: "📦",
    title: "BEACON L2 — Context Planner",
    subtitle: "Assembles the Briefing Packet (30s timeout)",
    description: "The Context Planner assembles a Briefing Packet by combining the Graph state + Axisweave evidence + Clinical Inference Engine results. It filters to only CPT-relevant information and caps at 20 evidence snippets with ≥0.5 relevance score. Inferred SDOH factors are tagged separately from explicit findings.",
    components: [
      "Graph State Query",
      "Evidence Snippets (max 20, score ≥ 0.5)",
      "CPT Relevance Filtering",
      "Clinical Inference Engine (SDOH detection)",
      "Briefing Packet Schema Validation",
    ],
    animation: "fadeInUp",
    color: "#f59e0b",
  },
  {
    id: "reasoning",
    icon: "🧠",
    title: "Clinical Reasoning Agent",
    subtitle: "Bounded reasoning with MCP Gateway tool access",
    description: "The agent reasons over the Briefing Packet within bounded constraints. Additional evidence retrieval is permitted through the MCP Gateway (max 10 calls), and each retrieved chunk is KMS-verified. The agent produces per-criterion MET/NOT_MET/INDETERMINATE assessments.",
    components: [
      "Bounded Retrieval (max 10 calls)",
      "MCP Gateway Tool Catalog",
      "Criteria Assessment (per-criterion)",
      "Evidence Bundle Assembly",
      "Lineage Trail Construction",
    ],
    animation: "fadeInRight",
    color: "#ec4899",
  },
  {
    id: "opa-challenger",
    icon: "⚖️",
    title: "BEACON L5 — OPA Challenger",
    subtitle: "Independent verification (no shared state with agent)",
    description: "The OPA Challenger Agent independently verifies: (1) all KMS signatures are valid (30s timeout), (2) the decision complies with OPA policy rules in rules.rego (10s timeout). It operates with zero shared mutable state with the primary agent. On any failure → escalate to Medical Director.",
    components: [
      "KMS Signature Verification (all evidence)",
      "OPA rules.rego Evaluation",
      "PASS/FAIL with violated rule IDs",
      "Tamper Alert Escalation",
      "Independent — No Shared State",
    ],
    animation: "fadeInLeft",
    color: "#ef4444",
  },
  {
    id: "human-gate",
    icon: "👨‍⚕️",
    title: "BEACON L7 — Human Gates",
    subtitle: "NEVER auto-denies. Only approves or escalates.",
    description: "The Human Gate enforces the no-automated-denial policy. All criteria MET + verification PASS → Auto-Approve. Any NOT_MET, INDETERMINATE, or verification FAIL → Escalate to Medical Director with full artifact package (Briefing Packet, criteria assessment, OPA findings, execution trace).",
    components: [
      "All MET + PASS → Auto-Approve ✅",
      "Any NOT_MET → Escalate to MD ⚠️",
      "INDETERMINATE → Escalate to MD ⚠️",
      "Verification FAIL → Escalate to MD ⚠️",
      "NEVER produces automated denial ❌",
    ],
    animation: "fadeInUp",
    color: "#22c55e",
  },
  {
    id: "evidence-bundle",
    icon: "📄",
    title: "Evidence Bundle Output",
    subtitle: "Complete lineage trail with cryptographic provenance",
    description: "The final Evidence Bundle contains: execution_id, decision, reasoning, ordered lineage trail (each conclusion linked to source evidence_id and retrieval timestamp), and original KMS document signatures. The complete execution trace (every agent action, tool call, retrieval) is attached for full audit reconstruction.",
    components: [
      "Decision + Reason",
      "Lineage Trail (conclusion → evidence_id → timestamp)",
      "Original Document Signatures (KMS)",
      "Complete Execution Trace (immutable, append-only)",
      "7-Year Retention",
    ],
    animation: "fadeInDown",
    color: "#6366f1",
  },
];


/**
 * Renders the CRF Workflow Explainer with step-by-step animation.
 */
export function render(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return { play: () => {}, reset: () => {} };

  container.innerHTML = buildExplainerHTML();
  let currentStep = -1;
  let autoPlayInterval = null;

  // Start auto-play
  setTimeout(() => advanceStep(), 500);
  autoPlayInterval = setInterval(() => advanceStep(), 4000);

  function advanceStep() {
    currentStep = (currentStep + 1) % WORKFLOW_STEPS.length;
    highlightStep(currentStep);
  }

  function highlightStep(idx) {
    // Update flow diagram
    document.querySelectorAll('.wf-flow-node').forEach((el, i) => {
      el.classList.remove('wf-active', 'wf-completed');
      if (i < idx) el.classList.add('wf-completed');
      if (i === idx) el.classList.add('wf-active');
    });

    // Update detail panel
    const step = WORKFLOW_STEPS[idx];
    const detail = document.getElementById('wf-detail-panel');
    if (detail) {
      detail.style.borderColor = step.color;
      detail.innerHTML = `
        <div class="wf-detail-header" style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;">
          <span style="font-size:1.8rem;">${step.icon}</span>
          <div>
            <h3 style="font-size:0.85rem;font-weight:700;color:var(--text-main);margin:0;">${step.title}</h3>
            <p style="font-size:0.68rem;color:${step.color};margin:0;font-weight:600;">${step.subtitle}</p>
          </div>
          ${step.badge ? `<span style="margin-left:auto;font-size:0.55rem;font-weight:700;color:${step.color};background:${step.color}15;padding:0.15rem 0.5rem;border-radius:12px;border:1px solid ${step.color}30;">${step.badge}</span>` : ''}
        </div>
        <p style="font-size:0.72rem;color:var(--text-main);line-height:1.5;margin-bottom:0.6rem;">${step.description}</p>
        <div class="wf-components" style="display:flex;flex-direction:column;gap:0.25rem;">
          ${step.components.map(c => `
            <div style="display:flex;align-items:center;gap:0.4rem;padding:0.2rem 0.5rem;background:var(--bg-secondary);border-radius:6px;border:1px solid var(--card-border);">
              <span style="color:${step.color};font-size:0.6rem;">●</span>
              <span style="font-size:0.65rem;color:var(--text-main);">${c}</span>
            </div>
          `).join('')}
        </div>
      `;
    }

    // Update data flow arrows
    const arrows = document.getElementById('wf-data-flow');
    if (arrows) {
      const flows = getDataFlowForStep(idx);
      arrows.innerHTML = flows.map(f => `
        <div style="display:flex;align-items:center;gap:0.3rem;padding:0.15rem 0;font-size:0.58rem;">
          <span style="color:${step.color};">→</span>
          <span style="color:var(--text-muted);">${f}</span>
        </div>
      `).join('');
    }
  }

  function getDataFlowForStep(idx) {
    const flows = [
      ["Provider → BEACON Pipeline"],
      ["Credentials → RBAC Policy → Trace Context"],
      ["Query Embedding → Qdrant Dense Index", "Query Terms → BM25 Sparse Index", "RRF Fusion → Top 20 Chunks", "KMS Verify → Valid Evidence Only"],
      ["Cypher Query → Active Diagnoses", "Member → Prescriptions (failed/active)", "Member → SDOH Factors", "Events → Policy Rules"],
      ["Graph State + Evidence + Inferred SDOH → Briefing Packet"],
      ["Briefing Packet → Agent → Criteria Assessment", "MCP Gateway → Additional Retrieval (bounded)"],
      ["Evidence Bundle → KMS Check (30s)", "Decision → OPA rules.rego (10s)", "Violations → Escalation Trigger"],
      ["All MET + PASS → AUTO-APPROVE", "Any gap → ESCALATE with full artifacts"],
      ["Lineage Trail → Source Provenance", "Execution Trace → 7-Year Immutable Store"],
    ];
    return flows[idx] || [];
  }

  function buildExplainerHTML() {
    return `
      <div class="crf-workflow-explainer glass-card" style="padding:1rem;">
        <div class="wf-header" style="text-align:center;margin-bottom:1rem;">
          <h2 style="font-size:1.1rem;font-weight:800;color:var(--text-main);margin:0;">
            🧬 Clinical Reasoning Fabric — PA Workflow
          </h2>
          <p style="font-size:0.7rem;color:var(--text-muted);margin:0.3rem 0 0;">
            How Axisweave, Causal Graph, and BEACON work together
          </p>
        </div>

        <!-- Architecture Flow Diagram -->
        <div class="wf-architecture" style="display:flex;align-items:center;justify-content:center;gap:0.15rem;padding:0.6rem;background:var(--bg-secondary);border-radius:12px;border:1px solid var(--card-border);margin-bottom:0.8rem;overflow-x:auto;">
          ${WORKFLOW_STEPS.map((step, i) => `
            ${i > 0 ? '<span class="wf-arrow" style="color:var(--text-muted);font-size:0.5rem;">▶</span>' : ''}
            <div class="wf-flow-node" data-step="${i}" style="display:flex;flex-direction:column;align-items:center;gap:0.1rem;padding:0.3rem 0.4rem;border-radius:8px;border:2px solid transparent;cursor:pointer;transition:all 0.3s;min-width:50px;" onclick="document.dispatchEvent(new CustomEvent('wf-goto',{detail:${i}}))">
              <span style="font-size:1rem;">${step.icon}</span>
              <span style="font-size:0.5rem;font-weight:600;color:var(--text-muted);text-align:center;line-height:1.1;max-width:60px;">${step.title.split('—')[0].trim()}</span>
            </div>
          `).join('')}
        </div>

        <!-- Two-Column Layout: Detail + Data Flow -->
        <div style="display:grid;grid-template-columns:2fr 1fr;gap:0.8rem;">
          <!-- Detail Panel -->
          <div id="wf-detail-panel" class="glass-card" style="padding:0.8rem;border-left:3px solid var(--accent-blue);transition:border-color 0.3s;">
            <p style="color:var(--text-muted);font-size:0.72rem;">Click a step above or wait for auto-play...</p>
          </div>
          <!-- Data Flow Panel -->
          <div class="glass-card" style="padding:0.6rem;">
            <div style="font-size:0.62rem;font-weight:700;color:var(--text-main);margin-bottom:0.4rem;">📡 Data Flow</div>
            <div id="wf-data-flow" style="font-family:var(--font-mono);"></div>
          </div>
        </div>

        <!-- Legend -->
        <div style="display:flex;gap:0.8rem;justify-content:center;margin-top:0.8rem;flex-wrap:wrap;">
          <span style="display:flex;align-items:center;gap:0.3rem;font-size:0.58rem;color:var(--text-muted);">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#06b6d4;"></span> Axisweave (Semantic Retrieval)
          </span>
          <span style="display:flex;align-items:center;gap:0.3rem;font-size:0.58rem;color:var(--text-muted);">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;"></span> Causal Graph (Neo4j)
          </span>
          <span style="display:flex;align-items:center;gap:0.3rem;font-size:0.58rem;color:var(--text-muted);">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#8b5cf6;"></span> BEACON Harness (Safety)
          </span>
          <span style="display:flex;align-items:center;gap:0.3rem;font-size:0.58rem;color:var(--text-muted);">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;"></span> OPA Challenger (Verification)
          </span>
        </div>
      </div>

      <!-- Key Differentiator: On-Demand vs Batch -->
      <div class="glass-card" style="margin-top:0.8rem;padding:0.8rem;border-left:3px solid #06b6d4;">
        <h3 style="font-size:0.78rem;font-weight:700;color:var(--text-main);margin:0 0 0.4rem;">⚡ Why On-Demand Semantic Search — Not Batch NLP</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;font-size:0.65rem;">
          <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);border-radius:8px;padding:0.5rem;">
            <div style="font-weight:700;color:#ef4444;margin-bottom:0.3rem;">❌ Traditional (Batch NLP)</div>
            <div style="color:var(--text-muted);line-height:1.6;">• Pre-extracts structured data once<br>• Misclassifications are permanent<br>• New rules require full re-processing<br>• Fixed extraction ≠ context-aware<br>• No provenance back to source</div>
          </div>
          <div style="background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.15);border-radius:8px;padding:0.5rem;">
            <div style="font-weight:700;color:#06b6d4;margin-bottom:0.3rem;">✅ CRF (On-Demand Semantic)</div>
            <div style="color:var(--text-muted);line-height:1.6;">• Retrieves at decision-time per query<br>• Different questions → different evidence<br>• New rules apply instantly (no reprocess)<br>• Agent asks context-specific questions<br>• Every fact has KMS-signed provenance</div>
          </div>
        </div>
        <div style="margin-top:0.5rem;padding:0.4rem 0.6rem;background:var(--bg-secondary);border-radius:6px;border:1px solid var(--card-border);">
          <div style="font-size:0.62rem;font-weight:700;color:var(--text-main);margin-bottom:0.2rem;">🛡️ Evidence Strength Harness</div>
          <div style="display:flex;gap:0.4rem;flex-wrap:wrap;font-size:0.58rem;color:var(--text-muted);">
            <span style="background:rgba(99,102,241,0.1);padding:0.1rem 0.4rem;border-radius:10px;border:1px solid rgba(99,102,241,0.2);">🔐 KMS Cryptographic Provenance</span>
            <span style="background:rgba(239,68,68,0.1);padding:0.1rem 0.4rem;border-radius:10px;border:1px solid rgba(239,68,68,0.2);">⚖️ Independent OPA Verification</span>
            <span style="background:rgba(34,197,94,0.1);padding:0.1rem 0.4rem;border-radius:10px;border:1px solid rgba(34,197,94,0.2);">📎 Lineage → Source → Timestamp</span>
            <span style="background:rgba(249,115,22,0.1);padding:0.1rem 0.4rem;border-radius:10px;border:1px solid rgba(249,115,22,0.2);">🚫 No Hallucinated Evidence</span>
          </div>
        </div>
      </div>

      <style>
        .wf-flow-node:hover { background: var(--bg-secondary); }
        .wf-flow-node.wf-active { border-color: var(--accent-blue) !important; background: rgba(99,102,241,0.08); transform: scale(1.1); }
        .wf-flow-node.wf-completed { opacity: 0.6; }
        .wf-flow-node.wf-completed::after { content: '✓'; position: absolute; top: -4px; right: -4px; font-size: 0.5rem; color: var(--accent-green); }
        .wf-flow-node { position: relative; }
      </style>
    `;
  }

  // Listen for manual step clicks
  document.addEventListener('wf-goto', (e) => {
    clearInterval(autoPlayInterval);
    currentStep = e.detail;
    highlightStep(currentStep);
    // Resume auto-play after 10s of inactivity
    autoPlayInterval = setInterval(() => advanceStep(), 4000);
  });

  return {
    play: () => { autoPlayInterval = setInterval(() => advanceStep(), 4000); },
    reset: () => { clearInterval(autoPlayInterval); currentStep = -1; },
    goTo: (step) => { currentStep = step; highlightStep(step); },
  };
}

export class CRFWorkflowExplainer {
  constructor(containerId) {
    this.containerId = containerId;
    this.control = null;
  }
  mount() { this.control = render(this.containerId); return this; }
  destroy() { if (this.control) this.control.reset(); }
}

export default { render, CRFWorkflowExplainer };
