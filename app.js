import { PriorAuthAgent } from './agent.js';

document.addEventListener('DOMContentLoaded', async () => {
  const agent = new PriorAuthAgent();
  let activeFile = "rules_declaration.md";

  // Elements
  const presetContainer = document.getElementById('preset-cases-container');
  const rulesContainer = document.getElementById('rules-checkbox-container');
  const form = document.getElementById('pa-request-form');
  const outcomeCard = document.querySelector('.outcome-card');
  const outcomeBadge = document.getElementById('outcome-badge');
  const outcomeReason = document.getElementById('outcome-reason');
  const timeline = document.getElementById('trace-timeline');
  const traceCounter = document.getElementById('trace-counter');
  const letterContent = document.getElementById('letter-content');
  
  // Workspace File Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const codeViewer = document.getElementById('code-viewer');
  const codeViewerPre = document.getElementById('code-viewer-pre');
  const editorTextarea = document.getElementById('editor-textarea');
  const pdfDownloadPanel = document.getElementById('pdf-download-panel');
  const activeSkillsPills = document.getElementById('active-skills-pills');
  const discoveredSkillsCount = document.getElementById('discovered-skills-count');
  const auditConsole = document.getElementById('audit-console');

  // Input Fields
  const inputMemberId = document.getElementById('input-member-id');
  const inputPatientName = document.getElementById('input-patient-name');
  const inputSsn = document.getElementById('input-ssn');
  const inputDob = document.getElementById('input-dob');
  const inputCpt = document.getElementById('input-cpt');
  const inputIcd = document.getElementById('input-icd10');
  const inputProvider = document.getElementById('input-provider-name');
  const inputNpi = document.getElementById('input-npi');
  const inputNotes = document.getElementById('input-notes');

  // Tab navigation
  document.querySelectorAll('.main-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.main-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const view = tab.getAttribute('data-view');
      document.querySelectorAll('.view-panel').forEach(p => p.style.display = 'none');
      document.getElementById(`view-${view}`).style.display = view === 'review' ? 'grid' : 'block';
    });
  });

  // Helper for non-blocking notifications
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('visible'), 50);
    setTimeout(() => {
      toast.classList.remove('visible');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  // Load initial workspace file view
  try {
    await refreshActiveFileView();
  } catch (e) {
    console.error("Initial file view load failed:", e);
  }
  refreshSkillsPills();
  loadPolicyWorkspace();

  // Load policies into the workspace dropdown and detail view
  let workspacePolicies = [];
  
  async function loadPolicyWorkspace() {
    const select = document.getElementById('policy-select');
    try {
      const res = await fetch('/agent/policies');
      workspacePolicies = await res.json();
      select.innerHTML = '';
      workspacePolicies.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.policyId;
        opt.textContent = `${p.name} (${p.category}) — ${p.cptCodes.join(', ')}`;
        select.appendChild(opt);
      });
      if (workspacePolicies.length > 0) showWorkspacePolicy(workspacePolicies[0].policyId);
    } catch (e) {
      select.innerHTML = '<option>Failed to load policies</option>';
    }
    
    // Dropdown change
    select.addEventListener('change', () => showWorkspacePolicy(select.value));
    
    // Generate All button
    document.getElementById('btn-gen-all').addEventListener('click', async () => {
      const policyId = select.value;
      if (!policyId) return;
      const btn = document.getElementById('btn-gen-all');
      btn.disabled = true; btn.textContent = '🧠 Generating...';
      auditConsole.textContent = `Generating rules, skills, and hooks for ${policyId}...\n`;
      
      for (const action of ['rules', 'skills', 'hooks']) {
        try {
          const res = await fetch('/agent/build-for-policy', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ policyId, action })
          });
          const data = await res.json();
          auditConsole.textContent += `\n[${action.toUpperCase()}] ${data.auditLog || data.error || 'Done'}\n`;
        } catch(e) {
          auditConsole.textContent += `\n[${action.toUpperCase()}] Error: ${e.message}\n`;
        }
      }
      btn.disabled = false; btn.textContent = '🧠 Generate All';
      await refreshActiveFileView();
      refreshSkillsPills();
      showToast(`All artifacts generated for ${policyId}`, 'success');
    });
    
    // Individual generate buttons
    ['rules', 'skills', 'hooks'].forEach(action => {
      const btnId = action === 'skills' ? 'btn-gen-skills-ws' : `btn-gen-${action}`;
      document.getElementById(btnId).addEventListener('click', async () => {
        const policyId = select.value;
        if (!policyId) return;
        const btn = document.getElementById(btnId);
        btn.disabled = true;
        try {
          const res = await fetch('/agent/build-for-policy', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ policyId, action })
          });
          const data = await res.json();
          auditConsole.textContent = data.auditLog || data.error || 'Done';
          if (action === 'skills') refreshSkillsPills();
          await refreshActiveFileView();
          showToast(`${action} generated for ${policyId}`, 'success');
        } catch(e) {
          auditConsole.textContent = `Error: ${e.message}`;
        } finally { btn.disabled = false; }
      });
    });
  }

  function showWorkspacePolicy(policyId) {
    const title = document.getElementById('ws-policy-title');
    const payer = document.getElementById('ws-policy-payer');
    const body = document.getElementById('ws-policy-body');
    
    fetch(`/agent/policy-detail?id=${policyId}`).then(r => r.json()).then(p => {
      title.textContent = `${p.policyId}: ${p.policyName}`;
      payer.textContent = `${p.payer} • ${p.category}`;
      
      let html = '';
      html += '<div style="margin-bottom:0.5rem;"><strong style="font-size:0.72rem;color:var(--text-muted);">CPT CODES:</strong></div>';
      html += '<div class="policy-codes-row">';
      (p.cptCodes || []).forEach(c => {
        const desc = p.cptDescriptions ? (p.cptDescriptions[c] || '') : '';
        html += `<span class="policy-code-chip" title="${desc}">${c}</span>`;
      });
      html += '</div>';
      
      if (p.allowedIcd10 && p.allowedIcd10.length > 0) {
        html += '<div style="margin:0.5rem 0 0.3rem;"><strong style="font-size:0.72rem;color:var(--text-muted);">ICD-10:</strong></div>';
        html += '<div class="policy-codes-row">';
        p.allowedIcd10.slice(0, 8).forEach(c => { html += `<span class="policy-code-chip">${c}</span>`; });
        if (p.allowedIcd10.length > 8) html += `<span class="policy-code-chip">+${p.allowedIcd10.length - 8}</span>`;
        html += '</div>';
      }
      
      if (p.criteria && p.criteria.length > 0) {
        html += `<div style="margin:0.5rem 0 0.3rem;"><strong style="font-size:0.72rem;color:var(--text-muted);">CRITERIA (${p.criteria.length}):</strong></div>`;
        html += '<div class="policy-criteria-list">';
        p.criteria.forEach(c => {
          html += `<div class="policy-criteria-item"><span class="crit-id">${c.id}</span><span>${c.description}</span></div>`;
        });
        html += '</div>';
      }
      
      body.innerHTML = html;
    }).catch(() => { body.innerHTML = '<p>Error loading policy detail.</p>'; });
  }

  // 1. (loadRegoSource removed — agent review handled by Python backend)

  // 2. Fetch and display file content in Workspace Editor / Pre previewer
  async function refreshActiveFileView() {
    codeViewerPre.style.display = "none";
    editorTextarea.style.display = "none";
    if (pdfDownloadPanel) pdfDownloadPanel.style.display = "none";
    
    // Handle PDF tab specially
    const isPdf = activeFile.endsWith('.pdf');
    if (isPdf) {
      if (pdfDownloadPanel) pdfDownloadPanel.style.display = "flex";
      return;
    }

    try {
      const res = await fetch(`./${activeFile}?update=${Date.now()}`);
      if (res.ok) {
        const text = await res.text();
        // JSON files show in pre (read-only, formatted)
        if (activeFile.endsWith('.json') || activeFile.endsWith('.rego') || activeFile.endsWith('.js')) {
          codeViewerPre.style.display = "block";
          if (activeFile.endsWith('.json')) {
            try { codeViewer.textContent = JSON.stringify(JSON.parse(text), null, 2); }
            catch(_) { codeViewer.textContent = text; }
          } else {
            codeViewer.textContent = text;
          }
        } else {
          editorTextarea.style.display = "block";
          editorTextarea.value = text;
        }
      } else {
        codeViewerPre.style.display = "block";
        codeViewer.textContent = `File not found: ${activeFile}`;
      }
    } catch (e) {
      codeViewerPre.style.display = "block";
      codeViewer.textContent = `Error loading: ${e.message}`;
    }
  }

  // 3. Render active registered skills pills in the UI
  function refreshSkillsPills() {
    activeSkillsPills.innerHTML = '';
    const skillsList = Object.keys(agent.skills);
    discoveredSkillsCount.textContent = `${skillsList.length} Active Skills`;

    skillsList.forEach(name => {
      const pill = document.createElement('span');
      pill.className = 'skill-pill';
      if (name === "VerifyNpiStatusSkill") {
        pill.className = 'skill-pill new-injected';
      }
      pill.textContent = name;
      activeSkillsPills.appendChild(pill);
    });
  }

  // 4. File Workspace Tab Event Handlers
  tabButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFile = btn.getAttribute('data-file');
      await refreshActiveFileView();
    });
  });

  // 5. (Save/compile removed — replaced by per-policy AI generation)
  // 6. (Reset removed — policies are managed through JSON files)

  // 6.5. (Removed — per-policy action buttons handle this now)

  // 6.6. (Removed — per-policy action buttons handle this now)

  // 6.7. Natural Language Skill Creator
  const nlSkillInput = document.getElementById('nl-skill-input');
  const btnCreateSkill = document.getElementById('btn-create-skill');
  const nlSkillStatus = document.getElementById('nl-skill-status');

  btnCreateSkill.addEventListener('click', async () => {
    const description = nlSkillInput.value.trim();
    if (!description) {
      nlSkillStatus.textContent = "Please describe a skill in plain English.";
      nlSkillStatus.className = "nl-skill-status error";
      return;
    }

    btnCreateSkill.disabled = true;
    btnCreateSkill.textContent = "🧠 Generating...";
    nlSkillStatus.textContent = "Parsing natural language and generating skill contract...";
    nlSkillStatus.className = "nl-skill-status processing";

    try {
      const res = await fetch('/generate-skill', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description })
      });

      if (res.ok) {
        const data = await res.json();

        // Show audit output
        auditConsole.textContent = data.auditLogs.join('\n');

        // Reload skills into agent
        await agent.reloadSkills();
        refreshSkillsPills();

        // Refresh editor if viewing skills_declaration.md
        if (activeFile === 'skills_declaration.md' || activeFile === 'skills.js') {
          await refreshActiveFileView();
        }

        nlSkillStatus.textContent = `✓ Skill "${data.skillName}" registered.\n  Inputs: ${data.inputs.join(', ')}\n  Outputs: ${data.outputs.join(', ')}`;
        nlSkillStatus.className = "nl-skill-status success";
        nlSkillInput.value = '';
        showToast(`Skill "${data.skillName}" created and compiled!`, "success");
      } else {
        let errText = '';
        try {
          const errData = await res.json();
          errText = errData.error || errData.message || JSON.stringify(errData);
          if (errData.auditLogs) {
            auditConsole.textContent = errData.auditLogs.join('\n');
          }
        } catch (_) {
          errText = await res.text().catch(() => `HTTP ${res.status}`);
        }
        nlSkillStatus.textContent = `Error (${res.status}): ${errText}`;
        nlSkillStatus.className = "nl-skill-status error";
        auditConsole.textContent = `Skill generation error (${res.status}):\n${errText}`;
        showToast(`Skill generation failed: ${errText}`, "error");
      }
    } catch (e) {
      nlSkillStatus.textContent = `Network error: ${e.message}\nMake sure the server is running (python3 server.py)`;
      nlSkillStatus.className = "nl-skill-status error";
      auditConsole.textContent = `Network error: ${e.message}\nIs the server running? Try: python3 server.py`;
      showToast(`Network error: ${e.message}`, "error");
    } finally {
      btnCreateSkill.disabled = false;
      btnCreateSkill.textContent = "🧠 Generate & Register Skill";
    }
  });

  // 7. Preset Case loaders (fetched from Python backend)
  async function renderPresets() {
    try {
      const res = await fetch('/agent/cases');
      const cases = await res.json();
      presetContainer.innerHTML = '';
      
      cases.forEach(c => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'case-btn';
        btn.innerHTML = `<h3>${c.title.split('—')[0].trim()}</h3><p style="font-size:0.6rem;color:var(--text-muted);margin-top:2px;">${c.category}</p>`;
        btn.addEventListener('click', () => {
          document.querySelectorAll('.case-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          loadCaseToForm(c);
        });
        presetContainer.appendChild(btn);
      });
      
      // Load first case by default
      if (cases.length > 0) {
        presetContainer.querySelector('.case-btn').classList.add('active');
        loadCaseToForm(cases[0]);
      }
    } catch(e) {
      console.error('Failed to load cases:', e);
    }
  }

  function loadCaseToForm(caseObj) {
    const req = caseObj.request;
    inputMemberId.value = req.memberId;
    inputPatientName.value = req.patientName;
    inputSsn.value = req.patientSsn;
    inputDob.value = req.patientDob;
    inputCpt.value = req.cptCode;
    inputIcd.value = req.icd10Code;
    inputProvider.value = req.providerName;
    inputNpi.value = req.providerNpi;
    inputNotes.value = req.clinicalNotes;

    // Clear all output panels when loading a new case
    outcomeCard.className = 'glass-card outcome-card pending';
    outcomeBadge.textContent = 'Ready to Run';
    outcomeReason.textContent = 'Case loaded: ' + caseObj.title;
    letterContent.textContent = 'Awaiting review execution to draft notice letter...';
    timeline.innerHTML = `<div class="timeline-empty-state"><span class="timeline-empty-icon">🤖</span><p>Execution trace outputs will display here.</p></div>`;
    traceCounter.textContent = '0 events logged';
    
    // Reset pipeline flow
    [1, 2, 3, 4, 5].forEach(s => {
      const el = document.getElementById(`flow-step-${s}`);
      el.className = 'flow-step';
      const disc = el.querySelector('.flow-step-disclosure');
      if (disc) disc.classList.remove('disc-active');
    });
    
    // Reset disclosure panel
    document.getElementById('disclosure-stage-badge').textContent = 'Idle';
    document.getElementById('disclosure-stage-badge').className = 'disclosure-badge';
    document.getElementById('disc-disclosed').textContent = '—';
    document.getElementById('disc-withheld').textContent = '—';
    
    // Hide AI panels
    const aiCard = document.getElementById('ai-reasoning-card');
    if (aiCard) aiCard.style.display = 'none';
    const aiInsights = document.getElementById('ai-insights-panel');
    if (aiInsights) aiInsights.style.display = 'none';
    const aiModelBadge = document.getElementById('ai-model-badge');
    if (aiModelBadge) { aiModelBadge.textContent = ''; aiModelBadge.className = 'badge'; }
    const npiBadge = document.getElementById('npi-status-badge');
    if (npiBadge) { npiBadge.textContent = ''; npiBadge.className = 'badge'; }
    
    // Remove any note summary/quality popups
    document.querySelectorAll('.notes-summary-popup').forEach(el => el.remove());
  }

  // 8. Compact Rules Checkbox toggles — inline pills
  function renderRulesToggles() {
    rulesContainer.innerHTML = '';
    Object.entries(agent.rules).forEach(([key, rule]) => {
      const item = document.createElement('div');
      item.className = 'rule-item';
      // Show short name (e.g., "PHI" instead of "PHI Redaction Rule")
      const shortName = rule.name.replace(' Rule', '').replace('Clinical ', '');
      item.innerHTML = `
        <div class="rule-checkbox-wrapper">
          <input type="checkbox" id="chk-${key}" ${rule.enabled ? 'checked' : ''}>
        </div>
        <div class="rule-info">
          <h4>${shortName}</h4>
        </div>
      `;
      const chk = item.querySelector('input');
      chk.addEventListener('change', () => {
        agent.rules[key].enabled = chk.checked;
      });
      rulesContainer.appendChild(item);
    });
  }

  // 9. Render Timeline Trace Output (Python backend format: {ts, type, name, msg, status})
  function renderTrace(traceLog) {
    timeline.innerHTML = '';
    traceCounter.textContent = `${traceLog.length} events`;

    if (!traceLog || traceLog.length === 0) {
      timeline.innerHTML = '<div class="timeline-empty-state"><span class="timeline-empty-icon">🤖</span><p>Trace outputs will display here.</p></div>';
      return;
    }

    traceLog.forEach(step => {
      const item = document.createElement('div');
      const isDisclosure = step.name === 'Progressive Disclosure';
      const isAiPowered = step.name && step.name.includes('ClinicalNLP');
      item.className = `timeline-item ${step.status}${isDisclosure ? ' disclosure' : ''}${isAiPowered ? ' ai-powered' : ''}`;

      const badgeClass = isDisclosure ? 'type-disclosure' : `type-${step.type}`;
      const displayType = isDisclosure ? 'disclosure' : step.type;
      const ts = step.ts ? new Date(step.ts) : new Date();
      const timeStr = ts.toLocaleTimeString() + '.' + String(ts.getMilliseconds()).padStart(3, '0');

      item.innerHTML = `
        <div class="timeline-dot"></div>
        <div class="item-meta">
          <span class="item-type-badge ${badgeClass}">${displayType}</span>
          <span class="item-time">${timeStr}</span>
        </div>
        <div class="item-title">${step.name || ''}</div>
        <div class="item-msg">${step.msg || step.message || ''}</div>
      `;
      timeline.appendChild(item);
    });
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&gt;");
  }

  // 10. Pipeline execution submission — calls Python backend
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const request = {
      memberId: inputMemberId.value.trim(),
      patientName: inputPatientName.value.trim(),
      patientSsn: inputSsn.value.trim(),
      patientDob: inputDob.value.trim(),
      cptCode: inputCpt.value.trim(),
      icd10Code: inputIcd.value.trim(),
      providerName: inputProvider.value.trim(),
      providerNpi: inputNpi.value.trim(),
      clinicalNotes: inputNotes.value.trim()
    };

    // Reset UI
    [1,2,3,4,5].forEach(s => {
      const el = document.getElementById(`flow-step-${s}`);
      el.className = 'flow-step';
      const disc = el.querySelector('.flow-step-disclosure');
      if (disc) disc.classList.remove('disc-active');
    });
    outcomeBadge.textContent = 'Processing...';
    outcomeReason.textContent = 'Calling Python agent backend...';
    letterContent.textContent = '';
    timeline.innerHTML = '<div class="timeline-empty-state">🤖 Agent processing...</div>';
    document.getElementById('flow-step-1').className = 'flow-step active';

    const aiModeEnabled = document.getElementById('ai-mode-toggle').checked;

    try {
      const res = await fetch('/agent/run-review', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ request, useAI: aiModeEnabled })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const outcome = await res.json();

      // Update outcome card
      outcomeCard.className = `glass-card outcome-card ${outcome.status}`;
      outcomeBadge.textContent = outcome.decision;
      outcomeReason.textContent = `${outcome.policyName} (${outcome.category}) — ${outcome.reason}`;
      letterContent.textContent = outcome.notice;

      // Render trace
      renderTrace(outcome.trace);

      // Update pipeline flow from trace
      const traceStages = outcome.trace.filter(t => t.name === 'Progressive Disclosure');
      let maxStage = 0;
      traceStages.forEach(t => {
        const m = t.msg.match(/Stage (\d)/);
        if (m) maxStage = Math.max(maxStage, parseInt(m[1]));
      });
      // Light up stages
      for (let s = 1; s <= 5; s++) {
        const el = document.getElementById(`flow-step-${s}`);
        if (s <= maxStage) {
          el.className = outcome.status === 'approved' || s < maxStage ? 'flow-step passed' : 'flow-step warning';
        }
      }
      // If there were failed criteria, mark stage 4 as warning
      if (outcome.status !== 'approved') {
        document.getElementById('flow-step-4').className = 'flow-step warning';
        document.getElementById('flow-step-5').className = 'flow-step warning';
      }

      // Update NPI badge
      const npiBadge = document.getElementById('npi-status-badge');
      npiBadge.textContent = '';
      npiBadge.className = 'badge';

      // AI model badge
      const aiModelBadge = document.getElementById('ai-model-badge');
      if (aiModeEnabled) {
        aiModelBadge.textContent = '🧠 ClinicalNLP';
        aiModelBadge.className = 'badge badge-ai';
      } else {
        aiModelBadge.textContent = '';
        aiModelBadge.className = 'badge';
      }

      // Update disclosure panel
      const discBadge = document.getElementById('disclosure-stage-badge');
      const discDisclosed = document.getElementById('disc-disclosed');
      const discWithheld = document.getElementById('disc-withheld');
      discBadge.textContent = 'Complete';
      discBadge.className = 'disclosure-badge complete';
      discDisclosed.textContent = `Policy: ${outcome.policyUsed}`;
      discWithheld.textContent = 'None — all disclosed';

      // Show AI evidence panel if available
      const aiReasoningCard = document.getElementById('ai-reasoning-card');
      if (aiModeEnabled && outcome.evidence) {
        aiReasoningCard.style.display = 'block';
        const aiStatusBadge = document.getElementById('ai-status-badge');
        const aiReasoningText = document.getElementById('ai-reasoning-text');
        aiStatusBadge.textContent = 'Complete';
        aiStatusBadge.className = 'badge badge-success';
        aiReasoningText.textContent = typeof outcome.evidence === 'string' ? outcome.evidence : JSON.stringify(outcome.evidence, null, 2);
      } else {
        aiReasoningCard.style.display = 'none';
      }

      showToast(`Decision: ${outcome.decision}`, outcome.status === 'approved' ? 'success' : 'info');

    } catch (e) {
      outcomeBadge.textContent = 'Error';
      outcomeReason.textContent = e.message;
      showToast(`Agent error: ${e.message}`, 'error');
    }
  });

  // Init UI
  await renderPresets();
  renderRulesToggles();

  // ═══════════════════════════════════════════════════════════════
  // AI FEATURES (All powered by ClinicalNLP Engine via /ai-chat)
  // ═══════════════════════════════════════════════════════════════

  // Helper: Call the AI chat endpoint
  async function callAI(prompt, systemContext = '', mode = 'chat') {
    const res = await fetch('/ai-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, systemContext, mode })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    return data.response;
  }

  // Get current form context for AI features
  function getCurrentCaseContext() {
    return `Patient: ${inputPatientName.value}, Member: ${inputMemberId.value}
CPT Code: ${inputCpt.value}, ICD-10: ${inputIcd.value}
Provider: ${inputProvider.value}, NPI: ${inputNpi.value}
Clinical Notes: ${inputNotes.value}
Current Decision: ${outcomeBadge.textContent}
Decision Reason: ${outcomeReason.textContent}`;
  }

  // FEATURE 1: AI-Powered Notice Letter Rewrite
  document.getElementById('btn-ai-letter').addEventListener('click', async () => {
    const btn = document.getElementById('btn-ai-letter');
    btn.disabled = true;
    btn.textContent = '🧠 Writing...';
    try {
      const result = await callAI(
        `Rewrite this prior authorization decision letter to be warm, empathetic, and in plain language that a patient can understand. Keep all factual content but make it human and clear. Include the policy reference. Here is the current letter:\n\n${letterContent.textContent}`,
        getCurrentCaseContext(),
        'letter'
      );
      letterContent.textContent = result;
      showToast('🧠 Letter rewritten with empathetic AI tone', 'success');
    } catch (e) {
      showToast(`AI letter rewrite failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = '🧠 Rewrite with AI';
    }
  });

  // FEATURE 5: Appeal Reasoning Generator
  document.getElementById('btn-appeal-reason').addEventListener('click', async () => {
    const btn = document.getElementById('btn-appeal-reason');
    btn.disabled = true;
    btn.textContent = '📋 Generating...';
    try {
      const result = await callAI(
        `This prior authorization case was escalated/denied. Generate a concise appeal guide for the provider explaining:
1. What specific documentation is missing
2. What criteria were not met
3. Exactly what the provider should submit to get approval
4. Suggested language for the appeal letter

Be specific and actionable.`,
        getCurrentCaseContext(),
        'appeal'
      );
      letterContent.textContent = `═══ APPEAL GUIDE ═══\n\n${result}\n\n═══ ORIGINAL NOTICE ═══\n\n${letterContent.textContent}`;
      showToast('📋 Appeal guide generated', 'success');
    } catch (e) {
      showToast(`Appeal guide failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = '📋 Generate Appeal Guide';
    }
  });

  // FEATURE 6: Clinical Notes Summarizer
  document.getElementById('btn-summarize-notes').addEventListener('click', async () => {
    const btn = document.getElementById('btn-summarize-notes');
    btn.disabled = true;
    btn.textContent = '📝 Summarizing...';
    try {
      const notes = inputNotes.value;
      const result = await callAI(
        `Summarize these clinical notes in exactly 3 bullet points for quick reviewer triage. Each bullet should capture a key clinical fact relevant to prior authorization:\n\n${notes}`,
        `CPT Code: ${inputCpt.value}, ICD-10: ${inputIcd.value}`,
        'summarize'
      );
      showToast('📝 Notes summarized', 'success');
      // Show summary above the notes field
      const summaryDiv = document.createElement('div');
      summaryDiv.className = 'notes-summary-popup';
      summaryDiv.innerHTML = `<strong>📝 AI Summary:</strong><br>${result.replace(/\n/g, '<br>')}`;
      const notesField = document.getElementById('input-notes');
      if (notesField.parentElement.querySelector('.notes-summary-popup')) {
        notesField.parentElement.querySelector('.notes-summary-popup').remove();
      }
      notesField.parentElement.insertBefore(summaryDiv, notesField);
    } catch (e) {
      showToast(`Summarize failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = '📝 Summarize Notes';
    }
  });

  // FEATURE 2: Clinical Notes Quality Scoring
  document.getElementById('btn-quality-score').addEventListener('click', async () => {
    const btn = document.getElementById('btn-quality-score');
    btn.disabled = true;
    btn.textContent = '📊 Scoring...';
    try {
      const notes = inputNotes.value;
      const result = await callAI(
        `Score these clinical notes for completeness on a scale of 1-10 for prior authorization review. Respond with ONLY a JSON object:
{"score": <1-10>, "missing": ["<list of missing elements>"], "suggestion": "<one-sentence improvement suggestion>"}

Clinical notes to score:\n${notes}`,
        `CPT Code: ${inputCpt.value}. Required: symptom duration, conservative therapy history, objective findings, imaging results.`,
        'score'
      );
      // Parse the score
      let scoreData;
      try {
        const jsonStr = result.substring(result.indexOf('{'), result.lastIndexOf('}') + 1);
        scoreData = JSON.parse(jsonStr);
      } catch (_) {
        scoreData = { score: 5, missing: ['Could not parse'], suggestion: result.substring(0, 100) };
      }
      // Display inline
      const summaryDiv = document.createElement('div');
      summaryDiv.className = 'notes-summary-popup quality-popup';
      const scoreColor = scoreData.score >= 7 ? 'var(--accent-green)' : scoreData.score >= 4 ? 'var(--accent-orange)' : 'var(--accent-red)';
      summaryDiv.innerHTML = `<strong style="color:${scoreColor}">📊 Quality: ${scoreData.score}/10</strong><br>` +
        (scoreData.missing?.length ? `<span style="color:var(--text-muted)">Missing: ${scoreData.missing.join(', ')}</span><br>` : '') +
        `<span style="color:var(--text-muted)">${scoreData.suggestion || ''}</span>`;
      const notesField = document.getElementById('input-notes');
      if (notesField.parentElement.querySelector('.quality-popup')) {
        notesField.parentElement.querySelector('.quality-popup').remove();
      }
      notesField.parentElement.insertBefore(summaryDiv, notesField);
      showToast(`📊 Quality score: ${scoreData.score}/10`, scoreData.score >= 7 ? 'success' : 'info');
    } catch (e) {
      showToast(`Quality score failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = '📊 Score Notes Quality';
    }
  });

  // FEATURE 7: AVI Conversational Chat
  const aviFab = document.getElementById('avi-fab');
  const aviPanel = document.getElementById('avi-chat-panel');
  const aviClose = document.getElementById('avi-close');
  const aviInput = document.getElementById('avi-input');
  const aviSend = document.getElementById('avi-send');
  const aviMessages = document.getElementById('avi-messages');

  const AVI_SYSTEM_PROMPT = `You are AVI, a clinical intelligence assistant built into the PriorAuthAI Governance Console. You help users understand prior authorization cases, policies, and decisions.

Your capabilities within this tool:
1. Explain why a case was approved or escalated
2. Identify what documentation is missing for approval
3. Explain what specific rules (RULE-01 through RULE-05) do
4. Describe how the 5-stage progressive disclosure pipeline works
5. Help interpret CPT codes, ICD-10 codes, and clinical criteria
6. Suggest what providers should include in clinical notes
7. Explain the difference between regex and ClinicalNLP extraction

IMPORTANT REFERENCE DATA (use ONLY this data for code lookups — do NOT guess or hallucinate code meanings):
- CPT 73721 = MRI, any joint of lower extremity (knee), without contrast
- CPT J0135 = Injection, adalimumab (Humira), 20mg subcutaneous
- ICD-10 M25.561 = Pain in right knee
- ICD-10 M25.562 = Pain in left knee
- ICD-10 M25.569 = Pain in unspecified knee
- ICD-10 S83.206A = Unspecified tear of lateral meniscus, initial encounter
- ICD-10 M05.79 = Rheumatoid arthritis with rheumatoid factor, unspecified site
- ICD-10 M06.9 = Rheumatoid arthritis, unspecified

RULES in this system:
- RULE-01 (PHI Redaction): Scrubs patient names/SSN/DOB from logs
- RULE-02 (Clinical Conservatism): Never auto-deny; escalate uncertain cases to human
- RULE-03 (Citation Compulsory): Every notice must cite the policy ID
- RULE-04 (Code Match): Validates CPT/ICD format and guideline alignment
- RULE-05 (Plain Language): Translates medical acronyms for patients

POLICIES in this system:
- POL-RAD-402: MRI Knee — requires 6+ weeks symptoms, 6+ weeks PT, objective findings, radiographs
- POL-PHARM-809: Humira/Biologic — requires RA diagnosis, 12+ weeks DMARD failure, rheumatologist consult

Keep answers concise (3-5 sentences max). Be direct and clinical. If you don't know something, say so — never fabricate medical code definitions. You ARE AVI, this tool's assistant.`;

  aviFab.addEventListener('click', () => {
    aviPanel.style.display = aviPanel.style.display === 'none' ? 'flex' : 'none';
    if (aviPanel.style.display === 'flex') aviInput.focus();
  });
  aviClose.addEventListener('click', () => { aviPanel.style.display = 'none'; });

  async function sendAviMessage() {
    const msg = aviInput.value.trim();
    if (!msg) return;
    
    // Show user message
    const userBubble = document.createElement('div');
    userBubble.className = 'avi-msg avi-user';
    userBubble.innerHTML = `<p>${escapeHtml(msg)}</p>`;
    aviMessages.appendChild(userBubble);
    aviInput.value = '';
    
    // Show typing indicator
    const typing = document.createElement('div');
    typing.className = 'avi-msg avi-bot';
    typing.innerHTML = '<p class="avi-typing">AVI is thinking...</p>';
    aviMessages.appendChild(typing);
    aviMessages.scrollTop = aviMessages.scrollHeight;
    
    try {
      const response = await callAI(msg, AVI_SYSTEM_PROMPT + '\n\nCURRENT CASE CONTEXT:\n' + getCurrentCaseContext());
      typing.remove();
      const botBubble = document.createElement('div');
      botBubble.className = 'avi-msg avi-bot';
      botBubble.innerHTML = `<p>${response.replace(/\n/g, '<br>')}</p>`;
      aviMessages.appendChild(botBubble);
    } catch (e) {
      typing.remove();
      const errBubble = document.createElement('div');
      errBubble.className = 'avi-msg avi-bot';
      errBubble.innerHTML = `<p style="color:var(--accent-red);">ClinicalNLP Engine offline: ${e.message}</p>`;
      aviMessages.appendChild(errBubble);
    }
    aviMessages.scrollTop = aviMessages.scrollHeight;
  }

  aviSend.addEventListener('click', sendAviMessage);
  aviInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendAviMessage(); });

});
