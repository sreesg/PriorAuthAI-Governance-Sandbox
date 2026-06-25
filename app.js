import { PRESET_CASES } from './cases.js';
import { PriorAuthAgent } from './agent.js';

document.addEventListener('DOMContentLoaded', async () => {
  const agent = new PriorAuthAgent();
  let activeCaseId = "case-1";
  let activeFile = "rules_declaration.md"; // Default editor active view
  let regoSourceText = "";

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
  const btnSaveCompile = document.getElementById('btn-save-compile');
  const btnResetWorkspace = document.getElementById('btn-reset-workspace');
  const activeSkillsPills = document.getElementById('active-skills-pills');
  const discoveredSkillsCount = document.getElementById('discovered-skills-count');
  const auditConsole = document.getElementById('audit-console');
  const jurorStatusBadge = document.getElementById('juror-status-badge');

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

  // Load and cache rules.rego initially for execution
  try {
    await loadRegoSource();
  } catch (e) {
    console.error("Initial rules.rego load failed:", e);
  }
  try {
    await refreshActiveFileView();
  } catch (e) {
    console.error("Initial file view load failed:", e);
  }
  refreshSkillsPills();

  // 1. Fetch OPA rules.rego source code
  async function loadRegoSource() {
    try {
      const res = await fetch(`./rules.rego?update=${Date.now()}`);
      if (res.ok) {
        regoSourceText = await res.text();
      }
    } catch (e) {
      console.error("Failed to load rules.rego source:", e);
      regoSourceText = "";
    }
  }

  // 2. Fetch and display file content in Workspace Editor / Pre previewer
  async function refreshActiveFileView() {
    codeViewerPre.style.display = "none";
    editorTextarea.style.display = "none";
    pdfDownloadPanel.style.display = "none";
    
    // Handle PDF tab specially
    const isPdf = activeFile.endsWith('.pdf');
    if (isPdf) {
      pdfDownloadPanel.style.display = "flex";
      btnSaveCompile.disabled = true;
      btnSaveCompile.classList.add('disabled');
      btnSaveCompile.textContent = "🔒 PDF View";
      return;
    }
    
    // Declarations are editable Markdown textareas, compiled files are read-only pre components
    const isEditable = activeFile.endsWith('.md');
    
    // Disable compile button if viewing compiled files
    btnSaveCompile.disabled = !isEditable;
    if (!isEditable) {
      btnSaveCompile.classList.add('disabled');
      btnSaveCompile.textContent = "🔒 Compiled Code View";
    } else {
      btnSaveCompile.classList.remove('disabled');
      btnSaveCompile.textContent = "💾 Save & Compile Policies";
    }

    try {
      const res = await fetch(`./${activeFile}?update=${Date.now()}`);
      if (res.ok) {
        const text = await res.text();
        if (isEditable) {
          editorTextarea.style.display = "block";
          editorTextarea.value = text;
        } else {
          codeViewerPre.style.display = "block";
          codeViewer.textContent = text;
        }
      } else {
        if (isEditable) {
          editorTextarea.style.display = "block";
          editorTextarea.value = `Error loading file ${activeFile}.`;
        } else {
          codeViewerPre.style.display = "block";
          codeViewer.textContent = `Error loading file ${activeFile}.`;
        }
      }
    } catch (e) {
      console.error(e);
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

  // 5. Save and compile human-readable declarations
  btnSaveCompile.addEventListener('click', async () => {
    btnSaveCompile.disabled = true;
    btnSaveCompile.textContent = "⚙️ SkillJuror Auditing...";
    jurorStatusBadge.textContent = "Juror: AUDITING...";
    jurorStatusBadge.className = "badge badge-orange";
    
    try {
      const res = await fetch('/save-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: json_payload_string()
      });

      if (res.ok) {
        const data = await res.json();
        
        // Print SkillJuror validation results in console
        auditConsole.textContent = data.auditLogs.join('\n');
        
        // Reload agent memory
        await loadRegoSource();
        await agent.reloadSkills();
        
        // Update components
        refreshSkillsPills();
        showToast("Success: File compiled & SkillJuror audited!", "success");
        jurorStatusBadge.textContent = "Juror: SECURED";
        jurorStatusBadge.className = "badge badge-purple";
      } else {
        showToast("Failed to save and compile. Check server log.", "error");
        jurorStatusBadge.textContent = "Juror: ERROR";
        jurorStatusBadge.className = "badge badge-red";
      }
    } catch (e) {
      showToast(`Network error compiling: ${e.message}`, "error");
      jurorStatusBadge.textContent = "Juror: OFFLINE";
      jurorStatusBadge.className = "badge badge-red";
    } finally {
      btnSaveCompile.disabled = false;
      btnSaveCompile.textContent = "💾 Save & Compile Policies";
    }
  });

  function json_payload_string() {
    return JSON.stringify({
      filename: activeFile,
      content: editorTextarea.value
    });
  }

  // 6. Reset Workspace templates
  btnResetWorkspace.addEventListener('click', async () => {
    btnResetWorkspace.disabled = true;
    try {
      const res = await fetch('/reset-workspace', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        
        // Reset local agent instances
        delete agent.skills.VerifyNpiStatusSkill;
        await loadRegoSource();
        await refreshActiveFileView();
        refreshSkillsPills();
        
        auditConsole.textContent = "Workspace restored to initial defaults.";
        showToast("Workspace templates reset successfully.", "success");
        jurorStatusBadge.textContent = "Juror: ACTIVE";
        jurorStatusBadge.className = "badge badge-purple";
      }
    } catch (e) {
      showToast(`Error resetting workspace: ${e.message}`, "error");
    } finally {
      btnResetWorkspace.disabled = false;
    }
  });

  // 6.5. Natural Language Skill Creator
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

  // 7. Preset Case loaders
  function renderPresets() {
    presetContainer.innerHTML = '';
    PRESET_CASES.forEach(c => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `case-btn ${c.id === activeCaseId ? 'active' : ''}`;
      btn.innerHTML = `<h3>Case ${c.id.split('-')[1]}</h3>`;
      
      btn.addEventListener('click', () => {
        activeCaseId = c.id;
        document.querySelectorAll('.case-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        loadCaseToForm(c);
      });
      presetContainer.appendChild(btn);
    });
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

    outcomeCard.className = 'glass-card outcome-card pending';
    outcomeBadge.textContent = 'Ready to Run';
    outcomeReason.textContent = 'Case loaded: ' + caseObj.title;
  }

  // 8. Compact Rules Checkbox toggles
  function renderRulesToggles() {
    rulesContainer.innerHTML = '';
    Object.entries(agent.rules).forEach(([key, rule]) => {
      const item = document.createElement('div');
      item.className = 'rule-item';
      item.innerHTML = `
        <div class="rule-checkbox-wrapper">
          <input type="checkbox" id="chk-${key}" ${rule.enabled ? 'checked' : ''}>
        </div>
        <div class="rule-info">
          <h4>${rule.name}</h4>
        </div>
      `;
      const chk = item.querySelector('input');
      chk.addEventListener('change', () => {
        agent.rules[key].enabled = chk.checked;
      });
      rulesContainer.appendChild(item);
    });
  }

  // 9. Render Timeline Trace Output
  function renderTrace(traceLog) {
    timeline.innerHTML = '';
    traceCounter.textContent = `${traceLog.length} events logged`;

    if (traceLog.length === 0) {
      timeline.innerHTML = `
        <div class="timeline-empty-state">
          <span class="timeline-empty-icon">🤖</span>
          <p>Execution trace outputs will display here.</p>
        </div>
      `;
      return;
    }

    // Update the disclosure panel with the final progressive disclosure summary
    updateDisclosurePanel(traceLog);

    traceLog.forEach(step => {
      const item = document.createElement('div');
      // Special styling for Progressive Disclosure events
      const isDisclosure = step.name === 'Progressive Disclosure';
      item.className = `timeline-item ${step.status}${isDisclosure ? ' disclosure' : ''}`;
      
      const badgeClass = isDisclosure ? 'type-disclosure' : `type-${step.type}`;
      const displayType = isDisclosure ? 'disclosure' : step.type;
      const timestamp = new Date(step.timestamp);
      const timeStr = timestamp.toLocaleTimeString() + '.' + String(timestamp.getMilliseconds()).padStart(3, '0');

      item.innerHTML = `
        <div class="timeline-dot"></div>
        <div class="item-meta">
          <span class="item-type-badge ${badgeClass}">${displayType}</span>
          <span class="item-time">${timeStr}</span>
        </div>
        <div class="item-title">${step.name}</div>
        <div class="item-msg">${step.message}</div>
        ${step.details ? `
          <div class="item-expand">
            <pre>${escapeHtml(JSON.stringify(step.details, null, 2))}</pre>
          </div>
        ` : ''}
      `;

      if (step.details) {
        item.addEventListener('click', () => {
          item.classList.toggle('expanded');
        });
      }

      timeline.appendChild(item);
    });
  }

  // Update the Progressive Disclosure status panel based on trace events
  function updateDisclosurePanel(traceLog) {
    const discPanel = document.getElementById('disclosure-panel');
    const discBadge = document.getElementById('disclosure-stage-badge');
    const discDisclosed = document.getElementById('disc-disclosed');
    const discWithheld = document.getElementById('disc-withheld');
    
    const pdEvents = traceLog.filter(t => t.name === 'Progressive Disclosure');
    
    if (pdEvents.length === 0) {
      discBadge.textContent = 'Idle';
      discBadge.className = 'disclosure-badge';
      discDisclosed.textContent = '—';
      discWithheld.textContent = '—';
      return;
    }

    // Get the final summary event
    const finalEvent = pdEvents[pdEvents.length - 1];
    const isComplete = finalEvent.message.includes('Pipeline complete');
    
    if (isComplete) {
      discBadge.textContent = 'Complete';
      discBadge.className = 'disclosure-badge complete';
      discDisclosed.textContent = 'All 5 rules + guidelines + Rego policies';
      discWithheld.textContent = 'None — fully disclosed';
    } else {
      // Show the last active stage
      const lastStageEvent = pdEvents.filter(e => e.message.includes('Stage')).pop();
      if (lastStageEvent) {
        const stageMatch = lastStageEvent.message.match(/Stage (\d)/);
        const stageNum = stageMatch ? stageMatch[1] : '?';
        discBadge.textContent = `Stage ${stageNum}`;
        discBadge.className = 'disclosure-badge active';
        
        // Build disclosed/withheld lists
        const allRules = ['RULE-01', 'RULE-02', 'RULE-03', 'RULE-04', 'RULE-05'];
        const disclosed = [];
        const withheld = [];
        
        pdEvents.forEach(e => {
          allRules.forEach(r => {
            if (e.message.includes(r) && !disclosed.includes(r)) {
              disclosed.push(r);
            }
          });
          if (e.message.includes('Rego')) disclosed.push('Rego');
          if (e.message.includes('Guidelines')) disclosed.push('Guidelines');
        });
        
        allRules.forEach(r => {
          if (!disclosed.includes(r)) withheld.push(r);
        });
        if (!disclosed.includes('Rego')) withheld.push('Rego Policy');
        
        discDisclosed.textContent = disclosed.join(', ') || '—';
        discWithheld.textContent = withheld.join(', ') || 'None';
      }
    }
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&gt;");
  }

  // 10. Pipeline execution submission — Animated Progressive Disclosure
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

    // Reset all UI state
    const steps = [1, 2, 3, 4, 5];
    steps.forEach(s => {
      const el = document.getElementById(`flow-step-${s}`);
      el.className = "flow-step";
      el.querySelector('.flow-step-disclosure').className = 'flow-step-disclosure';
    });

    outcomeBadge.textContent = 'Reviewing...';
    outcomeReason.textContent = 'Progressive disclosure in progress...';
    letterContent.textContent = '';
    timeline.innerHTML = '';

    // Reset disclosure panel
    const discBadge = document.getElementById('disclosure-stage-badge');
    const discDisclosed = document.getElementById('disc-disclosed');
    const discWithheld = document.getElementById('disc-withheld');
    discBadge.textContent = 'Starting';
    discBadge.className = 'disclosure-badge active';
    discDisclosed.textContent = '—';
    discWithheld.textContent = 'RULE-01, RULE-02, RULE-03, RULE-04, RULE-05, Rego, Guidelines';

    // Disclosure animation data per stage
    const disclosureStages = [
      { stage: 1, label: 'Stage 1: Intake', disclosed: 'RULE-01 (PHI)', withheld: 'RULE-02, RULE-03, RULE-04, RULE-05, Rego, Guidelines' },
      { stage: 2, label: 'Stage 2: Coverage', disclosed: 'RULE-01, RULE-04, Guidelines (CPT-specific)', withheld: 'RULE-02, RULE-03, RULE-05, Rego Policy' },
      { stage: 3, label: 'Stage 3: Evidence', disclosed: 'RULE-01, RULE-04, Guidelines, Clinical Evidence', withheld: 'RULE-02, RULE-03, RULE-05, Rego Policy' },
      { stage: 4, label: 'Stage 4: Evaluation', disclosed: 'RULE-01, RULE-02, RULE-04, Guidelines, Evidence, Rego Policy', withheld: 'RULE-03, RULE-05' },
      { stage: 5, label: 'Stage 5: Notice', disclosed: 'All rules + policies disclosed', withheld: 'None' },
    ];

    // Animate stages progressively
    await loadRegoSource();
    
    // Helper to delay
    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    // Stage 1 animation
    document.getElementById('flow-step-1').className = 'flow-step active';
    document.getElementById('flow-disc-1').classList.add('disc-active');
    discBadge.textContent = 'Stage 1';
    discDisclosed.textContent = disclosureStages[0].disclosed;
    discWithheld.textContent = disclosureStages[0].withheld;
    await delay(600);

    // Stage 2 animation
    document.getElementById('flow-step-1').className = 'flow-step passed';
    document.getElementById('flow-step-2').className = 'flow-step active';
    document.getElementById('flow-disc-2').classList.add('disc-active');
    discBadge.textContent = 'Stage 2';
    discDisclosed.textContent = disclosureStages[1].disclosed;
    discWithheld.textContent = disclosureStages[1].withheld;
    await delay(600);

    // Stage 3 animation
    document.getElementById('flow-step-2').className = 'flow-step passed';
    document.getElementById('flow-step-3').className = 'flow-step active';
    document.getElementById('flow-disc-3').classList.add('disc-active');
    discBadge.textContent = 'Stage 3';
    discDisclosed.textContent = disclosureStages[2].disclosed;
    discWithheld.textContent = disclosureStages[2].withheld;
    await delay(600);

    // Stage 4 animation — Rego rules NOW disclosed
    document.getElementById('flow-step-3').className = 'flow-step passed';
    document.getElementById('flow-step-4').className = 'flow-step active';
    document.getElementById('flow-disc-4').classList.add('disc-active');
    discBadge.textContent = 'Stage 4';
    discDisclosed.textContent = disclosureStages[3].disclosed;
    discWithheld.textContent = disclosureStages[3].withheld;
    await delay(600);

    // Stage 5 animation — All disclosed
    document.getElementById('flow-step-4').className = 'flow-step passed';
    document.getElementById('flow-step-5').className = 'flow-step active';
    document.getElementById('flow-disc-5').classList.add('disc-active');
    discBadge.textContent = 'Stage 5';
    discDisclosed.textContent = disclosureStages[4].disclosed;
    discWithheld.textContent = disclosureStages[4].withheld;
    await delay(400);

    // Now actually run the agent (execution already happened logically during animation)
    const outcome = await agent.run(request, regoSourceText);
    
    outcomeCard.className = `glass-card outcome-card ${outcome.status}`;
    outcomeBadge.textContent = outcome.decision;
    outcomeReason.textContent = outcome.reason;
    letterContent.textContent = outcome.notice;

    // Mark disclosure complete
    discBadge.textContent = 'Complete';
    discBadge.className = 'disclosure-badge complete';
    discDisclosed.textContent = 'All 5 rules + guidelines + Rego policies';
    discWithheld.textContent = 'None — fully disclosed';

    renderTrace(outcome.trace);

    // Update NPI status badge
    const npiBadge = document.getElementById('npi-status-badge');
    if (outcome.trace.some(t => t.name === 'VerifyNpiStatusSkill' && t.status === 'fail')) {
      npiBadge.textContent = 'NPI Invalid';
      npiBadge.className = 'badge badge-fail';
    } else if (outcome.trace.some(t => t.name === 'VerifyNpiStatusSkill' && t.status === 'success')) {
      npiBadge.textContent = 'NPI Valid';
      npiBadge.className = 'badge badge-success';
    } else {
      npiBadge.textContent = '';
      npiBadge.className = 'badge';
    }

    // --- Final Gate Highlights (post-execution) ---
    // Re-evaluate gates based on actual outcome
    const hasCoverageError = outcome.trace.some(t => t.name === "VerifyCoverageSkill" && (t.status === "fail" || t.status === "warning")) ||
                             outcome.trace.some(t => t.name === "VerifyNpiStatusSkill" && t.status === "fail");
    const hasCodeMatchError = outcome.trace.some(t => t.name === "RULE-04 (Code Match)" && t.status === "fail");

    if (hasCoverageError || hasCodeMatchError) {
      document.getElementById('flow-step-2').className = "flow-step failed";
      [3, 4, 5].forEach(s => document.getElementById(`flow-step-${s}`).className = "flow-step");
      return;
    }
    document.getElementById('flow-step-2').className = "flow-step passed";

    const hasOpaMismatch = outcome.trace.some(t => t.name === "EvaluateClinicalCriteriaSkill" && t.message.includes("Not Met")) ||
                           outcome.trace.some(t => t.name === "rules.rego" && t.message.includes("approve -> FALSE"));
    document.getElementById('flow-step-3').className = hasOpaMismatch ? "flow-step warning" : "flow-step passed";

    if (outcome.decision === "Approved") {
      document.getElementById('flow-step-4').className = "flow-step passed";
    } else {
      document.getElementById('flow-step-4').className = "flow-step warning";
    }

    const hasNoticeError = outcome.trace.some(t => t.type === "rule" && t.status === "fail" && (t.name.includes("RULE-03") || t.name.includes("RULE-05")));
    document.getElementById('flow-step-5').className = hasNoticeError ? "flow-step failed" : "flow-step passed";
  });

  // Init UI
  renderPresets();
  renderRulesToggles();
  loadCaseToForm(PRESET_CASES[0]);
});
