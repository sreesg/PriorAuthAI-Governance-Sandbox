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

  // 6.5. AI Policy Extraction from PDF
  document.getElementById('btn-extract-pdf').addEventListener('click', async () => {
    const btn = document.getElementById('btn-extract-pdf');
    btn.disabled = true;
    btn.textContent = '🧠 Reading PDF...';
    auditConsole.textContent = 'ClinicalNLP Engine: Reading real_payer_policy_uhc.pdf and extracting policies...';
    
    try {
      const res = await fetch('/agent/extract-policy', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
      const data = await res.json();
      if (res.ok) {
        let log = `✓ Extracted ${data.policies?.length || 0} policies from PDF\n\n`;
        (data.trace || []).forEach(t => { log += `[${t.step}] ${t.message}\n`; });
        log += `\nPolicies saved to rules_declaration.md and rules.rego`;
        auditConsole.textContent = log;
        await refreshActiveFileView();
        showToast(`🧠 Extracted ${data.policies?.length || 0} policies from PDF`, 'success');
      } else {
        auditConsole.textContent = `Error: ${data.error}`;
        showToast(`Extraction failed: ${data.error}`, 'error');
      }
    } catch (e) {
      auditConsole.textContent = `Network error: ${e.message}`;
      showToast(`Error: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = '🧠 Extract Rules from PDF (AI)';
    }
  });

  // 6.6. AI Skill Generation from Policies
  document.getElementById('btn-gen-skills').addEventListener('click', async () => {
    const btn = document.getElementById('btn-gen-skills');
    btn.disabled = true;
    btn.textContent = '🛠️ Generating...';
    auditConsole.textContent = 'ClinicalNLP Engine: Designing skills for extracted policies...';
    
    try {
      const res = await fetch('/agent/generate-skills', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
      const data = await res.json();
      if (res.ok) {
        let log = `✓ Generated ${data.skills?.length || 0} skill definitions\n\n`;
        (data.skills || []).forEach(s => { log += `• ${s.skillName}: ${s.description}\n`; });
        auditConsole.textContent = log;
        showToast(`🛠️ Generated ${data.skills?.length || 0} skills`, 'success');
      } else {
        auditConsole.textContent = `Error: ${data.error}`;
        showToast(`Skill generation failed: ${data.error}`, 'error');
      }
    } catch (e) {
      auditConsole.textContent = `Network error: ${e.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = '🛠️ Generate Skills (AI)';
    }
  });

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
      const isAiPowered = step.name.includes('ClinicalNLP') || step.name.includes('Gemma 4') || (step.details && step.details.model);
      item.className = `timeline-item ${step.status}${isDisclosure ? ' disclosure' : ''}${isAiPowered ? ' ai-powered' : ''}`;
      
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

  // Helper: Run regex extraction locally for comparison with AI
  function runRegexExtraction(request) {
    const notes = request.clinicalNotes.toLowerCase();
    const cptCode = request.cptCode;
    let symptomsDurationWeeks = 0, therapyWeeks = 0, hasObjectiveFindings = false, isRheumatologist = false, hasRadiographs = false;
    
    if (cptCode === "73721") {
      const durationMatch = notes.match(/(\d+)\s*weeks?\s*of\s*(pain|symptom)/) || notes.match(/pain\s*for\s*(\d+)\s*weeks?/);
      if (durationMatch) symptomsDurationWeeks = parseInt(durationMatch[1], 10);
      else if (notes.includes("persistent knee pain")) symptomsDurationWeeks = 8;
      
      const therapyMatch = notes.match(/(\d+)\s*weeks?\s*of\s*(physical therapy|pt|therapy|ibuprofen|nsaids)/);
      if (therapyMatch) therapyWeeks = parseInt(therapyMatch[1], 10);
      else if (notes.includes("physical therapy")) therapyWeeks = 6;
      
      if (notes.includes("tenderness") || notes.includes("swelling") || notes.includes("instability") || notes.includes("locking")) hasObjectiveFindings = true;
      if (notes.includes("radiograph") || notes.includes("x-ray")) hasRadiographs = true;
    }
    if (cptCode === "J0135") {
      if (notes.includes("rheumatoid arthritis")) hasObjectiveFindings = true;
      if (notes.includes("methotrexate") || notes.includes("dmard")) {
        const m = notes.match(/(\d+)\s*months?/);
        therapyWeeks = m ? parseInt(m[1], 10) * 4 : 12;
      }
      if (notes.includes("rheumatologist")) isRheumatologist = true;
    }
    return { symptomsDurationWeeks, therapyWeeks, hasObjectiveFindings, isRheumatologist, hasRadiographs };
  }

  // Helper: Render evidence as HTML with match/mismatch highlights and AI accuracy insights
  function renderEvidenceComparison(data, other, mode) {
    const fields = [
      { key: 'symptomsDurationWeeks', label: 'Symptoms' , unit: 'wks' },
      { key: 'therapyWeeks', label: 'Therapy', unit: 'wks' },
      { key: 'hasObjectiveFindings', label: 'Findings' },
      { key: 'isRheumatologist', label: 'Specialist' },
      { key: 'hasRadiographs', label: 'X-rays' },
    ];
    
    let html = '';
    fields.forEach(f => {
      const val = data[f.key];
      const otherVal = other[f.key];
      const matches = val === otherVal;
      const displayVal = typeof val === 'boolean' ? (val ? '✓ Yes' : '✗ No') : `${val} ${f.unit || ''}`;
      const cls = matches ? 'match' : 'mismatch';
      const icon = matches ? '' : (mode === 'ai' ? ' ✦' : ' ⚠');
      html += `<span class="${cls}">${f.label}: ${displayVal}${icon}</span>\n`;
    });
    return html;
  }

  // Helper: Generate AI accuracy insight callout explaining where regex fails
  function generateAccuracyInsight(regexResult, aiResult, clinicalNotes) {
    const insights = [];
    const notesLower = clinicalNotes.toLowerCase();
    
    // Check therapy negation
    if (regexResult.therapyWeeks > 0 && aiResult.therapyWeeks === 0) {
      if (notesLower.includes('no physical therapy') || notesLower.includes('no pt') || notesLower.includes('not completed')) {
        insights.push({
          field: 'Therapy',
          issue: 'Negation missed',
          detail: `Regex matched "physical therapy" keyword → defaulted to ${regexResult.therapyWeeks} wks. But the text says "No physical therapy completed." The AI understands negation.`,
          severity: 'high'
        });
      }
    }
    
    // Check radiograph negation
    if (regexResult.hasRadiographs && !aiResult.hasRadiographs) {
      if (notesLower.includes('no plain radiographs') || notesLower.includes('no x-ray') || notesLower.includes('not performed')) {
        insights.push({
          field: 'X-rays',
          issue: 'Negation missed',
          detail: `Regex matched "radiograph" keyword → marked as done. But the text says "No plain radiographs performed." The AI reads the full sentence.`,
          severity: 'high'
        });
      }
    }
    
    // Check if AI found radiographs regex missed (positive case)
    if (!regexResult.hasRadiographs && aiResult.hasRadiographs) {
      insights.push({
        field: 'X-rays',
        issue: 'Paraphrase detection',
        detail: `Regex didn't match the exact keywords "radiograph" or "x-ray." The AI understood a paraphrased reference to completed imaging.`,
        severity: 'medium'
      });
    }
    
    // Check therapy duration differences (non-negation)
    if (regexResult.therapyWeeks !== aiResult.therapyWeeks && regexResult.therapyWeeks > 0 && aiResult.therapyWeeks > 0) {
      insights.push({
        field: 'Therapy',
        issue: 'Duration parsing',
        detail: `Regex extracted ${regexResult.therapyWeeks} wks, AI extracted ${aiResult.therapyWeeks} wks. The AI may be interpreting context like "completed 6 weeks" vs "enrolled for 8 weeks" more accurately.`,
        severity: 'medium'
      });
    }
    
    // Check symptom duration
    if (regexResult.symptomsDurationWeeks !== aiResult.symptomsDurationWeeks) {
      insights.push({
        field: 'Symptoms',
        issue: 'Duration interpretation',
        detail: `Regex: ${regexResult.symptomsDurationWeeks} wks, AI: ${aiResult.symptomsDurationWeeks} wks. The AI may handle temporal references like "since last month" that regex can't parse.`,
        severity: 'medium'
      });
    }
    
    return insights;
  }

  // Helper: Render accuracy insights as HTML
  function renderAccuracyInsights(insights) {
    if (insights.length === 0) {
      return '<div class="ai-insight-match">✓ Regex and AI agree on all fields. Both extractions are consistent.</div>';
    }
    
    let html = `<div class="ai-insight-header">⚡ AI Accuracy Advantages Found: ${insights.length}</div>`;
    insights.forEach(insight => {
      html += `<div class="ai-insight-item ${insight.severity}">`;
      html += `<div class="ai-insight-field">${insight.field} — <span class="ai-insight-issue">${insight.issue}</span></div>`;
      html += `<div class="ai-insight-detail">${insight.detail}</div>`;
      html += `</div>`;
    });
    return html;
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
    const aiModeEnabled = document.getElementById('ai-mode-toggle').checked;
    const aiReasoningCard = document.getElementById('ai-reasoning-card');
    const aiStatusBadge = document.getElementById('ai-status-badge');
    const aiModelBadge = document.getElementById('ai-model-badge');
    const aiCompRegex = document.getElementById('ai-comp-regex');
    const aiCompAi = document.getElementById('ai-comp-ai');
    const aiReasoningText = document.getElementById('ai-reasoning-text');
    let aiEvidence = null;
    
    if (aiModeEnabled) {
      // Show AI panel and set loading state
      aiReasoningCard.style.display = 'block';
      aiStatusBadge.textContent = 'Processing...';
      aiStatusBadge.className = 'badge badge-orange';
      aiCompAi.innerHTML = '<span class="ai-loading">🧠 ClinicalNLP Engine is analyzing notes...</span>';
      aiCompRegex.textContent = 'Waiting for AI comparison...';
      aiReasoningText.textContent = '';
      aiModelBadge.textContent = '🧠 ClinicalNLP';
      aiModelBadge.className = 'badge badge-ai';
      
      // Call Gemma 4 12B via Ollama for real AI clinical extraction
      try {
        const aiRes = await fetch('/ai-extract', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            clinicalNotes: request.clinicalNotes,
            cptCode: request.cptCode,
            guidelinesText: `CPT ${request.cptCode}, ICD-10 ${request.icd10Code}`
          })
        });
        if (aiRes.ok) {
          const aiData = await aiRes.json();
          aiEvidence = aiData.extracted;
          aiStatusBadge.textContent = 'Complete';
          aiStatusBadge.className = 'badge badge-success';
          
          // Run regex extraction too for comparison
          const regexResult = runRegexExtraction(request);
          
          // Render comparison
          aiCompRegex.innerHTML = renderEvidenceComparison(regexResult, aiEvidence, 'regex');
          aiCompAi.innerHTML = renderEvidenceComparison(aiEvidence, regexResult, 'ai');
          
          // Generate and show accuracy insights
          const insights = generateAccuracyInsight(regexResult, aiEvidence, request.clinicalNotes);
          const insightsPanel = document.getElementById('ai-insights-panel');
          insightsPanel.innerHTML = renderAccuracyInsights(insights);
          insightsPanel.style.display = 'block';
          
          // Show AI reasoning
          if (aiEvidence.reasoning) {
            aiReasoningText.textContent = `💬 "${aiEvidence.reasoning}"`;
          }
          
          showToast(`🧠 ClinicalNLP: semantic extraction complete`, "success");
        } else {
          const errData = await aiRes.json().catch(() => ({}));
          aiStatusBadge.textContent = 'Offline';
          aiStatusBadge.className = 'badge badge-fail';
          aiCompAi.innerHTML = `<span style="color:var(--accent-red);">Model unavailable: ${errData.error || 'Ollama not running'}</span>`;
          aiModelBadge.textContent = '⚠️ NLP Offline';
          aiModelBadge.className = 'badge badge-fail';
          showToast(`AI extraction unavailable: ${errData.error || 'Ollama offline'}`, "error");
        }
      } catch (e) {
        aiStatusBadge.textContent = 'Error';
        aiStatusBadge.className = 'badge badge-fail';
        aiCompAi.innerHTML = `<span style="color:var(--accent-red);">Network error: ${e.message}</span>`;
        showToast(`AI endpoint error: ${e.message}`, "error");
      }
    } else {
      aiReasoningCard.style.display = 'none';
      aiModelBadge.textContent = '';
      aiModelBadge.className = 'badge';
    }
    
    const outcome = await agent.run(request, regoSourceText, aiEvidence);
    
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
