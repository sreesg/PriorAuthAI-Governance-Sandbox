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
      const panel = document.getElementById(`view-${view}`);
      if (view === 'review') panel.style.display = 'grid';
      else panel.style.display = 'block';
    });
  });

  // Replay demo animation (if present)
  // Concept story animation
  const storyBtn = document.getElementById('btn-play-story');
  if (storyBtn) {
    storyBtn.addEventListener('click', playConceptStory);
  }

  async function playConceptStory() {
    const narration = document.getElementById('story-narration');
    const paAgent = document.getElementById('cv-pa');
    const aviAgent = document.getElementById('cv-avi');
    const paStatus = document.getElementById('cv-pa-status');
    const aviStatus = document.getElementById('cv-avi-status');
    const result = document.getElementById('cv-result');
    const resultIcon = document.getElementById('cv-result-icon');
    const resultText = document.getElementById('cv-result-text');
    const btn = document.getElementById('btn-play-story');
    
    if (!narration || !btn) return;
    btn.disabled = true; btn.textContent = '\u23f8 Playing...';
    
    // Reset
    document.querySelectorAll('.cv-stage').forEach(s => s.className = 'cv-stage');
    document.querySelectorAll('.cv-hook-wire').forEach(h => h.classList.remove('fired'));
    if (paAgent) paAgent.classList.remove('active');
    if (aviAgent) aviAgent.classList.remove('active');
    if (paStatus) paStatus.textContent = 'idle';
    if (aviStatus) aviStatus.textContent = 'idle';
    if (result) { result.className = 'cv-result'; resultIcon.textContent = '\u23f3'; resultText.textContent = 'Pending'; }

    const delay = ms => new Promise(r => setTimeout(r, ms));

    // PA Agent activates
    if (paAgent) { paAgent.classList.add('active'); paStatus.textContent = 'running pipeline'; }
    narration.textContent = '\U0001f916 PA Agent starts. Receives the request...';
    await delay(1200);

    // Stage 1
    const cv1 = document.getElementById('cv-1');
    if (cv1) cv1.classList.add('active');
    narration.textContent = '\U0001f4e5 Stage 1: Hook on_request fires \u2192 PHI Rule scrubs sensitive data.';
    await delay(800);
    const hw1 = document.getElementById('cv-hw-1');
    if (hw1) hw1.classList.add('fired');
    await delay(1000);
    if (cv1) { cv1.classList.remove('active'); cv1.classList.add('done'); }

    // Stage 2
    const cv2 = document.getElementById('cv-2');
    if (cv2) cv2.classList.add('active');
    narration.textContent = '\U0001f6e0\ufe0f Stage 2: PolicyRouter skill matches policy. Hook fires \u2192 Code Match rule validates.';
    await delay(800);
    const hw2 = document.getElementById('cv-hw-2');
    if (hw2) hw2.classList.add('fired');
    await delay(1000);
    if (cv2) { cv2.classList.remove('active'); cv2.classList.add('done'); }

    // Stage 3
    const cv3 = document.getElementById('cv-3');
    if (cv3) cv3.classList.add('active');
    narration.textContent = '\U0001f9e0 Stage 3: ExtractEvidence calls LLM. AI reads notes. Hook validates quality.';
    await delay(800);
    const hw3 = document.getElementById('cv-hw-3');
    if (hw3) hw3.classList.add('fired');
    if (aviAgent) { aviAgent.classList.add('active'); aviStatus.textContent = 'loading context'; }
    await delay(1200);
    if (cv3) { cv3.classList.remove('active'); cv3.classList.add('done'); }

    // Stage 4
    const cv4 = document.getElementById('cv-4');
    if (cv4) cv4.classList.add('active');
    narration.textContent = '\u2699\ufe0f Stage 4: Criteria validation — checking each threshold against policy requirements...';
    const criteriaEl = document.getElementById('cv-criteria-check');
    if (criteriaEl) criteriaEl.classList.add('checking');
    await delay(800);
    const hw4 = document.getElementById('cv-hw-4');
    if (hw4) hw4.classList.add('fired');
    if (paStatus) paStatus.textContent = 'evaluating';
    await delay(600);
    if (criteriaEl) { criteriaEl.classList.remove('checking'); criteriaEl.classList.add('passed'); criteriaEl.textContent = 'symptom \u2713 therapy \u2713 findings \u2713'; }
    narration.textContent = '\u2705 All criteria thresholds met. Conservatism rule: no escalation needed.';
    await delay(1000);
    if (cv4) { cv4.classList.remove('active'); cv4.classList.add('done'); }

    // Stage 5
    const cv5 = document.getElementById('cv-5');
    if (cv5) cv5.classList.add('active');
    narration.textContent = '\U0001f4dd Stage 5: GenNotice drafts letter. Hook fires \u2192 Plain Language rule translates.';
    await delay(800);
    const hw5 = document.getElementById('cv-hw-5');
    if (hw5) hw5.classList.add('fired');
    await delay(1000);
    if (cv5) { cv5.classList.remove('active'); cv5.classList.add('done'); }

    // Decision
    if (paStatus) paStatus.textContent = 'decision made';
    if (aviStatus) aviStatus.textContent = 'ready to explain';
    if (result) { result.classList.add('approved'); resultIcon.textContent = '\u2705'; resultText.textContent = 'Approved'; }
    narration.textContent = '\u2705 All criteria met. PA Agent approved. AVI Agent ready to explain the reasoning.';
    
    btn.disabled = false; btn.textContent = '\u25b6 Play Again';
  }

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
  await refreshSkillsPills();
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
    
    // Dropdown change — update detail AND refresh artifacts for new policy
    select.addEventListener('change', () => {
      showWorkspacePolicy(select.value);
      // Re-trigger the active tab to load the correct policy file
      const activeTabBtn = document.querySelector('.file-tabs .tab-btn.active');
      if (activeTabBtn) activeTabBtn.click();
      auditConsole.textContent = `Selected: ${select.options[select.selectedIndex]?.text || select.value}\nClick Generate to build artifacts for this policy.`;
    });
    
    // Generate All button
    document.getElementById('btn-gen-all').addEventListener('click', async () => {
      const policyId = select.value;
      if (!policyId) return;
      const btn = document.getElementById('btn-gen-all');
      btn.disabled = true;
      
      const actions = ['rules', 'skills', 'hooks'];
      
      for (let i = 0; i < actions.length; i++) {
        const action = actions[i];
        btn.textContent = `⏳ ${action} (${i+1}/${actions.length})`;
        auditConsole.textContent += (i === 0 ? '' : '\n') + `⏳ Generating ${action} for ${policyId}...\n`;
        
        // Force DOM update before the fetch
        await new Promise(r => setTimeout(r, 50));
        
        try {
          const res = await fetch('/agent/build-for-policy', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ policyId, action })
          });
          const data = await res.json();
          auditConsole.textContent += `✓ ${action.toUpperCase()}: ${(data.auditLog || data.error || 'Done').split('\n')[0]}\n`;
        } catch(e) {
          auditConsole.textContent += `✗ ${action.toUpperCase()}: ${e.message}\n`;
        }
      }
      
      auditConsole.textContent += `\n✅ All artifacts generated for ${policyId}.`;
      btn.disabled = false;
      btn.textContent = '🧠 Generate All';
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
        const origText = btn.textContent;
        btn.disabled = true;
        btn.textContent = '⏳';
        auditConsole.textContent = `🧠 AI model is working on generating ${action} for ${policyId}...\n\nPlease wait while the ClinicalNLP Engine processes this request.`;
        try {
          const res = await fetch('/agent/build-for-policy', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ policyId, action })
          });
          const data = await res.json();
          auditConsole.textContent = `✅ ${action.toUpperCase()} generated:\n\n${data.auditLog || data.error || 'Done'}`;
          if (action === 'skills') refreshSkillsPills();
          await refreshActiveFileView();
          showToast(`${action} generated for ${policyId}`, 'success');
        } catch(e) {
          auditConsole.textContent = `✗ Error generating ${action}: ${e.message}`;
        } finally { btn.disabled = false; btn.textContent = origText; }
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
    
    // Handle PDF viewer tab
    if (activeFile === 'view-pdf') {
      if (pdfDownloadPanel) {
        pdfDownloadPanel.style.display = "block";
        const frame = document.getElementById('pdf-viewer-frame');
        const note = document.getElementById('pdf-context-note');
        
        // Get the selected policy's PDF file
        const policySelect = document.getElementById('policy-select');
        const selectedPolicyId = policySelect ? policySelect.value : '';
        
        // Find the matching policy to get its PDF path
        let pdfPath = './real_payer_policy_uhc.pdf';
        let policyLabel = 'UHC General Policy';
        
        if (selectedPolicyId && workspacePolicies) {
          // Fetch the full policy detail which includes pdfFile
          fetch(`/agent/policy-detail?id=${selectedPolicyId}`).then(r => r.json()).then(p => {
            if (p.pdfFile) {
              frame.src = `./${p.pdfFile}`;
              if (note) note.textContent = `📌 ${p.payer}: ${p.policyName} (${p.policyId})`;
            }
          }).catch(() => {});
        }
        
        if (frame && !frame.src.includes('policy_')) {
          frame.src = pdfPath;
        }
        if (note && !note.textContent) {
          note.textContent = `📌 ${policyLabel}`;
        }
      }
      return;
    }

    try {
      const res = await fetch(`./${activeFile}?update=${Date.now()}`);
      if (res.ok) {
        const text = await res.text();
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

  // 3. Render active registered skills pills in the UI (from both JS agent and backend)
  async function refreshSkillsPills() {
    activeSkillsPills.innerHTML = '';
    
    // Get JS agent skills
    const jsSkills = Object.keys(agent.skills);
    
    // Get generated skills from all policy files
    let backendSkills = [];
    try {
      const res = await fetch('/agent/policies');
      const policies = await res.json();
      for (const p of policies) {
        try {
          const sRes = await fetch(`/policies/${p.policyId}_skills.json?t=${Date.now()}`);
          if (sRes.ok) {
            const data = await sRes.json();
            (data.skills || []).forEach(s => {
              if (s.skillName && !backendSkills.includes(s.skillName)) {
                backendSkills.push(s.skillName);
              }
            });
          }
        } catch(_) {}
      }
    } catch(_) {}
    
    const allSkills = [...new Set([...jsSkills, ...backendSkills])];
    discoveredSkillsCount.textContent = `${allSkills.length} Active Skills`;

    allSkills.forEach(name => {
      const pill = document.createElement('span');
      pill.className = backendSkills.includes(name) ? 'skill-pill new-injected' : 'skill-pill';
      pill.textContent = name;
      activeSkillsPills.appendChild(pill);
    });
  }

  // 4. File Workspace Tab Event Handlers
  tabButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const fileAttr = btn.getAttribute('data-file');
      
      // For policy-specific files, prepend the selected policy ID
      const policySelect = document.getElementById('policy-select');
      const selectedPolicyId = policySelect ? policySelect.value : '';
      
      if (fileAttr === 'rules_declaration.md' && selectedPolicyId) {
        activeFile = `policies/${selectedPolicyId}_rules.md`;
      } else if (fileAttr === 'generated_skills.json' && selectedPolicyId) {
        activeFile = `policies/${selectedPolicyId}_skills.json`;
      } else if (fileAttr === 'rules.rego') {
        // Try per-policy hooks as "rego" tab shows hooks
        activeFile = selectedPolicyId ? `policies/${selectedPolicyId}_hooks.json` : 'rules.rego';
      } else {
        activeFile = fileAttr;
      }
      
      await refreshActiveFileView();
    });
  });

  // 5. (Save/compile removed — replaced by per-policy AI generation)
  // 6. (Reset removed — policies are managed through JSON files)

  // 6.5. (Removed — per-policy action buttons handle this now)

  // 6.6. (Removed — per-policy action buttons handle this now)

  // 6.7. Natural Language Creator (Skill, Rule, or Hook)
  const nlSkillInput = document.getElementById('nl-skill-input');
  const btnCreateSkill = document.getElementById('btn-create-skill');
  const nlSkillStatus = document.getElementById('nl-skill-status');

  btnCreateSkill.addEventListener('click', async () => {
    const description = nlSkillInput.value.trim();
    const entryType = document.getElementById('manual-entry-type').value; // skill, rule, or hook
    
    if (!description) {
      nlSkillStatus.textContent = "Please describe in plain English.";
      nlSkillStatus.className = "nl-skill-status error";
      return;
    }

    // Get currently selected policy for context
    const policySelect = document.getElementById('policy-select');
    const policyId = policySelect ? policySelect.value : '';

    btnCreateSkill.disabled = true;
    btnCreateSkill.textContent = "⏳";
    nlSkillStatus.textContent = `🧠 AI generating ${entryType} from description...`;
    nlSkillStatus.className = "nl-skill-status processing";

    try {
      if (entryType === 'skill' && !policyId) {
        // Use the old /generate-skill endpoint for standalone skills
        const res = await fetch('/generate-skill', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ description })
        });
        if (res.ok) {
          const data = await res.json();
          auditConsole.textContent = data.auditLogs.join('\n');
          await agent.reloadSkills();
          await refreshSkillsPills();
          nlSkillStatus.textContent = `✓ Skill "${data.skillName}" registered.`;
          nlSkillStatus.className = "nl-skill-status success";
          nlSkillInput.value = '';
          showToast(`Skill "${data.skillName}" created!`, "success");
        } else {
          const err = await res.json().catch(() => ({}));
          nlSkillStatus.textContent = `Error: ${err.error || 'Failed'}`;
          nlSkillStatus.className = "nl-skill-status error";
        }
      } else if (policyId) {
        // Use /agent/build-for-policy with custom description context
        const res = await fetch('/agent/build-for-policy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ policyId, action: entryType === 'hook' ? 'hooks' : entryType + 's', customDescription: description })
        });
        const data = await res.json();
        if (res.ok) {
          auditConsole.textContent = data.auditLog || 'Done';
          nlSkillStatus.textContent = `✓ Custom ${entryType} added for ${policyId}`;
          nlSkillStatus.className = "nl-skill-status success";
          nlSkillInput.value = '';
          await refreshSkillsPills();
          showToast(`${entryType} added for ${policyId}`, 'success');
        } else {
          nlSkillStatus.textContent = `Error: ${data.error || 'Failed'}`;
          nlSkillStatus.className = "nl-skill-status error";
        }
      } else {
        nlSkillStatus.textContent = "Select a policy first, or use 'Skill' type for standalone skills.";
        nlSkillStatus.className = "nl-skill-status error";
      }
    } catch (e) {
      nlSkillStatus.textContent = `Network error: ${e.message}`;
      nlSkillStatus.className = "nl-skill-status error";
    } finally {
      btnCreateSkill.disabled = false;
      btnCreateSkill.textContent = "🧠 Add";
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
        btn.innerHTML = `<h3>${c.title.split('—')[0].trim()}</h3><p style="font-size:0.62rem;margin-top:2px;color:${c.title.includes('Approved') || c.title.includes('Approve') ? '#059669' : '#d97706'};font-weight:700;">${c.title.includes('Approved') || c.title.includes('Approve') ? '✓ Approve' : '⚠ Escalate'}</p>`;
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
    
    // Load policy-specific hooks dynamically
    loadPolicyHooksForCase(req.cptCode);
  }

  async function loadPolicyHooksForCase(cptCode) {
    const hooksRow = document.getElementById('policy-hooks-row');
    const hooksContainer = document.getElementById('policy-hooks-container');
    if (!hooksRow || !hooksContainer) return;
    hooksContainer.innerHTML = '';
    hooksRow.style.display = 'none';
    
    try {
      // Find which policy matches this CPT
      const policiesRes = await fetch('/agent/policies');
      const policies = await policiesRes.json();
      const matched = policies.find(p => p.cptCodes.includes(cptCode));
      if (!matched) return;
      
      // Try loading generated hooks for this policy
      const hooksRes = await fetch(`/policies/${matched.policyId}_hooks.json?t=${Date.now()}`);
      if (!hooksRes.ok) return;
      
      const hooksData = await hooksRes.json();
      const hooks = hooksData.hooks || [];
      if (hooks.length === 0) return;
      
      hooks.forEach(h => {
        const pill = document.createElement('div');
        pill.className = 'rule-item';
        pill.style.borderColor = '#0ea5e9';
        pill.innerHTML = `<div class="rule-info"><h4>🪝 ${h.hookName || h.stage || '?'}</h4></div>`;
        pill.title = h.description || '';
        hooksContainer.appendChild(pill);
      });
      hooksRow.style.display = 'flex';
    } catch(e) {
      console.error('Hook loading error:', e);
    }
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
    outcomeReason.textContent = 'Calling AI agent...';
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

      // AI model badge — not needed in outcome (toggle already visible)
      const aiModelBadge = document.getElementById('ai-model-badge');
      if (aiModelBadge) {
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
      if (aiModeEnabled && outcome.evidence && Object.keys(outcome.evidence).length > 0) {
        aiReasoningCard.style.display = 'block';
        const aiStatusBadge = document.getElementById('ai-status-badge');
        const aiReasoningText = document.getElementById('ai-reasoning-text');
        const aiCompRegex = document.getElementById('ai-comp-regex');
        const aiCompAi = document.getElementById('ai-comp-ai');
        
        aiStatusBadge.textContent = 'Complete';
        aiStatusBadge.className = 'badge badge-success';
        
        // Format AI evidence nicely
        const ev = outcome.evidence;
        let aiHtml = '';
        aiHtml += `<span class="match">Therapy: ${ev.conservativeTherapyWeeks || 0} wks</span>\n`;
        aiHtml += `<span class="${ev.hasNeurologicalSymptoms ? 'match' : ''}"'>Neurological: ${ev.hasNeurologicalSymptoms ? '✓ Yes' : '✗ No'}</span>\n`;
        aiHtml += `<span class="${ev.hasMechanicalSymptoms ? 'match' : ''}">Mechanical: ${ev.hasMechanicalSymptoms ? '✓ Yes' : '✗ No'}</span>\n`;
        aiHtml += `<span class="${ev.hasSpecialist ? 'match' : ''}">Specialist: ${ev.hasSpecialist ? '✓ Yes' : '✗ No'}</span>\n`;
        aiHtml += `<span class="${ev.hasImagingFindings ? 'match' : ''}">Imaging: ${ev.hasImagingFindings ? '✓ Yes' : '✗ No'}</span>\n`;
        if (ev.failedMedications && ev.failedMedications.length > 0) {
          aiHtml += `<span class="match">Failed meds: ${ev.failedMedications.join(', ')}</span>\n`;
        }
        if (ev.severityScore) {
          aiHtml += `<span class="match">Severity: ${ev.severityScore}</span>\n`;
        }
        aiCompAi.innerHTML = aiHtml;
        
        // Show what regex would have found (run locally for comparison)
        const notes = request.clinicalNotes.toLowerCase();
        let regexHtml = '';
        const rxTherapy = (notes.match(/(\d+)\s*weeks?\s*(?:of\s+)?(?:pt|physical therapy|therapy)/)||[])[1] || '0';
        regexHtml += `<span class="${rxTherapy == (ev.conservativeTherapyWeeks||0) ? 'match' : 'mismatch'}">Therapy: ${rxTherapy} wks</span>\n`;
        const rxNeuro = /radiculopathy|numbness|tingling|weakness|radiating|straight leg/.test(notes);
        regexHtml += `<span class="${rxNeuro === ev.hasNeurologicalSymptoms ? 'match' : 'mismatch'}">Neurological: ${rxNeuro ? '✓ Yes' : '✗ No'}</span>\n`;
        const rxMech = /locking|catching|giving way|mechanical/.test(notes);
        regexHtml += `<span class="${rxMech === ev.hasMechanicalSymptoms ? 'match' : 'mismatch'}">Mechanical: ${rxMech ? '✓ Yes' : '✗ No'}</span>\n`;
        const rxSpec = /oncologist|dermatologist|orthopedic|board-certified|rheumatologist/.test(notes);
        regexHtml += `<span class="${rxSpec === ev.hasSpecialist ? 'match' : 'mismatch'}">Specialist: ${rxSpec ? '✓ Yes' : '✗ No'}</span>\n`;
        const rxImg = /mri shows|mri dated|ct shows|biopsy/.test(notes);
        regexHtml += `<span class="${rxImg === ev.hasImagingFindings ? 'match' : 'mismatch'}">Imaging: ${rxImg ? '✓ Yes' : '✗ No'}</span>\n`;
        aiCompRegex.innerHTML = regexHtml;
        
        // Show reasoning
        aiReasoningText.textContent = ev.reasoning ? `💬 "${ev.reasoning}"` : '';
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

  const AVI_SYSTEM_PROMPT = ''; // Moved to Python backend (agent_engine.py)

  aviFab.addEventListener('click', () => {
    aviPanel.style.display = aviPanel.style.display === 'none' ? 'flex' : 'none';
    if (aviPanel.style.display === 'flex') aviInput.focus();
  });
  aviClose.addEventListener('click', () => { aviPanel.style.display = 'none'; });

  function getAviUiContext() {
    // Determine which tab is active
    const activeTabEl = document.querySelector('.main-tab.active');
    const activeTab = activeTabEl ? activeTabEl.getAttribute('data-view') : 'review';
    
    const context = { activeTab };
    
    if (activeTab === 'workspace') {
      const policySelect = document.getElementById('policy-select');
      context.selectedPolicy = policySelect ? policySelect.value : '';
      context.auditLog = auditConsole ? auditConsole.textContent.substring(0, 300) : '';
    } else {
      context.caseContext = `Patient: ${inputPatientName.value}, Member: ${inputMemberId.value}
CPT: ${inputCpt.value}, ICD-10: ${inputIcd.value}, Provider: ${inputProvider.value}
Notes: ${inputNotes.value.substring(0, 200)}
Decision: ${outcomeBadge.textContent} — ${outcomeReason.textContent}`;
    }
    return context;
  }

  async function sendAviMessage() {
    const msg = aviInput.value.trim();
    if (!msg) return;
    
    const userBubble = document.createElement('div');
    userBubble.className = 'avi-msg avi-user';
    userBubble.innerHTML = `<p>${escapeHtml(msg)}</p>`;
    aviMessages.appendChild(userBubble);
    aviInput.value = '';
    
    const typing = document.createElement('div');
    typing.className = 'avi-msg avi-bot';
    typing.innerHTML = '<p class="avi-typing">AVI is thinking...</p>';
    aviMessages.appendChild(typing);
    aviMessages.scrollTop = aviMessages.scrollHeight;
    
    try {
      const res = await fetch('/agent/avi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: msg, uiContext: getAviUiContext() })
      });
      const data = await res.json();
      typing.remove();
      
      if (res.ok) {
        const botBubble = document.createElement('div');
        botBubble.className = 'avi-msg avi-bot';
        botBubble.innerHTML = `<p>${(data.response || '').replace(/\n/g, '<br>')}</p>`;
        aviMessages.appendChild(botBubble);
      } else {
        const errBubble = document.createElement('div');
        errBubble.className = 'avi-msg avi-bot';
        errBubble.innerHTML = `<p style="color:var(--accent-red);">Error: ${data.error || 'Unknown error'}</p>`;
        aviMessages.appendChild(errBubble);
      }
    } catch (e) {
      typing.remove();
      const errBubble = document.createElement('div');
      errBubble.className = 'avi-msg avi-bot';
      errBubble.innerHTML = `<p style="color:var(--accent-red);">Offline: ${e.message}</p>`;
      aviMessages.appendChild(errBubble);
    }
    aviMessages.scrollTop = aviMessages.scrollHeight;
  }

  aviSend.addEventListener('click', sendAviMessage);
  aviInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendAviMessage(); });

});
