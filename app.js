import { PRESET_CASES } from './cases.js';
import { PriorAuthAgent } from './agent.js';
import { render as renderBeacon } from './static/crf/beacon_harness_viz.js';
import { render as renderAxisweave } from './static/crf/axisweave_context_panel.js';
import { render as renderGraph } from './static/crf/causal_graph_viz.js';

document.addEventListener('DOMContentLoaded', async () => {
  const agent = new PriorAuthAgent();
  let activeFile = "rules_declaration.md";
  let activeCaseId = null;

  // Load S3 asset URLs for image and video
  try {
    const assetRes = await fetch('/agent/asset-urls');
    const assets = await assetRes.json();
    const archImg = document.getElementById('arch-image');
    if (archImg && assets.architecture) {
      archImg.src = assets.architecture;
      archImg.style.display = 'block';
    }
    const videoSrc = document.getElementById('video-source');
    const videoEl = document.getElementById('tutorial-video');
    if (videoSrc && assets.video) {
      videoSrc.src = assets.video;
      if (videoEl) videoEl.load();
    }
  } catch(e) {
    console.warn('Asset URLs not available, using local fallback');
    const archImg = document.getElementById('arch-image');
    if (archImg) { archImg.src = './PA Agentic architecture.png'; archImg.style.display = 'block'; }
    const videoSrc = document.getElementById('video-source');
    if (videoSrc) videoSrc.src = './PA agent.mp4';
  }

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
  document.querySelectorAll('.sidebar-tab[data-view]').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const view = tab.getAttribute('data-view');
      document.querySelectorAll('.view-panel').forEach(p => p.style.display = 'none');
      const panel = document.getElementById(`view-${view}`);
      if (view === 'review') panel.style.display = 'grid';
      else panel.style.display = 'block';

      // Video pause/resume
      const video = document.getElementById('tutorial-video');
      if (video) {
        if (view !== 'video') {
          video.pause();
        }
      }
    });
  });

  // Replay demo animation (if present)
  // Concept animation — 3-column with live data and decision explanations
  const storyBtn = document.getElementById('btn-play-story');
  if (storyBtn) storyBtn.addEventListener('click', playConceptStory);

  async function playConceptStory() {
    const btn = document.getElementById('btn-play-story');
    const narr = document.getElementById('story-narration');
    const aviMsg = document.getElementById('avi-concept-msg');
    const paPath = document.getElementById('pa-path');
    const chPath = document.getElementById('ch-path');
    if (!btn || !narr) return;
    btn.disabled = true; btn.textContent = 'Playing...';

    // Reset
    document.querySelectorAll('.ac-item').forEach(el => el.classList.remove('lit'));
    document.querySelectorAll('.agent-col').forEach(el => el.classList.remove('active'));
    const paDec = document.getElementById('pa-decision');
    const chDec = document.getElementById('ch-decision');
    if (paDec) { paDec.className = 'ac-decision'; paDec.textContent = '\u2014'; }
    if (chDec) { chDec.className = 'ac-decision'; chDec.textContent = '\u2014'; }
    if (paPath) paPath.textContent = '';
    if (chPath) chPath.textContent = '';
    document.getElementById('ac-pa-status').textContent = 'idle';
    document.getElementById('ac-ch-status').textContent = 'idle';
    document.getElementById('cc-c1').textContent = '\u2460 Therapy \u22656wk: ?'; document.getElementById('cc-c1').className = '';
    document.getElementById('cc-c2').textContent = '\u2461 Neuro symptoms: ?'; document.getElementById('cc-c2').className = '';
    document.getElementById('cc-c3').textContent = '\u2462 Red flags: ?'; document.getElementById('cc-c3').className = '';
    document.getElementById('cc-c4').textContent = '\u2463 No prior MRI: ?'; document.getElementById('cc-c4').className = '';
    if (aviMsg) aviMsg.textContent = 'watching';

    const delay = ms => new Promise(r => setTimeout(r, ms));
    const light = id => { const el = document.getElementById(id); if(el) el.classList.add('lit'); };
    const dim = id => { const el = document.getElementById(id); if(el) el.classList.remove('lit'); };

    // === PA AGENT ===
    document.getElementById('ac-pa').classList.add('active');
    document.getElementById('ac-pa-status').textContent = 'running';
    if (paPath) paPath.textContent = 'Processing request...\nCPT: 72148 (Lumbar MRI)\nICD: M54.16 (Radiculopathy)';
    narr.textContent = 'PA Agent starts. Hook fires, PHI rule scrubs SSN/DOB from logs.';
    light('pa-h1'); light('pa-r1');
    await delay(900); dim('pa-h1'); dim('pa-r1');

    light('pa-s1'); light('pa-r3');
    narr.textContent = 'PolicyRouter matches CPT 72148 to POL-RAD-501. Code Match validates ICD M54.16.';
    if (paPath) paPath.textContent = 'Matched: POL-RAD-501\nPayer: UnitedHealthcare\nCategory: Radiology\nCriteria: 4 to evaluate';
    await delay(900); dim('pa-s1'); dim('pa-r3');

    light('pa-s2');
    narr.textContent = 'ExtractEvidence calls LLM. Finds: 8wk symptoms, PT completed, positive SLR test.';
    if (paPath) paPath.textContent = 'LLM Evidence Extraction:\n- Symptoms: 8 weeks\n- Therapy: PT completed (8wk)\n- Neuro: Positive SLR at 30 deg\n- Imaging: No prior MRI';
    await delay(1000); dim('pa-s2');

    light('pa-s3');
    narr.textContent = 'EvalCriteria checking each threshold against policy...';
    await delay(400);
    document.getElementById('cc-c1').textContent = '\u2460 Therapy \u22656wk: 8wk \u2713'; document.getElementById('cc-c1').classList.add('met');
    await delay(350);
    document.getElementById('cc-c2').textContent = '\u2461 Neuro: positive SLR \u2713'; document.getElementById('cc-c2').classList.add('met');
    await delay(350);
    document.getElementById('cc-c3').textContent = '\u2462 Red flags: none \u2713'; document.getElementById('cc-c3').classList.add('met');
    await delay(350);
    document.getElementById('cc-c4').textContent = '\u2463 Prior MRI: pass \u2713'; document.getElementById('cc-c4').classList.add('met');
    await delay(400);

    light('pa-h2'); light('pa-r2');
    narr.textContent = 'Hook on_criteria fires. Conservatism rule: all met, no escalation needed.';
    if (paPath) paPath.textContent = 'Criteria Results:\n1. Therapy >= 6wk: 8wk PASS\n2. Neuro symptoms: SLR+ PASS\n3. Red flags: None PASS\n4. Prior MRI: No conflict PASS\n\nConservatism: No escalation needed.';
    await delay(800); dim('pa-s3'); dim('pa-h2'); dim('pa-r2');

    light('pa-s4'); light('pa-h3');
    narr.textContent = 'GenNotice drafts approval letter. Hook on_notice applies plain language.';
    await delay(700); dim('pa-s4'); dim('pa-h3');

    paDec.textContent = 'APPROVED'; paDec.className = 'ac-decision approved';
    if (paPath) paPath.textContent = 'DECISION: APPROVED\n\nReason: All 4 criteria satisfied.\n- 8 weeks PT (needs 6)\n- Positive neurological exam\n- No red flag symptoms\n- No conflicting prior imaging\n\nNotice letter generated.';
    document.getElementById('ac-pa-status').textContent = 'done';
    document.getElementById('ac-pa').classList.remove('active');
    narr.textContent = 'PA Agent approved. Handing to Challenger for independent quality review...';
    await delay(1100);

    // === CHALLENGER ===
    document.getElementById('ac-ch').classList.add('active');
    document.getElementById('ac-ch-status').textContent = 'reviewing';
    if (chPath) chPath.textContent = 'Received PA decision: APPROVED\nReviewing as quality auditor...\nLooking for weak or assumed evidence.';

    light('ch-h1');
    narr.textContent = 'Hook on_pa_decision fires. Challenger receives all evidence and criteria results.';
    await delay(800); dim('ch-h1');

    light('ch-s1');
    narr.textContent = 'ReinterpretEvidence: Notes say "no prior MRI" but CRIT-4 was marked pass. Why?';
    if (chPath) chPath.textContent = 'FINDING: Notes state "no prior MRI"\nbut PA Agent marked CRIT-4 as PASS.\n\nCriterion says: "No MRI of same\nregion within 12 months."\n\nIs this truly met or just assumed?';
    await delay(1100); dim('ch-s1');

    document.getElementById('cc-c4').textContent = '\u2463 Prior MRI: WEAK'; 
    document.getElementById('cc-c4').className = 'unmet';

    light('ch-s2'); light('ch-r1');
    narr.textContent = 'AssessGaps: PA logic assumed "no prior = pass" but phrasing is ambiguous. Rule: cite evidence.';
    if (chPath) chPath.textContent = 'Analysis:\n- "No prior MRI" could mean never\n  had one, OR not in 12 months.\n- PA Agent assumed favorable reading\n- No documentation confirms timeline\n- This is ASSUMED evidence, not cited.';
    await delay(1100); dim('ch-s2'); dim('ch-r1');

    light('ch-s3'); light('ch-r2');
    narr.textContent = 'EvalStrength: Approval rests on ambiguous CRIT-4 evidence. Not a rubber stamp.';
    await delay(900); dim('ch-s3'); dim('ch-r2');

    light('ch-r3');
    narr.textContent = 'Confidence rule: score 8/10 >= 7. FORMAL CHALLENGE authorized.';
    if (chPath) chPath.textContent = 'Assessment: Evidence WEAK on CRIT-4\nConfidence: 8/10 (threshold: 7)\n\nRULE-C3: Formal challenge authorized.\nRULE-C1: Cited "no prior MRI" text.\nRULE-C2: Substantive analysis done.';
    await delay(900); dim('ch-r3');

    light('ch-h2');
    narr.textContent = 'Hook on_challenge fires. RED FLAG: decision overridden, sent to Medical Director.';
    await delay(600); dim('ch-h2');

    chDec.textContent = 'CHALLENGE 8/10'; chDec.className = 'ac-decision challenge';
    paDec.textContent = 'FLAGGED'; paDec.className = 'ac-decision flagged';
    if (chPath) chPath.textContent = 'VERDICT: CHALLENGE (8/10)\n\nOverride: APPROVED -> FLAGGED\nReason: CRIT-4 relies on ambiguous\nphrasing without explicit timeline.\n\nRouted to Medical Director.\nProvider may need to clarify MRI\nhistory before approval proceeds.';
    document.getElementById('ac-ch-status').textContent = 'overridden';
    document.getElementById('ac-ch').classList.remove('active');
    if (aviMsg) aviMsg.textContent = 'explains: "CRIT-4 evidence contradicts notes wording"';
    narr.textContent = 'RESULT: Approval overridden. Medical Director review required. AVI explains the disagreement.';
    
    btn.disabled = false; btn.textContent = 'Replay';
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
        let pdfPath = '/agent/pdf/real_payer_policy_uhc.pdf';
        let policyLabel = 'UHC General Policy';
        
        if (selectedPolicyId && workspacePolicies) {
          // Fetch the full policy detail which includes pdfFile
          fetch(`/agent/policy-detail?id=${selectedPolicyId}`).then(r => r.json()).then(p => {
            if (p.pdfFile) {
              frame.src = `/agent/pdf/${p.pdfFile}`;
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
    activeCaseId = caseObj.id;
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
    
    // Reset pipeline flow (only if elements exist)
    [1, 2, 3, 4, 5].forEach(s => {
      const el = document.getElementById(`flow-step-${s}`);
      if (el) {
        el.className = 'flow-step';
        const disc = el.querySelector('.flow-step-disclosure');
        if (disc) disc.classList.remove('disc-active');
      }
    });
    
    // Reset disclosure panel (only if elements exist)
    const discBadge = document.getElementById('disclosure-stage-badge');
    if (discBadge) {
      discBadge.textContent = 'Idle';
      discBadge.className = 'disclosure-badge';
    }
    const discDisclosed = document.getElementById('disc-disclosed');
    if (discDisclosed) discDisclosed.textContent = '—';
    const discWithheld = document.getElementById('disc-withheld');
    if (discWithheld) discWithheld.textContent = '—';
    
    // Hide AI panels
    const aiCard = document.getElementById('ai-reasoning-card');
    if (aiCard) aiCard.style.display = 'none';
    const challengerCard = document.getElementById('challenger-card');
    if (challengerCard) challengerCard.style.display = 'none';
    const aiInsights = document.getElementById('ai-insights-panel');
    if (aiInsights) aiInsights.style.display = 'none';
    const aiModelBadge = document.getElementById('ai-model-badge');
    if (aiModelBadge) { aiModelBadge.textContent = ''; aiModelBadge.className = 'badge'; }
    const npiBadge = document.getElementById('npi-status-badge');
    if (npiBadge) { npiBadge.textContent = ''; npiBadge.className = 'badge'; }
    
    // Remove any note summary/quality popups
    document.querySelectorAll('.notes-summary-popup').forEach(el => el.remove());
    
    // Show evidence bundle actions
    const evidenceActions = document.getElementById('evidence-actions');
    const bundleLink = document.getElementById('evidence-bundle-link');
    if (evidenceActions && caseObj.evidenceBundle) {
      evidenceActions.style.display = 'flex';
      bundleLink.href = `/agent/pdf/${caseObj.evidenceBundle}`;
      // Store bundle path for the read button
      evidenceActions.dataset.bundle = caseObj.evidenceBundle;
      evidenceActions.dataset.cpt = req.cptCode;
    } else if (evidenceActions) {
      evidenceActions.style.display = 'none';
    }
    
    // Load policy-specific hooks dynamically
    loadPolicyHooksForCase(req.cptCode);

    // Trigger cockpit panel updates
    const requestId = caseObj.id || 'req-001';
    const memberId = req.memberId || 'MEM-4401';
    
    // Render BEACON 7-Layer Harness
    renderBeacon('review-beacon-panel', requestId, { autoRefresh: true });
    
    // Render Axisweave Context (Qdrant)
    renderAxisweave('review-axisweave-panel', requestId);
    
    // Render Causal Graph (Neo4j)
    renderGraph('review-graph-panel', memberId);
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

  // 7b. Read Evidence Bundle button handler
  document.getElementById('btn-read-bundle')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-read-bundle');
    const evidenceActions = document.getElementById('evidence-actions');
    const bundlePath = evidenceActions?.dataset.bundle;
    const cptCode = evidenceActions?.dataset.cpt;
    
    if (!bundlePath) return;
    btn.disabled = true; btn.textContent = '🧠 Reading...';
    
    try {
      const res = await fetch('/agent/read-bundle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ bundlePath, cptCode })
      });
      const data = await res.json();
      
      if (data.extracted) {
        const ex = data.extracted;
        // Auto-populate form from extracted data
        if (ex.clinicalSummary) inputNotes.value = ex.clinicalSummary;
        
        // Show validation results as a summary popup
        let summaryHtml = '<strong>📎 AI Bundle Analysis:</strong><br>';
        if (ex.criteriaFindings && ex.criteriaFindings.length > 0) {
          ex.criteriaFindings.forEach(f => {
            const icon = f.met ? '✓' : '✗';
            const color = f.met ? '#059669' : '#dc2626';
            summaryHtml += `<span style="color:${color}">${icon} ${f.criterion}: ${f.finding}</span><br>`;
          });
        }
        if (ex.missingDocumentation && ex.missingDocumentation.length > 0) {
          summaryHtml += `<br><strong style="color:#d97706;">Missing:</strong> ${ex.missingDocumentation.join(', ')}`;
        }
        if (ex.validationSummary) {
          summaryHtml += `<br><strong>${ex.validationSummary}</strong>`;
        }
        
        const popup = document.createElement('div');
        popup.className = 'notes-summary-popup';
        popup.innerHTML = summaryHtml;
        const existing = inputNotes.parentElement.querySelector('.notes-summary-popup');
        if (existing) existing.remove();
        inputNotes.parentElement.insertBefore(popup, inputNotes);
        
        showToast('📎 Bundle read and validated by AI', 'success');
      } else {
        showToast(data.error || 'Failed to read bundle', 'error');
      }
    } catch(e) {
      showToast(`Error: ${e.message}`, 'error');
    } finally {
      btn.disabled = false; btn.textContent = '📎 Read Bundle with AI';
    }
  });

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
      clinicalNotes: inputNotes.value.trim(),
      evidenceBundle: document.getElementById('evidence-actions')?.dataset?.bundle || ''
    };

    // Reset UI
    [1,2,3,4,5].forEach(s => {
      const el = document.getElementById(`flow-step-${s}`);
      if (el) {
        el.className = 'flow-step';
        const disc = el.querySelector('.flow-step-disclosure');
        if (disc) disc.classList.remove('disc-active');
      }
    });
    outcomeBadge.textContent = 'Processing...';
    outcomeReason.textContent = 'Calling AI agent...';
    letterContent.textContent = '';
    timeline.innerHTML = '<div class="timeline-empty-state">🤖 Agent processing...</div>';
    const flow1 = document.getElementById('flow-step-1');
    if (flow1) flow1.className = 'flow-step active';

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
      // Light up stages (only if elements exist)
      for (let s = 1; s <= 5; s++) {
        const el = document.getElementById(`flow-step-${s}`);
        if (el && s <= maxStage) {
          el.className = outcome.status === 'approved' || s < maxStage ? 'flow-step passed' : 'flow-step warning';
        }
      }
      // If there were failed criteria, mark stage 4 as warning (only if elements exist)
      if (outcome.status !== 'approved') {
        const flow4 = document.getElementById('flow-step-4');
        if (flow4) flow4.className = 'flow-step warning';
        const flow5 = document.getElementById('flow-step-5');
        if (flow5) flow5.className = 'flow-step warning';
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

      // Update disclosure panel (only if elements exist)
      const discBadge = document.getElementById('disclosure-stage-badge');
      const discDisclosed = document.getElementById('disc-disclosed');
      const discWithheld = document.getElementById('disc-withheld');
      if (discBadge) {
        discBadge.textContent = 'Complete';
        discBadge.className = 'disclosure-badge complete';
      }
      if (discDisclosed) discDisclosed.textContent = `Policy: ${outcome.policyUsed}`;
      if (discWithheld) discWithheld.textContent = 'None — all disclosed';

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

      // Render Challenger Agent result
      const challengerCard = document.getElementById('challenger-card');
      const challengerBody = document.getElementById('challenger-body');
      const challengerBadge = document.getElementById('challenger-verdict-badge');
      
      if (outcome.challenger && outcome.challenger.verdict) {
        challengerCard.style.display = 'block';
        const ch = outcome.challenger;
        const verdictClass = ch.verdict === 'AGREE' ? 'badge-agree' : ch.verdict === 'CHALLENGE' ? 'badge-challenge' : 'badge-concern';
        const verdictIcon = ch.verdict === 'AGREE' ? '✓' : ch.verdict === 'CHALLENGE' ? '🚩' : '⚠';
        challengerBadge.textContent = `${verdictIcon} ${ch.verdict} (${ch.confidence}/10)`;
        challengerBadge.className = `badge ${verdictClass}`;
        
        let bodyHtml = '';
        
        // Red flag banner if formal challenge
        if (ch.formalChallenge) {
          bodyHtml += `<div class="challenger-red-flag">🚩 FORMAL CHALLENGE — Decision overridden. Sent to Medical Director with findings below.</div>`;
          outcomeCard.className = 'glass-card outcome-card escalated';
          outcomeBadge.textContent = outcome.decision;
        } else if (ch.verdict === 'AGREE') {
          bodyHtml += `<div class="challenger-green-flag">✅ QUALITY CONFIRMED — Challenger reviewed and agrees with the decision. Documentation is adequate.</div>`;
        } else if (ch.verdict === 'CONCERN') {
          bodyHtml += `<div class="challenger-amber-flag">⚠️ MINOR CONCERN — Documentation could be stronger but does not warrant override.</div>`;
        }
        
        bodyHtml += `<div class="challenger-reasoning">${ch.reasoning || ''}</div>`;
        if (ch.findings && ch.findings.length > 0) {
          bodyHtml += '<div class="challenger-findings">';
          ch.findings.forEach(f => { bodyHtml += `<div class="challenger-finding">${f}</div>`; });
          bodyHtml += '</div>';
        }
        if (ch.recommendation) {
          bodyHtml += `<div style="margin-top:0.4rem;font-size:0.72rem;color:var(--text-muted);"><strong>Recommendation:</strong> ${ch.recommendation}</div>`;
        }
        challengerBody.innerHTML = bodyHtml;
      } else {
        challengerCard.style.display = 'none';
      }

      // Store execution references globally
      const currentRequestId = activeCaseId || 'req-001';
      const currentMemberId = request.memberId || 'MEM-4401';
      
      window.__lastRequestId = currentRequestId;
      window.__lastMemberId = currentMemberId;
      window.__lastExecutionId = outcome.executionId || 'exec-001';
      
      // Refresh cockpit visual panels
      renderBeacon('review-beacon-panel', currentRequestId, { autoRefresh: false });
      renderAxisweave('review-axisweave-panel', currentRequestId);
      renderGraph('review-graph-panel', currentMemberId);

    } catch (e) {
      outcomeBadge.textContent = 'Error';
      outcomeReason.textContent = e.message;
      showToast(`Agent error: ${e.message}`, 'error');
    }
  });

  // Init UI
  await renderPresets();
  renderRulesToggles();

  // Cockpit Tab Switching Listener
  document.querySelectorAll('.cockpit-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tabName = btn.dataset.tab;
      const card = btn.closest('.cockpit-tab-card');
      
      // Update active tab button style
      card.querySelectorAll('.cockpit-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Toggle tab content visibility
      card.querySelectorAll('.cockpit-tab-content').forEach(content => {
        content.style.display = content.id === `cockpit-content-${tabName}` ? 'block' : 'none';
      });
      
      // Trigger lazy panel refresh on view switch to graph or axisweave if last IDs exist
      const reqId = window.__lastRequestId || activeCaseId || 'req-001';
      const memId = window.__lastMemberId || inputMemberId.value || 'MEM-4401';
      
      if (tabName === 'graph') {
        renderGraph('review-graph-panel', memId);
      } else if (tabName === 'axisweave') {
        renderAxisweave('review-axisweave-panel', reqId);
      }
    });
  });

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
    const activeTabEl = document.querySelector('.sidebar-tab.active');
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
