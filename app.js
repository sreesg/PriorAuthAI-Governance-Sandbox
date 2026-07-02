import { PRESET_CASES } from './cases.js';
import { PriorAuthAgent } from './agent.js';
import { render as renderBeacon } from './static/crf/beacon_harness_viz.js';
import { render as renderAxisweave } from './static/crf/axisweave_context_panel.js';
import { render as renderGraph } from './static/crf/causal_graph_viz.js';

let activeCaseId = null;

document.addEventListener('DOMContentLoaded', async () => {
  const agent = new PriorAuthAgent();
  let activeFile = "rules_declaration.md";

  // Asset URLs and video sources will load concurrently in initializeApp()

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

  // Initialize workspacePolicies list
  let workspacePolicies = [];

  // Bind workspace dropdown and generation listeners immediately
  initPolicyWorkspaceListeners();

  function initPolicyWorkspaceListeners() {
    const select = document.getElementById('policy-select');
    
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
      refreshSkillsPills(workspacePolicies);
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
          if (action === 'skills') refreshSkillsPills(workspacePolicies);
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
      const builderCard = document.getElementById('ws-case-builder-card');
      if (builderCard) builderCard.style.display = 'block';
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
  async function refreshSkillsPills(policies) {
    activeSkillsPills.innerHTML = '';
    
    // Get JS agent skills
    const jsSkills = Object.keys(agent.skills);
    
    // Get generated skills from all policy files in parallel
    let backendSkills = [];
    try {
      let policyList = policies;
      if (!policyList) {
        const res = await fetch('/agent/policies');
        policyList = await res.json();
      }
      
      const skillPromises = policyList.map(async (p) => {
        try {
          const sRes = await fetch(`/policies/${p.policyId}_skills.json?t=${Date.now()}`);
          if (sRes.ok) {
            const data = await sRes.json();
            return (data.skills || []).map(s => s.skillName).filter(Boolean);
          }
        } catch(_) {}
        return [];
      });
      
      const skillsArrays = await Promise.all(skillPromises);
      skillsArrays.forEach(skills => {
        skills.forEach(sName => {
          if (!backendSkills.includes(sName)) {
            backendSkills.push(sName);
          }
        });
      });
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
    const outcomeDetails = document.getElementById('outcome-details');
    if (outcomeDetails) outcomeDetails.style.display = 'none';
    const aiInsights = document.getElementById('ai-insights-panel');
    if (aiInsights) aiInsights.style.display = 'none';
    const aiModelBadge = document.getElementById('ai-model-badge');
    if (aiModelBadge) { aiModelBadge.textContent = ''; aiModelBadge.className = 'badge'; }
    const npiBadge = document.getElementById('npi-status-badge');
    if (npiBadge) { npiBadge.textContent = ''; npiBadge.className = 'badge'; }
    
    // Remove any note summary/quality popups
    document.querySelectorAll('.notes-summary-popup').forEach(el => el.remove());
    
    // Hide bundle AI output panel
    const bundleAiOutput = document.getElementById('bundle-ai-output');
    if (bundleAiOutput) bundleAiOutput.style.display = 'none';
    
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

    // Load Raw Evidence Documents
    loadRawEvidenceDocuments(memberId);
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

  async function loadRawEvidenceDocuments(memberId) {
    const listEl = document.getElementById('raw-evidence-list');
    const viewerEl = document.getElementById('raw-evidence-viewer');
    const titleEl = document.getElementById('raw-evidence-title');
    if (!listEl || !viewerEl || !titleEl) return;
    
    listEl.innerHTML = '<span style="font-size:0.65rem;color:var(--text-muted);">Loading documents...</span>';
    viewerEl.textContent = 'Select a document from the list to review its full text content.';
    titleEl.textContent = 'Select a document...';
    
    try {
      const res = await fetch(`/api/evidence-documents/${memberId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const docs = data.documents || [];
      if (docs.length === 0) {
        listEl.innerHTML = '<span style="font-size:0.65rem;color:var(--text-muted);">No documents found.</span>';
        return;
      }
      
      listEl.innerHTML = '';
      docs.forEach(doc => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn-ai-action';
        btn.style.textAlign = 'left';
        btn.style.width = '100%';
        btn.style.fontSize = '0.62rem';
        btn.style.padding = '0.25rem 0.4rem';
        btn.style.whiteSpace = 'nowrap';
        btn.style.overflow = 'hidden';
        btn.style.textOverflow = 'ellipsis';
        btn.textContent = doc.name;
        btn.title = doc.name;
        
        btn.addEventListener('click', async () => {
          listEl.querySelectorAll('button').forEach(b => b.style.borderColor = 'var(--card-border)');
          btn.style.borderColor = 'var(--accent-blue)';
          titleEl.textContent = doc.name;
          viewerEl.textContent = 'Reading document content from S3...';
          
          try {
            const contentRes = await fetch(`/api/evidence-document/content?path=${encodeURIComponent(doc.path)}`);
            if (!contentRes.ok) throw new Error(`HTTP ${contentRes.status}`);
            const text = await contentRes.text();
            viewerEl.textContent = text || '(Empty document)';
          } catch (err) {
            viewerEl.textContent = `Error loading document content: ${err.message}`;
          }
        });
        
        listEl.appendChild(btn);
      });
    } catch (e) {
      console.error('Failed to load raw evidence:', e);
      listEl.innerHTML = `<span style="font-size:0.65rem;color:var(--accent-red);">Load failed: ${e.message}</span>`;
    }
  }

  // 7b. Read Evidence Bundle button handler
  document.getElementById('btn-read-bundle')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-read-bundle');
    const evidenceActions = document.getElementById('evidence-actions');
    const bundlePath = evidenceActions?.dataset.bundle;
    const cptCode = evidenceActions?.dataset.cpt;
    
    if (!bundlePath) return;
    btn.disabled = true; btn.textContent = '🧠 Reading PDF...';
    
    try {
      const res = await fetch('/agent/read-bundle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ bundlePath, cptCode })
      });
      const data = await res.json();
      
      if (data.extracted) {
        const ex = data.extracted;
        const outputPanel = document.getElementById('bundle-ai-output');
        const findingsEl = document.getElementById('bundle-ai-findings');
        
        let html = '';
        
        // Clinical summary extracted from PDF
        if (ex.clinicalSummary) {
          html += `<div style="margin-bottom:0.4rem;"><strong style="color:var(--text-main);">📝 Clinical Summary (from PDF):</strong></div>`;
          html += `<div style="padding:0.3rem 0.5rem;background:rgba(0,0,0,0.1);border-radius:4px;margin-bottom:0.5rem;font-size:0.62rem;white-space:pre-wrap;">${ex.clinicalSummary}</div>`;
        }
        
        // Criteria findings
        if (ex.criteriaFindings && ex.criteriaFindings.length > 0) {
          html += `<div style="margin-bottom:0.2rem;"><strong style="color:var(--text-main);">✓/✗ Criteria Findings:</strong></div>`;
          ex.criteriaFindings.forEach(f => {
            const icon = f.met ? '✓' : '✗';
            const color = f.met ? 'var(--accent-green)' : 'var(--accent-red)';
            html += `<div style="padding:0.1rem 0;"><span style="color:${color};font-weight:700;">${icon}</span> ${f.criterion}: <em>${f.finding}</em></div>`;
          });
        }
        
        // Missing documentation
        if (ex.missingDocumentation && ex.missingDocumentation.length > 0) {
          html += `<div style="margin-top:0.4rem;color:var(--accent-orange);font-weight:600;">⚠ Missing Documentation:</div>`;
          ex.missingDocumentation.forEach(m => {
            html += `<div style="padding:0.1rem 0.3rem;">• ${m}</div>`;
          });
        }
        
        // Validation summary
        if (ex.validationSummary) {
          html += `<div style="margin-top:0.4rem;padding:0.3rem 0.5rem;background:rgba(139,92,246,0.1);border-radius:4px;font-weight:600;">${ex.validationSummary}</div>`;
        }
        
        findingsEl.innerHTML = html;
        outputPanel.style.display = 'block';
        
        showToast('📎 Bundle analyzed — findings displayed below notes', 'success');
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
      
      // Build clear, human-readable outcome reason
      const isApproved = outcome.decision === 'Approved';
      const isEscalated = outcome.decision.includes('Escalated') || outcome.decision.includes('Flagged');
      
      if (isApproved) {
        outcomeReason.textContent = `All clinical criteria satisfied — approved per ${outcome.policyName} guidelines.`;
      } else if (isEscalated) {
        const failedCriteria = (outcome.criteriaMet || []).filter(c => !c.met);
        if (failedCriteria.length > 0) {
          outcomeReason.textContent = `${failedCriteria.length} criteria not met — escalated to Medical Director for human review.`;
        } else {
          outcomeReason.textContent = outcome.reason;
        }
      } else {
        outcomeReason.textContent = outcome.reason;
      }
      
      // Build expanded decision details
      const detailsEl = document.getElementById('outcome-details');
      const policySec = document.getElementById('outcome-policy-section');
      const criteriaSec = document.getElementById('outcome-criteria-section');
      const actionSec = document.getElementById('outcome-action-section');
      const challengerSummary = document.getElementById('outcome-challenger-summary');
      
      if (detailsEl) {
        detailsEl.style.display = 'block';
        
        // Policy section
        policySec.innerHTML = `
          <div style="font-size:0.68rem;font-weight:700;color:var(--text-main);margin-bottom:0.2rem;">📋 Policy Applied</div>
          <div style="font-size:0.62rem;color:var(--text-muted);padding-left:0.5rem;">
            <strong>${outcome.policyUsed || 'N/A'}</strong>: ${outcome.policyName || 'Default'} (${outcome.category || 'General'})
          </div>
        `;
        
        // Criteria breakdown
        const criteria = outcome.criteriaMet || [];
        if (criteria.length > 0) {
          let critHtml = '<div style="font-size:0.68rem;font-weight:700;color:var(--text-main);margin-bottom:0.2rem;">✓/✗ Criteria Results</div>';
          critHtml += '<div style="padding-left:0.5rem;">';
          criteria.forEach(c => {
            const icon = c.met ? '✅' : '❌';
            const color = c.met ? 'var(--accent-green)' : 'var(--accent-red)';
            critHtml += `<div style="font-size:0.62rem;padding:0.1rem 0;display:flex;align-items:baseline;gap:0.3rem;">
              <span>${icon}</span>
              <span style="color:${color};font-weight:600;">${c.met ? 'MET' : 'NOT MET'}</span>
              <span style="color:var(--text-main);">${c.name}</span>
              <span style="color:var(--text-muted);font-style:italic;margin-left:auto;">— ${c.detail}</span>
            </div>`;
          });
          critHtml += '</div>';
          criteriaSec.innerHTML = critHtml;
        }
        
        // Action required section
        if (isEscalated) {
          const failedCriteria = criteria.filter(c => !c.met);
          let actionHtml = '<div style="font-size:0.68rem;font-weight:700;color:var(--accent-orange);margin-bottom:0.2rem;">⚠ What Needs to Happen</div>';
          actionHtml += '<div style="padding-left:0.5rem;font-size:0.62rem;color:var(--text-main);">';
          if (failedCriteria.length > 0) {
            actionHtml += '<div style="margin-bottom:0.2rem;">To get this approved, the provider needs to submit:</div>';
            failedCriteria.forEach(c => {
              actionHtml += `<div style="padding:0.1rem 0;">• Documentation proving: <strong>${c.name.replace(/CRIT-\d+\s*/, '')}</strong></div>`;
            });
          } else {
            actionHtml += '<div>Medical Director will review the evidence for any ambiguities flagged by the quality system.</div>';
          }
          actionHtml += '</div>';
          actionSec.innerHTML = actionHtml;
        } else if (isApproved) {
          actionSec.innerHTML = `
            <div style="font-size:0.68rem;font-weight:700;color:var(--accent-green);margin-bottom:0.2rem;">✅ Approved — No Further Action</div>
            <div style="padding-left:0.5rem;font-size:0.62rem;color:var(--text-muted);">
              Authorization is granted. Notice letter generated for provider and patient.
            </div>
          `;
        } else {
          actionSec.innerHTML = '';
        }
        
        // Challenger summary (brief)
        challengerSummary.innerHTML = '';
      }

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
      
      // Populating the Evidence Cockpit Summary Tab
      const semanticQueryEl = document.getElementById('cockpit-semantic-query');
      if (semanticQueryEl) {
        semanticQueryEl.textContent = outcome.semanticQuery || 'N/A (Pattern matching used)';
      }

      const vectorFindingsEl = document.getElementById('cockpit-vector-findings');
      if (vectorFindingsEl && outcome.retrievedEvidence) {
        if (outcome.retrievedEvidence.length === 0) {
          vectorFindingsEl.innerHTML = '<span style="color:var(--text-muted);">No evidence chunks retrieved.</span>';
        } else {
          let vfHtml = '';
          outcome.retrievedEvidence.slice(0, 3).forEach((doc, idx) => {
            const scoreColor = doc.score >= 0.8 ? 'var(--accent-green)' : doc.score >= 0.5 ? 'var(--accent-blue)' : 'var(--accent-orange)';
            const pdfLink = doc.s3_key ? `/agent/pdf/${doc.s3_key}` : '#';
            vfHtml += `<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); border-radius:4px; padding:0.3rem; margin-bottom:0.25rem;">
              <div style="display:flex; justify-content:space-between; font-weight:600; font-size:0.6rem; color:var(--text-main);">
                <span>#${idx+1} ${doc.doc_type || 'Evidence Doc'}</span>
                <span style="color:${scoreColor};">${doc.score.toFixed(2)}</span>
              </div>
              <p style="margin:0.2rem 0; color:var(--text-muted); font-size:0.58rem; line-height:1.35;">"${doc.text.substring(0, 140)}..."</p>
              ${doc.s3_key ? `<a href="${pdfLink}" target="_blank" style="color:var(--accent-blue); text-decoration:none; font-size:0.55rem; font-family:monospace;">View Source PDF →</a>` : ''}
            </div>`;
          });
          vectorFindingsEl.innerHTML = vfHtml;
        }
      }

      const graphFindingsEl = document.getElementById('cockpit-graph-findings');
      if (graphFindingsEl && outcome.graphState) {
        const gs = outcome.graphState;
        if (!gs || gs.diagnosis_count === 0) {
          graphFindingsEl.innerHTML = '<span style="color:var(--text-muted);">No ontology graph records found.</span>';
        } else {
          let gfHtml = '';
          if (gs.diagnoses && gs.diagnoses.length > 0) {
            gfHtml += `<div style="font-weight:600; color:var(--accent-blue); margin-bottom:0.2rem;">🩺 Diagnoses:</div>`;
            gs.diagnoses.forEach(d => {
              gfHtml += `<div style="padding-left:0.3rem; color:var(--text-main); font-size:0.58rem;">• ${d.description || d.condition_code} (${d.condition_code})</div>`;
            });
          }
          if (gs.failed_therapies && gs.failed_therapies.length > 0) {
            gfHtml += `<div style="font-weight:600; color:var(--accent-red); margin-top:0.3rem; margin-bottom:0.2rem;">❌ Failed Therapies:</div>`;
            gs.failed_therapies.forEach(ft => {
              gfHtml += `<div style="padding-left:0.3rem; color:var(--text-muted); font-size:0.58rem;">• ${ft.drug || ft.therapy_type} — Outcome: ${ft.outcome}</div>`;
            });
          }
          if (gs.prescriptions && gs.prescriptions.length > 0) {
            gfHtml += `<div style="font-weight:600; color:var(--accent-green); margin-top:0.3rem; margin-bottom:0.2rem;">💊 Prescriptions:</div>`;
            gs.prescriptions.slice(0, 3).forEach(p => {
              gfHtml += `<div style="padding-left:0.3rem; color:var(--text-muted); font-size:0.58rem;">• ${p.drug} (${p.dose})</div>`;
            });
          }
          graphFindingsEl.innerHTML = gfHtml || '<span style="color:var(--text-muted);">No records.</span>';
        }
      }

      const policyChecklist = document.getElementById('cockpit-policy-checklist');
      if (policyChecklist && outcome.criteriaMet && outcome.criteriaMet.length > 0) {
        let pcHtml = '<table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.65rem;">';
        pcHtml += '<thead style="border-bottom:1px solid rgba(255,255,255,0.08);">';
        pcHtml += '<tr><th style="padding:0.2rem 0.4rem; color:var(--text-muted);">Criteria</th><th style="padding:0.2rem 0.4rem; color:var(--text-muted);">Status</th><th style="padding:0.2rem 0.4rem; color:var(--text-muted);">Details</th></tr>';
        pcHtml += '</thead><tbody>';
        outcome.criteriaMet.forEach(c => {
          const statusIcon = c.met ? '🟢' : '🔴';
          const statusText = c.met ? 'Passed' : 'Failed';
          const statusColor = c.met ? 'var(--accent-green)' : 'var(--accent-red)';
          pcHtml += `<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
            <td style="padding:0.3rem 0.4rem; font-weight:600; color:var(--text-main);">${c.name}</td>
            <td style="padding:0.3rem 0.4rem; color:${statusColor}; font-weight:700;">${statusIcon} ${statusText}</td>
            <td style="padding:0.3rem 0.4rem; color:var(--text-muted);">${c.detail}</td>
          </tr>`;
        });
        pcHtml += '</tbody></table>';
        policyChecklist.innerHTML = pcHtml;
      }

      const challengerAlert = document.getElementById('cockpit-challenger-alert');
      if (challengerAlert && outcome.challenger) {
        const ch = outcome.challenger;
        if (ch.formalChallenge) {
          challengerAlert.style.display = 'block';
          challengerAlert.innerHTML = `<span style="font-weight:700; color:var(--accent-red);">🚩 CHALLENGER OVERRIDE:</span> ${ch.reasoning || ''}`;
        } else {
          challengerAlert.style.display = 'none';
        }
      }

      // Render retrieved evidence documents panel in Execution Logs tab
      const evidencePanel = document.getElementById('retrieved-evidence-panel');
      if (evidencePanel && outcome.retrievedEvidence && outcome.retrievedEvidence.length > 0) {
        evidencePanel.style.display = 'block';
        let evHtml = `<div style="font-size:0.72rem;font-weight:700;color:var(--text-main);margin-bottom:0.4rem;">
          📎 Retrieved Evidence (${outcome.retrievedEvidence.length} chunks via Axisweave)
        </div>`;
        if (outcome.semanticQuery) {
          evHtml += `<div style="font-size:0.6rem;color:var(--text-muted);background:var(--bg-secondary);padding:0.3rem 0.5rem;border-radius:6px;margin-bottom:0.4rem;font-family:var(--font-mono);">
            🔍 Query: "${outcome.semanticQuery.substring(0, 100)}..."
          </div>`;
        }
        evHtml += '<div style="max-height:300px;overflow-y:auto;">';
        outcome.retrievedEvidence.forEach((doc, i) => {
          const scoreColor = doc.score >= 0.8 ? 'var(--accent-green)' : doc.score >= 0.5 ? 'var(--accent-blue)' : 'var(--accent-orange)';
          const pdfLink = doc.s3_key ? `/agent/pdf/${doc.s3_key}` : '#';
          evHtml += `<div style="padding:0.4rem;margin-bottom:0.3rem;background:var(--bg-secondary);border:1px solid var(--card-border);border-radius:6px;border-left:3px solid ${scoreColor};">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:0.65rem;font-weight:600;color:var(--text-main);">${doc.doc_type || 'document'}</span>
              <span style="font-size:0.6rem;font-weight:700;color:${scoreColor};">${doc.score.toFixed(2)}</span>
            </div>
            <div style="font-size:0.6rem;color:var(--text-muted);margin-top:0.2rem;line-height:1.3;">${doc.text.substring(0, 150)}...</div>
            <div style="display:flex;gap:0.5rem;margin-top:0.2rem;font-size:0.55rem;font-family:var(--font-mono);color:var(--text-muted);">
              <span>${doc.document_id ? doc.document_id.split('/').pop() : ''}</span>
              ${doc.s3_key ? `<a href="${pdfLink}" target="_blank" style="color:var(--accent-blue);text-decoration:none;">View PDF →</a>` : ''}
            </div>
          </div>`;
        });
        evHtml += '</div>';
        evidencePanel.innerHTML = evHtml;
      } else if (evidencePanel) {
        evidencePanel.style.display = 'none';
      }

      // Render graph state summary in Execution Logs tab
      const graphStatePanel = document.getElementById('graph-state-panel');
      if (graphStatePanel && outcome.graphState && outcome.graphState.diagnosis_count > 0) {
        graphStatePanel.style.display = 'block';
        const gs = outcome.graphState;
        let gsHtml = `<div style="font-size:0.72rem;font-weight:700;color:var(--text-main);margin-bottom:0.4rem;">🕸️ Patient Graph State (Neo4j)</div>`;
        gsHtml += `<div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.3rem;">
          <span style="font-size:0.62rem;background:var(--bg-secondary);padding:0.15rem 0.4rem;border-radius:8px;">🩺 ${gs.diagnosis_count} diagnoses</span>
          <span style="font-size:0.62rem;background:var(--bg-secondary);padding:0.15rem 0.4rem;border-radius:8px;">💊 ${gs.rx_count} prescriptions</span>
          <span style="font-size:0.62rem;background:var(--bg-secondary);padding:0.15rem 0.4rem;border-radius:8px;">🏥 ${gs.therapy_count} therapies</span>
        </div>`;
        if (gs.failed_therapies && gs.failed_therapies.length > 0) {
          gsHtml += `<div style="font-size:0.62rem;font-weight:600;color:var(--accent-red);margin-top:0.3rem;">Failed/Inadequate Therapies:</div>`;
          gs.failed_therapies.forEach(ft => {
            gsHtml += `<div style="font-size:0.58rem;color:var(--text-muted);padding:0.1rem 0;">❌ ${ft.drug} ${ft.dose} — ${ft.outcome}</div>`;
          });
        }
        graphStatePanel.innerHTML = gsHtml;
      } else if (graphStatePanel) {
        graphStatePanel.style.display = 'none';
      }

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

  async function loadS3PdfPicker() {
    const picker = document.getElementById('s3-policy-picker');
    if (!picker) return;
    try {
      picker.innerHTML = '<option value="">-- Scanning S3 Bucket --</option>';
      const res = await fetch('/agent/list-s3-pdfs');
      const data = await res.json();
      picker.innerHTML = '<option value="">-- Select from S3 bucket --</option>';
      if (data.files && data.files.length > 0) {
        data.files.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f;
          opt.textContent = f;
          picker.appendChild(opt);
        });
      } else {
        picker.innerHTML = '<option value="">No PDFs found in S3</option>';
      }
    } catch(e) {
      console.error("Failed to load S3 PDFs:", e);
      picker.innerHTML = '<option value="">Error listing S3 files</option>';
    }
  }

  async function runPolicyIngestion() {
    const s3Select = document.getElementById('s3-policy-picker');
    const uploadInput = document.getElementById('upload-policy-input');
    const btn = document.getElementById('btn-ingest-policy');
    const stepperContainer = document.getElementById('ingest-stepper-container');
    const percentEl = document.getElementById('ingest-overall-percent');
    
    const steps = {
      read: document.getElementById('step-read-pdf'),
      extract: document.getElementById('step-extract-criteria'),
      rego: document.getElementById('step-compile-rego'),
      build: document.getElementById('step-build-suite'),
      cases: document.getElementById('step-generate-cases')
    };

    // Reset step styles
    Object.values(steps).forEach(s => setStepStatus(s, 'pending', s.textContent.substring(2)));
    percentEl.textContent = '0%';

    let payload = {};
    
    // Check if uploaded local file exists
    if (uploadInput.files && uploadInput.files.length > 0) {
      const file = uploadInput.files[0];
      stepperContainer.style.display = 'block';
      setStepStatus(steps.read, 'active', '1. Reading local PDF file...');
      percentEl.textContent = '10%';
      
      const reader = new FileReader();
      const base64Promise = new Promise((resolve) => {
        reader.onload = () => resolve(reader.result.split(',')[1]);
      });
      reader.readAsDataURL(file);
      const b64Data = await base64Promise;
      payload = {
        uploadedFile: b64Data,
        filename: file.name
      };
    } else if (s3Select.value) {
      stepperContainer.style.display = 'block';
      setStepStatus(steps.read, 'active', '1. Requesting S3 file...');
      percentEl.textContent = '10%';
      payload = {
        s3Key: s3Select.value
      };
    } else {
      showToast("Please upload a local PDF or select a PDF file from S3 first.", "error");
      return;
    }

    btn.disabled = true;
    btn.textContent = '⏳ Processing Policy...';
    
    try {
      setStepStatus(steps.read, 'success', '1. PDF file read completed');
      setStepStatus(steps.extract, 'active', '2. ClinicalNLP Engine extracting criteria...');
      percentEl.textContent = '30%';
      
      await new Promise(r => setTimeout(r, 800));
      
      const res = await fetch('/agent/ingest-policy-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Ingestion request failed");
      }
      
      const data = await res.json();
      
      setStepStatus(steps.extract, 'success', '2. Clinical criteria extracted successfully');
      setStepStatus(steps.rego, 'active', '3. Rebuilding Rego engine and rules declaration...');
      percentEl.textContent = '60%';
      
      await new Promise(r => setTimeout(r, 600));
      setStepStatus(steps.rego, 'success', '3. Rego and rules_declaration.md recompiled');
      setStepStatus(steps.build, 'active', '4. Generating skills and hooks for workspace...');
      percentEl.textContent = '80%';
      
      await new Promise(r => setTimeout(r, 600));
      setStepStatus(steps.build, 'success', '4. Clinical validation skills and hooks designed');
      setStepStatus(steps.cases, 'active', '5. Designing Approved & Escalated cases...');
      percentEl.textContent = '95%';
      
      await new Promise(r => setTimeout(r, 800));
      setStepStatus(steps.cases, 'success', '5. Synthetic mock review cases completed');
      percentEl.textContent = '100%';
      
      showToast(data.message || "Policy suite ingested successfully!", "success");
      await reloadWorkspaceAfterIngest(data.policies[0]?.policyId);
      
    } catch (e) {
      console.error(e);
      Object.values(steps).forEach(s => {
        if (s.classList.contains('active')) {
          setStepStatus(s, 'fail', s.textContent.substring(2) + ' - FAILED');
        }
      });
      showToast(`Ingestion failed: ${e.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = '⚡ Ingest & Build Policy Suite';
      uploadInput.value = '';
    }
  }

  function setStepStatus(element, status, text) {
    element.className = `step-node ${status}`;
    const cleanText = text.replace(/^[⚪⏳✓✗]\s*/, '');
    if (status === 'success') {
      element.innerHTML = `✓ ${cleanText}`;
    } else if (status === 'active') {
      element.innerHTML = `⏳ ${cleanText}`;
    } else if (status === 'fail') {
      element.innerHTML = `✗ ${cleanText}`;
    } else {
      element.innerHTML = `⚪ ${cleanText}`;
    }
  }

  async function reloadWorkspaceAfterIngest(newPolicyId) {
    const res = await fetch('/agent/policies');
    const policies = await res.json();
    workspacePolicies = policies || [];
    
    const select = document.getElementById('policy-select');
    select.innerHTML = '';
    workspacePolicies.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.policyId;
      opt.textContent = `${p.name} (${p.category}) — ${p.cptCodes.join(', ')}`;
      select.appendChild(opt);
    });
    
    if (newPolicyId) {
      select.value = newPolicyId;
      showWorkspacePolicy(newPolicyId);
    } else if (workspacePolicies.length > 0) {
      showWorkspacePolicy(workspacePolicies[0].policyId);
    }
    
    await refreshActiveFileView();
    await refreshSkillsPills(workspacePolicies);
    await renderPresets();
  }

  // Concurrent Initialization logic
  async function initializeApp() {
    // Launch non-dependent asset and preset fetches concurrently
    const assetPromise = fetch('/agent/asset-urls')
      .then(r => r.json())
      .then(assets => {
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
      }).catch(e => {
        console.warn('Asset URLs not available, using local fallback');
        const archImg = document.getElementById('arch-image');
        if (archImg) { archImg.src = './PA Agentic architecture.png'; archImg.style.display = 'block'; }
        const videoSrc = document.getElementById('video-source');
        if (videoSrc) videoSrc.src = './PA agent.mp4';
      });

    const casesPromise = renderPresets();

    const fileViewPromise = refreshActiveFileView().catch(e => {
      console.error("Initial file view load failed:", e);
    });

    const policiesPromise = fetch('/agent/policies')
      .then(r => r.json())
      .then(async (policies) => {
        workspacePolicies = policies || [];
        const select = document.getElementById('policy-select');
        select.innerHTML = '';
        workspacePolicies.forEach(p => {
          const opt = document.createElement('option');
          opt.value = p.policyId;
          opt.textContent = `${p.name} (${p.category}) — ${p.cptCodes.join(', ')}`;
          select.appendChild(opt);
        });
        if (workspacePolicies.length > 0) showWorkspacePolicy(workspacePolicies[0].policyId);
        await refreshSkillsPills(workspacePolicies);
      }).catch(e => {
        console.error("Failed to load policies during init:", e);
        const select = document.getElementById('policy-select');
        if (select) select.innerHTML = '<option>Failed to load policies</option>';
      });

    const s3PdfPromise = loadS3PdfPicker();

    // Await all concurrently running initialization tasks
    await Promise.all([assetPromise, casesPromise, fileViewPromise, policiesPromise, s3PdfPromise]);
  }

  // Init UI
  await initializeApp();
  renderRulesToggles();

  // Ingestion Event Listeners
  const btnRefreshS3 = document.getElementById('btn-refresh-s3-pdfs');
  if (btnRefreshS3) {
    btnRefreshS3.addEventListener('click', loadS3PdfPicker);
  }
  const btnIngest = document.getElementById('btn-ingest-policy');
  if (btnIngest) {
    btnIngest.addEventListener('click', runPolicyIngestion);
  }

  const btnBuildCase = document.getElementById('btn-build-custom-case');
  if (btnBuildCase) {
    btnBuildCase.addEventListener('click', async () => {
      const select = document.getElementById('policy-select');
      const patientNameInput = document.getElementById('custom-case-patient-name');
      const memberIdInput = document.getElementById('custom-case-member-id');
      const scenarioSelect = document.getElementById('custom-case-scenario');
      
      const policyId = select.value;
      if (!policyId) {
        showToast("Please select a staged policy first.", "error");
        return;
      }
      
      const patientName = patientNameInput.value.trim() || 'John Doe';
      const memberId = memberIdInput.value.trim() || 'MEM-7701';
      const scenario = scenarioSelect.value;
      
      btnBuildCase.disabled = true;
      btnBuildCase.textContent = '🧪 Generating case & clinical PDF...';
      
      try {
        const res = await fetch('/agent/create-custom-case', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ policyId, patientName, memberId, scenario })
        });
        
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || "Failed to create custom case");
        }
        
        const data = await res.json();
        showToast(data.message || "Custom case created successfully!", "success");
        
        // Reload preset cases list on dashboard
        await renderPresets();
        
      } catch (e) {
        console.error(e);
        showToast(`Failed to create case: ${e.message}`, "error");
      } finally {
        btnBuildCase.disabled = false;
        btnBuildCase.textContent = '🧪 Generate Case & Clinical PDF';
      }
    });
  }

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
      } else if (tabName === 'rawevidence') {
        loadRawEvidenceDocuments(memId);
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
    const activeView = activeTabEl ? activeTabEl.getAttribute('data-view') : 'review';
    
    const context = { activeTab: activeView, activeView };
    
    // Last decision context (always include if available)
    const decision = outcomeBadge ? outcomeBadge.textContent : '';
    const reason = outcomeReason ? outcomeReason.textContent : '';
    if (decision && decision !== 'Ready to Run' && decision !== 'Processing...') {
      context.lastDecision = {
        decision: decision,
        reason: reason,
        policyName: reason.split('(')[0]?.trim() || ''
      };
    }

    // Last trace events (last 5 events from timeline)
    const traceItems = document.querySelectorAll('.timeline-item');
    if (traceItems.length > 0) {
      const lastTrace = [];
      const items = Array.from(traceItems).slice(-5);
      items.forEach(item => {
        const typeBadge = item.querySelector('.item-type-badge');
        const title = item.querySelector('.item-title');
        const msg = item.querySelector('.item-msg');
        lastTrace.push({
          type: typeBadge ? typeBadge.textContent.trim() : '',
          name: title ? title.textContent.trim() : '',
          msg: msg ? msg.textContent.trim().substring(0, 120) : ''
        });
      });
      context.lastTrace = lastTrace;
    }

    // Graph state (from last execution)
    const graphPanel = document.getElementById('graph-state-panel');
    if (graphPanel && graphPanel.style.display !== 'none') {
      context.graphState = graphPanel.textContent.substring(0, 400);
    }

    // Retrieved evidence summary
    const evidencePanel = document.getElementById('retrieved-evidence-panel');
    if (evidencePanel && evidencePanel.style.display !== 'none') {
      const evidenceChunks = evidencePanel.querySelectorAll('[style*="border-radius"]');
      context.retrievedEvidence = {
        count: evidenceChunks.length || 0,
        summary: evidencePanel.textContent.substring(0, 300)
      };
    }

    // Challenger verdict
    const challengerBadge = document.getElementById('challenger-verdict-badge');
    const challengerBody = document.getElementById('challenger-body');
    if (challengerBadge && challengerBadge.textContent.trim()) {
      context.challengerVerdict = {
        verdict: challengerBadge.textContent.trim(),
        reasoning: challengerBody ? challengerBody.textContent.substring(0, 300) : ''
      };
    }

    // Tab-specific context
    if (activeView === 'workspace') {
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
