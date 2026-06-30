/**
 * Sidebar Navigation — Panel switching for the Clinical Reasoning Fabric UI.
 *
 * Provides a sidebar navigation component that mounts/unmounts the 6 CRF panels:
 *   1. BEACON Harness Visualization
 *   2. Axisweave Context Panel
 *   3. Evidence Bundle Viewer
 *   4. Causal Graph Visualization
 *   5. SDOH Inference Display
 *   6. Medical Director Queue View
 *
 * Uses glass-card CSS pattern consistent with existing UI.
 * Does not interfere with existing index.html layout.
 *
 * Requirements: 15.8, 15.9
 */

import { BEACONHarnessVisualization } from './beacon_harness_viz.js';
import { AxisweaveContextPanel } from './axisweave_context_panel.js';
import { EvidenceBundleViewer } from './evidence_bundle_viewer.js';
import { CausalGraphVisualization } from './causal_graph_viz.js';
import { SDOHInferenceDisplay } from './sdoh_inference_display.js';
import { MedicalDirectorQueueView } from './md_queue_view.js';

/**
 * Panel definitions — the 6 registered panels.
 */
const PANELS = [
  { id: 'beacon-harness', name: 'BEACON Harness', icon: '🛡️', PanelClass: BEACONHarnessVisualization },
  { id: 'axisweave-context', name: 'Axisweave Context', icon: '📎', PanelClass: AxisweaveContextPanel },
  { id: 'evidence-bundle', name: 'Evidence Bundle', icon: '📦', PanelClass: EvidenceBundleViewer },
  { id: 'causal-graph', name: 'Causal Graph', icon: '🔗', PanelClass: CausalGraphVisualization },
  { id: 'sdoh-inference', name: 'SDOH Inference', icon: '🧠', PanelClass: SDOHInferenceDisplay },
  { id: 'md-queue', name: 'MD Queue', icon: '👨‍⚕️', PanelClass: MedicalDirectorQueueView },
];

/**
 * Creates and mounts the sidebar navigation into a container.
 *
 * @param {string} sidebarContainerId - DOM element ID for the sidebar.
 * @param {string} panelContainerId - DOM element ID where panels are rendered.
 * @param {object} [context] - Contextual IDs needed by panels.
 * @param {string} [context.requestId] - PA request ID for BEACON/Axisweave panels.
 * @param {string} [context.executionId] - Execution ID for Evidence Bundle panel.
 * @param {string} [context.memberId] - Member ID for Graph/SDOH panels.
 * @returns {object} Control object with switchPanel(), getActivePanel(), and destroy() methods.
 */
export function createSidebarNav(sidebarContainerId, panelContainerId, context = {}) {
  const sidebarContainer = document.getElementById(sidebarContainerId);
  const panelContainer = document.getElementById(panelContainerId);

  if (!sidebarContainer) {
    console.error(`[SidebarNav] Sidebar container #${sidebarContainerId} not found.`);
    return { switchPanel: () => {}, getActivePanel: () => null, destroy: () => {} };
  }
  if (!panelContainer) {
    console.error(`[SidebarNav] Panel container #${panelContainerId} not found.`);
    return { switchPanel: () => {}, getActivePanel: () => null, destroy: () => {} };
  }

  let activePanel = null;
  let activePanelInstance = null;

  // Render sidebar nav items
  sidebarContainer.innerHTML = buildSidebarHtml();

  // Attach click handlers to nav items
  PANELS.forEach(panel => {
    const navItem = document.getElementById(`nav-item-${panel.id}`);
    if (navItem) {
      navItem.addEventListener('click', () => switchPanel(panel.id));
    }
  });

  function buildSidebarHtml() {
    return `
      <nav class="crf-sidebar-nav glass-card" role="navigation" aria-label="Clinical Reasoning Fabric panels">
        <div class="sidebar-nav-header" style="padding:0.6rem 0.75rem;border-bottom:1px solid var(--card-border, rgba(255,255,255,0.06));">
          <span style="font-size:0.72rem;font-weight:700;color:var(--text-main, #e5e7eb);">CRF Panels</span>
        </div>
        <ul class="sidebar-nav-list" style="list-style:none;margin:0;padding:0.3rem 0;" role="tablist">
          ${PANELS.map(panel => `
            <li id="nav-item-${panel.id}" class="sidebar-nav-item" role="tab" aria-selected="false" tabindex="0"
                style="display:flex;align-items:center;gap:0.5rem;padding:0.45rem 0.75rem;cursor:pointer;font-size:0.68rem;color:var(--text-muted, #9ca3af);transition:background 0.15s,color 0.15s;border-radius:6px;margin:0.1rem 0.4rem;">
              <span class="nav-item-icon">${panel.icon}</span>
              <span class="nav-item-label">${panel.name}</span>
            </li>
          `).join('')}
        </ul>
      </nav>
    `;
  }

  /**
   * Switch to a specific panel by ID.
   * Unmounts the current panel and mounts the new one.
   *
   * @param {string} panelId - The panel ID to switch to.
   */
  function switchPanel(panelId) {
    const panelDef = PANELS.find(p => p.id === panelId);
    if (!panelDef) {
      console.warn(`[SidebarNav] Unknown panel ID: ${panelId}`);
      return;
    }

    // Unmount current panel
    unmountCurrentPanel();

    // Update active state in sidebar
    PANELS.forEach(p => {
      const navItem = document.getElementById(`nav-item-${p.id}`);
      if (navItem) {
        if (p.id === panelId) {
          navItem.classList.add('active');
          navItem.setAttribute('aria-selected', 'true');
          navItem.style.background = 'var(--bg-secondary, rgba(255,255,255,0.04))';
          navItem.style.color = 'var(--text-main, #e5e7eb)';
        } else {
          navItem.classList.remove('active');
          navItem.setAttribute('aria-selected', 'false');
          navItem.style.background = '';
          navItem.style.color = 'var(--text-muted, #9ca3af)';
        }
      }
    });

    // Mount new panel
    activePanel = panelId;
    mountPanel(panelDef);
  }

  /**
   * Mount a panel into the panel container.
   * @param {object} panelDef - Panel definition with id, PanelClass, etc.
   */
  function mountPanel(panelDef) {
    // Create a content div for the panel
    const panelContentId = `panel-content-${panelDef.id}`;
    panelContainer.innerHTML = `<div id="${panelContentId}" class="crf-panel-content" role="tabpanel"></div>`;

    // Determine the ID argument for the panel
    const panelArgs = getPanelArgs(panelDef.id);

    try {
      if (panelDef.PanelClass) {
        activePanelInstance = new panelDef.PanelClass(panelContentId, ...panelArgs);
        if (activePanelInstance.mount) {
          activePanelInstance.mount();
        }
      }
    } catch (error) {
      console.error(`[SidebarNav] Failed to mount panel ${panelDef.id}:`, error);
      panelContainer.innerHTML = `
        <div class="glass-card" style="padding:1rem;">
          <p style="color:var(--accent-red, #ef4444);font-size:0.72rem;">
            ⚠️ Failed to load ${panelDef.name} panel.
          </p>
        </div>
      `;
    }
  }

  /**
   * Get initialization arguments for a panel based on its type.
   * @param {string} panelId - The panel ID.
   * @returns {Array} Arguments to pass to the panel constructor.
   */
  function getPanelArgs(panelId) {
    switch (panelId) {
      case 'beacon-harness':
        return [context.requestId || ''];
      case 'axisweave-context':
        return [context.requestId || ''];
      case 'evidence-bundle':
        return [context.executionId || ''];
      case 'causal-graph':
        return [context.memberId || ''];
      case 'sdoh-inference':
        return [context.memberId || ''];
      case 'md-queue':
        return [];
      default:
        return [];
    }
  }

  /**
   * Unmount the currently active panel, cleaning up resources.
   */
  function unmountCurrentPanel() {
    if (activePanelInstance) {
      // Call destroy/stop if available
      if (activePanelInstance.destroy) {
        activePanelInstance.destroy();
      } else if (activePanelInstance.stop) {
        activePanelInstance.stop();
      }
      activePanelInstance = null;
    }
    panelContainer.innerHTML = '';
  }

  /**
   * Get the currently active panel ID.
   * @returns {string|null} Active panel ID, or null if none active.
   */
  function getActivePanel() {
    return activePanel;
  }

  /**
   * Destroy the sidebar navigation, unmount any active panel, and clean up.
   */
  function destroy() {
    unmountCurrentPanel();
    sidebarContainer.innerHTML = '';
    activePanel = null;
  }

  return {
    switchPanel,
    getActivePanel,
    destroy,
    getPanelIds: () => PANELS.map(p => p.id),
  };
}

export default { createSidebarNav, PANELS };
