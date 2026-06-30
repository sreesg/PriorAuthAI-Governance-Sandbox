/**
 * BEACONHarnessVisualization Panel
 *
 * Vanilla JS panel that renders the 7-layer BEACON harness execution status
 * with progressive disclosure. Fetches from GET /api/beacon/status/{request_id}.
 *
 * Layer states: pending, active, passed, failed
 * Updates within 2 seconds of actual state transition.
 *
 * Requirements: 15.1, 15.2, 15.8
 */

/**
 * BEACON layer definitions — the 7 layers in order.
 */
const BEACON_LAYERS = [
  { id: 'L1', name: 'Identity', icon: '🔐' },
  { id: 'L2', name: 'Context', icon: '📋' },
  { id: 'L3', name: 'MCP Gateway', icon: '🔧' },
  { id: 'L4', name: 'Sandbox', icon: '🏗️' },
  { id: 'L5', name: 'Verification', icon: '✓' },
  { id: 'L6', name: 'Observability', icon: '👁️' },
  { id: 'L7', name: 'Human Gates', icon: '👨‍⚕️' },
];

/**
 * Renders the BEACON Harness Visualization into a container.
 *
 * @param {string} containerId - The DOM element ID to render into.
 * @param {string} requestId - The PA request ID to fetch status for.
 * @param {object} [options] - Optional configuration.
 * @param {number} [options.pollInterval] - Polling interval in ms (default: 2000).
 * @param {boolean} [options.autoRefresh] - Whether to auto-poll (default: false).
 * @returns {object} Control object with stop() method to stop polling.
 */
export function render(containerId, requestId, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`[BEACONHarnessViz] Container #${containerId} not found.`);
    return { stop: () => {} };
  }

  const pollInterval = options.pollInterval || 2000;
  const autoRefresh = options.autoRefresh || false;
  let intervalId = null;
  let previousContent = null;

  // Build initial DOM structure
  container.innerHTML = buildShell();
  previousContent = container.innerHTML;

  // Initial fetch
  fetchAndRender();

  // Auto-refresh if enabled
  if (autoRefresh) {
    intervalId = setInterval(fetchAndRender, pollInterval);
  }

  async function fetchAndRender() {
    try {
      const response = await fetch(`/api/beacon/status/${encodeURIComponent(requestId)}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      renderStatus(data);
    } catch (error) {
      showError(error.message);
    }
  }

  function buildShell() {
    return `
      <div class="beacon-harness-viz glass-card">
        <h2>🛡️ BEACON 7-Layer Harness</h2>
        <p class="subtitle">Safety envelope execution status</p>
        <div class="beacon-pipeline-flow pipeline-flow" id="beacon-flow-${containerId}">
          ${BEACON_LAYERS.map((layer, index) => `
            ${index > 0 ? '<span class="flow-arrow"></span>' : ''}
            <div class="flow-step beacon-layer" id="beacon-layer-${layer.id}-${containerId}" data-layer-id="${layer.id}">
              <div class="flow-step-label">${layer.icon} ${layer.name}</div>
              <div class="flow-step-disclosure">${layer.id}</div>
            </div>
          `).join('')}
        </div>
        <div class="beacon-details" id="beacon-details-${containerId}">
          <div class="beacon-loading">Loading harness status...</div>
        </div>
        <div class="beacon-error" id="beacon-error-${containerId}" style="display:none;"></div>
      </div>
    `;
  }

  function renderStatus(data) {
    const { layers, current_layer } = data;
    const flowContainer = document.getElementById(`beacon-flow-${containerId}`);
    const detailsContainer = document.getElementById(`beacon-details-${containerId}`);
    const errorEl = document.getElementById(`beacon-error-${containerId}`);

    // Hide any previous error
    if (errorEl) errorEl.style.display = 'none';

    // Update each layer step
    layers.forEach((layer, index) => {
      const stepEl = document.getElementById(`beacon-layer-${layer.id}-${containerId}`);
      if (!stepEl) return;

      // Clear existing state classes
      stepEl.classList.remove('pending', 'active', 'passed', 'failed');
      stepEl.classList.add(layer.state || 'pending');

      // Update disclosure text with timestamp if available
      const discEl = stepEl.querySelector('.flow-step-disclosure');
      if (discEl) {
        if (layer.timestamp) {
          const time = new Date(layer.timestamp);
          discEl.textContent = time.toLocaleTimeString();
          discEl.classList.add('disc-active');
        } else {
          discEl.textContent = layer.id;
          discEl.classList.remove('disc-active');
        }
      }
    });

    // Build progressive disclosure details
    if (detailsContainer) {
      const activeLayer = layers.find(l => l.state === 'active');
      const failedLayer = layers.find(l => l.state === 'failed');
      const passedLayers = layers.filter(l => l.state === 'passed');

      let detailsHtml = '<div class="disclosure-panel">';
      detailsHtml += '<div class="disclosure-header">';
      detailsHtml += `<span style="font-size:0.72rem;font-weight:700;color:var(--text-main);">Status:</span>`;

      if (failedLayer) {
        detailsHtml += `<span style="color:var(--accent-red);font-size:0.72rem;font-weight:600;">❌ Failed at ${failedLayer.name}</span>`;
      } else if (activeLayer) {
        detailsHtml += `<span style="color:var(--accent-blue);font-size:0.72rem;font-weight:600;">⏳ Processing ${activeLayer.name}...</span>`;
      } else if (passedLayers.length === 7) {
        detailsHtml += `<span style="color:var(--accent-green);font-size:0.72rem;font-weight:600;">✅ All layers passed</span>`;
      } else if (passedLayers.length > 0) {
        detailsHtml += `<span style="color:var(--accent-green);font-size:0.72rem;font-weight:600;">${passedLayers.length}/7 layers complete</span>`;
      } else {
        detailsHtml += `<span style="color:var(--text-muted);font-size:0.72rem;">Awaiting execution</span>`;
      }

      detailsHtml += '</div>';

      // Layer summary list
      detailsHtml += '<div class="beacon-layer-summary" style="margin-top:0.4rem;">';
      layers.forEach(layer => {
        const stateIcon = getStateIcon(layer.state);
        const stateColor = getStateColor(layer.state);
        detailsHtml += `<div style="display:flex;align-items:center;gap:0.4rem;padding:0.15rem 0;font-size:0.68rem;">`;
        detailsHtml += `<span style="color:${stateColor};">${stateIcon}</span>`;
        detailsHtml += `<span style="font-weight:600;color:var(--text-main);">${layer.name}</span>`;
        detailsHtml += `<span style="color:var(--text-muted);margin-left:auto;">${layer.state}</span>`;
        if (layer.timestamp) {
          detailsHtml += `<span style="color:var(--text-muted);font-family:var(--font-mono);font-size:0.6rem;">${formatTime(layer.timestamp)}</span>`;
        }
        detailsHtml += '</div>';
      });
      detailsHtml += '</div></div>';

      detailsContainer.innerHTML = detailsHtml;
    }

    // Store as previous content
    previousContent = container.innerHTML;
  }

  function showError(message) {
    const errorEl = document.getElementById(`beacon-error-${containerId}`);
    if (errorEl) {
      errorEl.style.display = 'block';
      errorEl.innerHTML = `
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;">
          <span style="color:var(--accent-red);font-size:0.72rem;font-weight:600;">⚠️ Data source unavailable</span>
          <p style="font-size:0.65rem;color:var(--text-muted);margin-top:0.2rem;">BEACON Status: ${message}</p>
        </div>
      `;
      // Auto-dismiss after 10 seconds
      setTimeout(() => {
        if (errorEl) errorEl.style.display = 'none';
      }, 10000);
    }
    // Retain previous content — don't clear panel
  }

  function getStateIcon(state) {
    switch (state) {
      case 'passed': return '✅';
      case 'active': return '⏳';
      case 'failed': return '❌';
      default: return '⬜';
    }
  }

  function getStateColor(state) {
    switch (state) {
      case 'passed': return 'var(--accent-green)';
      case 'active': return 'var(--accent-blue)';
      case 'failed': return 'var(--accent-red)';
      default: return 'var(--text-muted)';
    }
  }

  function formatTime(isoTimestamp) {
    if (!isoTimestamp) return '';
    try {
      const d = new Date(isoTimestamp);
      return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return isoTimestamp;
    }
  }

  // Return control object
  return {
    stop: () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    },
    refresh: fetchAndRender,
  };
}

/**
 * BEACONHarnessVisualization class for object-oriented usage.
 */
export class BEACONHarnessVisualization {
  /**
   * @param {string} containerId - DOM element ID to render into.
   * @param {string} requestId - PA request ID to visualize.
   * @param {object} [options] - Configuration options.
   */
  constructor(containerId, requestId, options = {}) {
    this.containerId = containerId;
    this.requestId = requestId;
    this.options = options;
    this.control = null;
  }

  /**
   * Initialize and render the panel.
   * @returns {BEACONHarnessVisualization} this instance for chaining.
   */
  mount() {
    this.control = render(this.containerId, this.requestId, this.options);
    return this;
  }

  /** Refresh data from the API. */
  refresh() {
    if (this.control) this.control.refresh();
  }

  /** Stop auto-polling if enabled. */
  destroy() {
    if (this.control) this.control.stop();
  }
}

export default { render, BEACONHarnessVisualization };
