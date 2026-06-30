/**
 * EvidenceBundleViewer Panel
 *
 * Vanilla JS panel that renders the full Evidence Bundle with lineage trail
 * linking each conclusion to its source evidence. Fetches from
 * GET /api/evidence-bundle/{execution_id}.
 *
 * Each lineage entry displays:
 *   - conclusion statement
 *   - source chunk link (evidence_id)
 *   - retrieval timestamp
 *   - confidence score (0.00-1.00)
 *
 * Requirements: 15.4, 15.8, 15.9
 */

/**
 * Renders the Evidence Bundle Viewer into a container.
 *
 * @param {string} containerId - The DOM element ID to render into.
 * @param {string} executionId - The execution ID to fetch the bundle for.
 * @param {object} [options] - Optional configuration.
 * @returns {object} Control object with refresh() method.
 */
export function render(containerId, executionId, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`[EvidenceBundleViewer] Container #${containerId} not found.`);
    return { refresh: () => {} };
  }

  let previousContent = null;

  // Build initial shell
  container.innerHTML = buildShell();
  previousContent = container.innerHTML;

  // Initial fetch
  fetchAndRender();

  async function fetchAndRender() {
    try {
      const response = await fetch(`/api/evidence-bundle/${encodeURIComponent(executionId)}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      renderBundle(data);
    } catch (error) {
      showError(error.message);
    }
  }

  function buildShell() {
    return `
      <div class="evidence-bundle-viewer glass-card">
        <h2>📦 Evidence Bundle</h2>
        <p class="subtitle">Decision lineage trail with source provenance</p>
        <div class="bundle-header" id="bundle-header-${containerId}">
          <span style="font-size:0.68rem;color:var(--text-muted);">Loading evidence bundle...</span>
        </div>
        <div class="bundle-lineage" id="bundle-lineage-${containerId}"></div>
        <div class="bundle-signatures" id="bundle-signatures-${containerId}"></div>
        <div class="bundle-error" id="bundle-error-${containerId}" style="display:none;"></div>
      </div>
    `;
  }

  function renderBundle(data) {
    const headerEl = document.getElementById(`bundle-header-${containerId}`);
    const lineageEl = document.getElementById(`bundle-lineage-${containerId}`);
    const signaturesEl = document.getElementById(`bundle-signatures-${containerId}`);
    const errorEl = document.getElementById(`bundle-error-${containerId}`);

    // Hide error
    if (errorEl) errorEl.style.display = 'none';

    // Render header with decision summary
    if (headerEl) {
      const decisionColor = getDecisionColor(data.decision);
      const decisionIcon = getDecisionIcon(data.decision);

      headerEl.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:0.4rem;padding:0.5rem;background:var(--bg-secondary);border-radius:8px;border:1px solid var(--card-border);margin-bottom:0.6rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:0.62rem;font-family:var(--font-mono);color:var(--text-muted);">
              execution: ${truncateId(data.execution_id)}
            </span>
            <span style="font-size:0.72rem;font-weight:700;color:${decisionColor};background:${decisionColor}15;padding:0.15rem 0.5rem;border-radius:20px;border:1px solid ${decisionColor}30;">
              ${decisionIcon} ${(data.decision || 'unknown').toUpperCase()}
            </span>
          </div>
          <div style="font-size:0.7rem;color:var(--text-main);line-height:1.4;">
            ${escapeHtml(data.reason || 'No reason provided')}
          </div>
        </div>
      `;
    }

    // Render lineage trail
    if (lineageEl) {
      const lineageTrail = data.lineage_trail || [];
      if (lineageTrail.length === 0) {
        lineageEl.innerHTML = `
          <div style="text-align:center;padding:1rem;color:var(--text-muted);font-size:0.72rem;">
            No lineage entries available.
          </div>
        `;
      } else {
        lineageEl.innerHTML = `
          <div style="margin-bottom:0.4rem;">
            <span style="font-size:0.7rem;font-weight:700;color:var(--text-main);">Lineage Trail</span>
            <span style="font-size:0.62rem;color:var(--text-muted);margin-left:0.4rem;">(${lineageTrail.length} entries)</span>
          </div>
          <div class="lineage-entries">
            ${lineageTrail.map((entry, index) => renderLineageEntry(entry, index)).join('')}
          </div>
        `;
      }
    }

    // Render signatures section
    if (signaturesEl) {
      const signatures = data.signatures || [];
      if (signatures.length > 0) {
        signaturesEl.innerHTML = `
          <div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid var(--card-border);">
            <span style="font-size:0.65rem;font-weight:700;color:var(--text-main);">🔏 Document Signatures</span>
            <span style="font-size:0.6rem;color:var(--text-muted);margin-left:0.3rem;">(${signatures.length})</span>
            <div style="margin-top:0.3rem;">
              ${signatures.map(sig => renderSignature(sig)).join('')}
            </div>
          </div>
        `;
      } else {
        signaturesEl.innerHTML = '';
      }
    }

    previousContent = container.innerHTML;
  }

  function renderLineageEntry(entry, index) {
    const confidence = entry.confidence;
    const confidenceDisplay = confidence != null ? confidence.toFixed(2) : 'N/A';
    const confidenceColor = confidence != null ? getConfidenceColor(confidence) : 'var(--text-muted)';

    return `
      <div class="lineage-entry" style="position:relative;padding:0.5rem 0.6rem;margin-bottom:0.4rem;background:var(--bg-secondary);border:1px solid var(--card-border);border-radius:8px;border-left:3px solid ${confidenceColor};">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.25rem;">
          <span style="font-size:0.6rem;font-weight:700;color:var(--text-muted);background:var(--card-bg);padding:0.05rem 0.25rem;border-radius:3px;border:1px solid var(--card-border);">Step ${index + 1}</span>
          <span style="font-size:0.65rem;font-weight:700;color:${confidenceColor};" title="Confidence Score">
            ${confidenceDisplay}
          </span>
        </div>
        <div style="font-size:0.7rem;color:var(--text-main);line-height:1.4;margin-bottom:0.25rem;font-weight:500;">
          ${escapeHtml(entry.conclusion || '')}
        </div>
        <div style="display:flex;gap:0.6rem;flex-wrap:wrap;font-size:0.58rem;color:var(--text-muted);font-family:var(--font-mono);">
          <span title="Source evidence chunk">
            📎 ${truncateId(entry.evidence_id)}
          </span>
          ${entry.timestamp ? `<span title="Retrieval timestamp">🕐 ${formatTimestamp(entry.timestamp)}</span>` : ''}
        </div>
      </div>
    `;
  }

  function renderSignature(sig) {
    const sigPreview = sig.signature
      ? sig.signature.substring(0, 20) + '...'
      : 'N/A';

    return `
      <div style="display:flex;align-items:center;gap:0.4rem;padding:0.2rem 0;font-size:0.58rem;font-family:var(--font-mono);color:var(--text-muted);">
        <span style="color:var(--accent-green);">🔑</span>
        <span title="${sig.key_id || ''}">${truncateId(sig.key_id)}</span>
        <span style="color:var(--text-dark);">|</span>
        <span title="${sig.signature || ''}">${sigPreview}</span>
        <span style="color:var(--text-dark);">|</span>
        <span>${sig.algorithm || 'RSA'}</span>
      </div>
    `;
  }

  function showError(message) {
    const errorEl = document.getElementById(`bundle-error-${containerId}`);
    if (errorEl) {
      errorEl.style.display = 'block';
      errorEl.innerHTML = `
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;">
          <span style="color:var(--accent-red);font-size:0.72rem;font-weight:600;">⚠️ Data source unavailable</span>
          <p style="font-size:0.65rem;color:var(--text-muted);margin-top:0.2rem;">Evidence Bundle: ${escapeHtml(message)}</p>
        </div>
      `;
      // Auto-dismiss after 10 seconds
      setTimeout(() => {
        if (errorEl) errorEl.style.display = 'none';
      }, 10000);
    }
    // Retain previous content — don't clear panel
  }

  function getDecisionColor(decision) {
    if (!decision) return 'var(--text-muted)';
    const d = decision.toLowerCase();
    if (d === 'approve' || d === 'approved') return 'var(--accent-green)';
    if (d === 'escalate' || d === 'escalated') return 'var(--accent-orange)';
    if (d === 'deny' || d === 'denied') return 'var(--accent-red)';
    return 'var(--accent-blue)';
  }

  function getDecisionIcon(decision) {
    if (!decision) return '❓';
    const d = decision.toLowerCase();
    if (d === 'approve' || d === 'approved') return '✅';
    if (d === 'escalate' || d === 'escalated') return '⚠️';
    if (d === 'deny' || d === 'denied') return '❌';
    return '📋';
  }

  function getConfidenceColor(confidence) {
    if (confidence >= 0.8) return 'var(--accent-green)';
    if (confidence >= 0.5) return 'var(--accent-blue)';
    if (confidence >= 0.3) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  }

  function truncateId(id) {
    if (!id) return 'N/A';
    return id.length > 16 ? id.substring(0, 16) + '...' : id;
  }

  function formatTimestamp(ts) {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      return d.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return ts;
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  return {
    refresh: fetchAndRender,
  };
}

/**
 * EvidenceBundleViewer class for object-oriented usage.
 */
export class EvidenceBundleViewer {
  /**
   * @param {string} containerId - DOM element ID to render into.
   * @param {string} executionId - Execution ID for the bundle.
   * @param {object} [options] - Configuration options.
   */
  constructor(containerId, executionId, options = {}) {
    this.containerId = containerId;
    this.executionId = executionId;
    this.options = options;
    this.control = null;
  }

  /**
   * Initialize and render the panel.
   * @returns {EvidenceBundleViewer} this instance for chaining.
   */
  mount() {
    this.control = render(this.containerId, this.executionId, this.options);
    return this;
  }

  /** Refresh data from the API. */
  refresh() {
    if (this.control) this.control.refresh();
  }
}

export default { render, EvidenceBundleViewer };
