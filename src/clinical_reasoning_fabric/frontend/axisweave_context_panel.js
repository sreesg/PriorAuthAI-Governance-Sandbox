/**
 * AxisweaveContextPanel
 *
 * Vanilla JS panel that renders retrieved evidence chunks with provenance metadata,
 * relevance scores, and KMS signature status. Fetches from
 * GET /api/axisweave/context/{request_id}.
 *
 * Displays up to 50 evidence chunks with:
 *   - document_id
 *   - content_hash
 *   - relevance score (0.00-1.00)
 *   - KMS signature status (valid/invalid)
 *
 * Requirements: 15.3, 15.8, 15.9
 */

/**
 * Renders the Axisweave Context Panel into a container.
 *
 * @param {string} containerId - The DOM element ID to render into.
 * @param {string} requestId - The PA request ID to fetch context for.
 * @param {object} [options] - Optional configuration.
 * @param {number} [options.maxChunks] - Maximum chunks to display (default: 50).
 * @returns {object} Control object with refresh() method.
 */
export function render(containerId, requestId, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`[AxisweaveContextPanel] Container #${containerId} not found.`);
    return { refresh: () => {} };
  }

  const maxChunks = options.maxChunks || 50;
  let previousContent = null;

  // Build initial shell
  container.innerHTML = buildShell();
  previousContent = container.innerHTML;

  // Initial fetch
  fetchAndRender();

  async function fetchAndRender() {
    try {
      const response = await fetch(`/api/axisweave/context/${encodeURIComponent(requestId)}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      renderChunks(data);
    } catch (error) {
      showError(error.message);
    }
  }

  function buildShell() {
    return `
      <div class="axisweave-context-panel glass-card">
        <h2>📄 Axisweave Evidence Context</h2>
        <p class="subtitle">Retrieved evidence chunks with provenance metadata</p>
        <div class="axisweave-stats" id="axisweave-stats-${containerId}">
          <span style="font-size:0.68rem;color:var(--text-muted);">Loading evidence chunks...</span>
        </div>
        <div class="axisweave-chunks" id="axisweave-chunks-${containerId}"></div>
        <div class="axisweave-error" id="axisweave-error-${containerId}" style="display:none;"></div>
      </div>
    `;
  }

  function renderChunks(data) {
    const statsEl = document.getElementById(`axisweave-stats-${containerId}`);
    const chunksEl = document.getElementById(`axisweave-chunks-${containerId}`);
    const errorEl = document.getElementById(`axisweave-error-${containerId}`);

    // Hide error
    if (errorEl) errorEl.style.display = 'none';

    const chunks = (data.chunks || []).slice(0, maxChunks);

    // Stats bar
    if (statsEl) {
      const validCount = chunks.filter(c => c.kms_status === 'valid').length;
      const invalidCount = chunks.filter(c => c.kms_status === 'invalid').length;
      const avgScore = chunks.length > 0
        ? (chunks.reduce((sum, c) => sum + (c.relevance_score || 0), 0) / chunks.length).toFixed(2)
        : '0.00';

      statsEl.innerHTML = `
        <div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;padding:0.3rem 0;">
          <span style="font-size:0.7rem;font-weight:600;color:var(--text-main);">${chunks.length} chunk${chunks.length !== 1 ? 's' : ''}</span>
          <span style="font-size:0.65rem;color:var(--accent-green);">✓ ${validCount} valid</span>
          ${invalidCount > 0 ? `<span style="font-size:0.65rem;color:var(--accent-red);">✗ ${invalidCount} invalid</span>` : ''}
          <span style="font-size:0.65rem;color:var(--text-muted);">Avg relevance: ${avgScore}</span>
        </div>
      `;
    }

    // Render chunks list
    if (chunksEl) {
      if (chunks.length === 0) {
        chunksEl.innerHTML = `
          <div style="text-align:center;padding:1.5rem;color:var(--text-muted);font-size:0.75rem;">
            <p>No evidence chunks retrieved for this request.</p>
          </div>
        `;
      } else {
        chunksEl.innerHTML = chunks.map((chunk, index) => renderChunkCard(chunk, index)).join('');
      }
    }

    previousContent = container.innerHTML;
  }

  function renderChunkCard(chunk, index) {
    const scoreColor = getScoreColor(chunk.relevance_score);
    const kmsStatusClass = chunk.kms_status === 'valid' ? 'kms-valid' : 'kms-invalid';
    const kmsIcon = chunk.kms_status === 'valid' ? '✓' : '✗';
    const kmsColor = chunk.kms_status === 'valid' ? 'var(--accent-green)' : 'var(--accent-red)';

    // Truncate hash for display
    const hashDisplay = chunk.content_hash
      ? chunk.content_hash.substring(0, 12) + '...'
      : 'N/A';

    // Truncate text for preview
    const textPreview = chunk.text
      ? (chunk.text.length > 200 ? chunk.text.substring(0, 200) + '...' : chunk.text)
      : '';

    return `
      <div class="axisweave-chunk-card" style="background:var(--bg-secondary);border:1px solid var(--card-border);border-radius:8px;padding:0.6rem;margin-bottom:0.5rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
          <div style="display:flex;align-items:center;gap:0.4rem;">
            <span style="font-size:0.6rem;font-weight:700;color:var(--text-muted);background:var(--card-bg);padding:0.1rem 0.3rem;border-radius:4px;border:1px solid var(--card-border);">#${index + 1}</span>
            <span style="font-size:0.62rem;font-family:var(--font-mono);color:var(--text-muted);" title="${chunk.document_id || ''}">
              doc: ${truncateId(chunk.document_id)}
            </span>
          </div>
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <span style="font-size:0.65rem;font-weight:700;color:${scoreColor};" title="Relevance Score">
              ${(chunk.relevance_score || 0).toFixed(2)}
            </span>
            <span style="font-size:0.62rem;font-weight:600;color:${kmsColor};" title="KMS Signature: ${chunk.kms_status}">
              ${kmsIcon} ${chunk.kms_status}
            </span>
          </div>
        </div>
        ${textPreview ? `
          <div style="font-size:0.67rem;color:var(--text-main);line-height:1.4;margin-bottom:0.3rem;padding:0.3rem;background:var(--card-bg);border-radius:4px;">
            ${escapeHtml(textPreview)}
          </div>
        ` : ''}
        <div style="display:flex;gap:0.6rem;flex-wrap:wrap;font-size:0.58rem;color:var(--text-muted);font-family:var(--font-mono);">
          <span title="${chunk.content_hash || ''}">hash: ${hashDisplay}</span>
          <span>chunk: ${chunk.chunk_index ?? 'N/A'}</span>
          ${chunk.ingestion_timestamp ? `<span>ingested: ${formatTimestamp(chunk.ingestion_timestamp)}</span>` : ''}
        </div>
      </div>
    `;
  }

  function showError(message) {
    const errorEl = document.getElementById(`axisweave-error-${containerId}`);
    if (errorEl) {
      errorEl.style.display = 'block';
      errorEl.innerHTML = `
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;">
          <span style="color:var(--accent-red);font-size:0.72rem;font-weight:600;">⚠️ Data source unavailable</span>
          <p style="font-size:0.65rem;color:var(--text-muted);margin-top:0.2rem;">Axisweave Context: ${escapeHtml(message)}</p>
        </div>
      `;
      // Auto-dismiss after 10 seconds
      setTimeout(() => {
        if (errorEl) errorEl.style.display = 'none';
      }, 10000);
    }
    // Retain previous content
  }

  function getScoreColor(score) {
    if (score >= 0.8) return 'var(--accent-green)';
    if (score >= 0.5) return 'var(--accent-blue)';
    if (score >= 0.3) return 'var(--accent-orange)';
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
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
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
 * AxisweaveContextPanel class for object-oriented usage.
 */
export class AxisweaveContextPanel {
  /**
   * @param {string} containerId - DOM element ID to render into.
   * @param {string} requestId - PA request ID.
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
   * @returns {AxisweaveContextPanel} this instance for chaining.
   */
  mount() {
    this.control = render(this.containerId, this.requestId, this.options);
    return this;
  }

  /** Refresh data from the API. */
  refresh() {
    if (this.control) this.control.refresh();
  }
}

export default { render, AxisweaveContextPanel };
