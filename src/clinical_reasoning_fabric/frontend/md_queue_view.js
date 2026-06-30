/**
 * MedicalDirectorQueueView Panel
 *
 * Vanilla JS panel that renders escalated cases in the Medical Director queue
 * with per-criterion status, challenger findings, and briefing summary.
 * Fetches from GET /api/md-queue.
 *
 * Each escalated case displays all 4 required artifacts:
 *   - Briefing Packet summary
 *   - Criteria assessment (per-criterion: met/not_met/indeterminate/not_evaluated)
 *   - OPA Challenger findings
 *   - Execution trace summary
 *
 * Requirements: 15.7, 15.8, 15.9
 */

/**
 * Escapes HTML to prevent XSS.
 * @param {string} text - Raw text to escape.
 * @returns {string} Escaped HTML-safe string.
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Renders the Medical Director Queue View into a container.
 *
 * @param {string} containerId - The DOM element ID to render into.
 * @param {string} [identifier] - Optional identifier (unused, for API consistency).
 * @param {object} [options] - Optional configuration.
 * @returns {object} Control object with refresh() method.
 */
export function render(containerId, identifier, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`[MDQueueView] Container #${containerId} not found.`);
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
      const response = await fetch('/api/md-queue');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      renderQueue(data);
    } catch (error) {
      showError(error.message);
    }
  }

  function buildShell() {
    return `
      <div class="md-queue-view glass-card">
        <h2>👨‍⚕️ Medical Director Queue</h2>
        <p class="subtitle">Escalated cases awaiting physician review</p>
        <div class="md-queue-stats" id="md-queue-stats-${containerId}">
          <span style="font-size:0.68rem;color:var(--text-muted);">Loading escalated cases...</span>
        </div>
        <div class="md-queue-cases" id="md-queue-cases-${containerId}"></div>
        <div class="md-queue-error" id="md-queue-error-${containerId}" style="display:none;"></div>
      </div>
    `;
  }

  function renderQueue(data) {
    const statsEl = document.getElementById(`md-queue-stats-${containerId}`);
    const casesEl = document.getElementById(`md-queue-cases-${containerId}`);
    const errorEl = document.getElementById(`md-queue-error-${containerId}`);

    // Hide error
    if (errorEl) errorEl.style.display = 'none';

    const cases = data.cases || [];

    // Stats bar
    if (statsEl) {
      statsEl.innerHTML = `
        <div style="display:flex;gap:0.8rem;align-items:center;flex-wrap:wrap;padding:0.3rem 0;">
          <span style="font-size:0.7rem;font-weight:600;color:var(--text-main);">${cases.length} case${cases.length !== 1 ? 's' : ''} pending review</span>
          ${cases.length > 0 ? `<span style="font-size:0.62rem;color:var(--accent-orange);">⏳ Awaiting MD decision</span>` : ''}
        </div>
      `;
    }

    // Render cases list
    if (casesEl) {
      if (cases.length === 0) {
        casesEl.innerHTML = `
          <div style="text-align:center;padding:1.5rem;color:var(--text-muted);font-size:0.75rem;">
            <p>✅ No escalated cases in queue.</p>
          </div>
        `;
      } else {
        casesEl.innerHTML = cases.map((c, index) => renderCaseCard(c, index)).join('');
      }
    }

    previousContent = container.innerHTML;
  }

  function renderCaseCard(caseData, index) {
    const criteria = caseData.criteria_assessment || [];
    const escalatedAt = caseData.escalated_at ? formatTimestamp(caseData.escalated_at) : 'Unknown';

    return `
      <div class="md-queue-case-card" style="background:var(--bg-secondary);border:1px solid var(--card-border);border-radius:8px;padding:0.6rem;margin-bottom:0.6rem;border-left:3px solid var(--accent-orange);">
        <!-- Case Header -->
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
          <div style="display:flex;align-items:center;gap:0.3rem;">
            <span style="font-size:0.6rem;font-weight:700;color:var(--text-muted);background:var(--card-bg);padding:0.1rem 0.3rem;border-radius:4px;border:1px solid var(--card-border);">#${index + 1}</span>
            <span style="font-size:0.65rem;font-family:var(--font-mono);color:var(--text-muted);" title="${escapeHtml(caseData.case_id || '')}">
              ${escapeHtml(truncateId(caseData.case_id))}
            </span>
          </div>
          <span style="font-size:0.58rem;color:var(--text-muted);">🕐 ${escapeHtml(escalatedAt)}</span>
        </div>

        <!-- 1. Briefing Packet Summary -->
        <div class="artifact-section" style="margin-bottom:0.4rem;">
          <div style="font-size:0.6rem;font-weight:700;color:var(--accent-blue);margin-bottom:0.15rem;">📋 Briefing Summary</div>
          <div style="font-size:0.63rem;color:var(--text-main);line-height:1.4;padding:0.25rem 0.4rem;background:var(--card-bg);border-radius:4px;border:1px solid var(--card-border);">
            ${escapeHtml(caseData.briefing_summary || 'No briefing summary available')}
          </div>
        </div>

        <!-- 2. Criteria Assessment -->
        <div class="artifact-section" style="margin-bottom:0.4rem;">
          <div style="font-size:0.6rem;font-weight:700;color:var(--accent-blue);margin-bottom:0.15rem;">📊 Criteria Assessment</div>
          <div style="padding:0.2rem 0.4rem;background:var(--card-bg);border-radius:4px;border:1px solid var(--card-border);">
            ${criteria.length > 0
              ? criteria.map(c => renderCriterionStatus(c)).join('')
              : '<span style="font-size:0.6rem;color:var(--text-muted);">No criteria data</span>'
            }
          </div>
        </div>

        <!-- 3. OPA Challenger Findings -->
        <div class="artifact-section" style="margin-bottom:0.4rem;">
          <div style="font-size:0.6rem;font-weight:700;color:var(--accent-blue);margin-bottom:0.15rem;">⚖️ Challenger Findings</div>
          <div style="font-size:0.63rem;color:var(--text-main);line-height:1.4;padding:0.25rem 0.4rem;background:var(--card-bg);border-radius:4px;border:1px solid var(--card-border);">
            ${escapeHtml(caseData.challenger_findings || 'No findings')}
          </div>
        </div>

        <!-- 4. Execution Trace Summary -->
        <div class="artifact-section">
          <div style="font-size:0.6rem;font-weight:700;color:var(--accent-blue);margin-bottom:0.15rem;">🔍 Trace Summary</div>
          <div style="font-size:0.63rem;color:var(--text-main);line-height:1.4;padding:0.25rem 0.4rem;background:var(--card-bg);border-radius:4px;border:1px solid var(--card-border);">
            ${escapeHtml(caseData.trace_summary || 'No trace summary available')}
          </div>
        </div>
      </div>
    `;
  }

  function renderCriterionStatus(criterion) {
    const status = (criterion.status || 'not_evaluated').toLowerCase();
    const statusDisplay = getStatusDisplay(status);

    return `
      <div style="display:flex;align-items:center;justify-content:space-between;padding:0.12rem 0;font-size:0.6rem;${status === 'not_met' || status === 'indeterminate' ? 'font-weight:500;' : ''}">
        <span style="color:var(--text-main);">${escapeHtml(criterion.criterion || '')}</span>
        <span style="color:${statusDisplay.color};font-weight:600;font-size:0.55rem;background:${statusDisplay.color}10;padding:0.05rem 0.3rem;border-radius:8px;border:1px solid ${statusDisplay.color}25;">
          ${statusDisplay.icon} ${statusDisplay.label}
        </span>
      </div>
    `;
  }

  function getStatusDisplay(status) {
    switch (status) {
      case 'met':
        return { icon: '✅', label: 'MET', color: 'var(--accent-green)' };
      case 'not_met':
        return { icon: '❌', label: 'NOT MET', color: 'var(--accent-red)' };
      case 'indeterminate':
        return { icon: '❓', label: 'INDETERMINATE', color: 'var(--accent-orange)' };
      case 'not_evaluated':
      default:
        return { icon: '⬜', label: 'NOT EVALUATED', color: 'var(--text-muted)' };
    }
  }

  function showError(message) {
    const errorEl = document.getElementById(`md-queue-error-${containerId}`);
    if (errorEl) {
      errorEl.style.display = 'block';
      errorEl.innerHTML = `
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;">
          <span style="color:var(--accent-red);font-size:0.72rem;font-weight:600;">⚠️ Data source unavailable</span>
          <p style="font-size:0.65rem;color:var(--text-muted);margin-top:0.2rem;">MD Queue: ${escapeHtml(message)}</p>
        </div>
      `;
      // Auto-dismiss after 10 seconds
      setTimeout(() => {
        if (errorEl) errorEl.style.display = 'none';
      }, 10000);
    }
    // Retain previous content — don't clear panel
  }

  function truncateId(id) {
    if (!id) return 'N/A';
    return id.length > 20 ? id.substring(0, 20) + '...' : id;
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
      });
    } catch {
      return ts;
    }
  }

  return {
    refresh: fetchAndRender,
  };
}

/**
 * MedicalDirectorQueueView class for object-oriented usage.
 */
export class MedicalDirectorQueueView {
  /**
   * @param {string} containerId - DOM element ID to render into.
   * @param {string} [identifier] - Optional identifier (for API consistency).
   * @param {object} [options] - Configuration options.
   */
  constructor(containerId, identifier, options = {}) {
    this.containerId = containerId;
    this.identifier = identifier;
    this.options = options;
    this.control = null;
  }

  /**
   * Initialize and render the panel.
   * @returns {MedicalDirectorQueueView} this instance for chaining.
   */
  mount() {
    this.control = render(this.containerId, this.identifier, this.options);
    return this;
  }

  /** Refresh data from the API. */
  refresh() {
    if (this.control) this.control.refresh();
  }
}

export default { render, MedicalDirectorQueueView };
