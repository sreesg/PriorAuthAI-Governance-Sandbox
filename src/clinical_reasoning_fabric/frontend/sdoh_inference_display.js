/**
 * SDOHInferenceDisplay Panel
 *
 * Vanilla JS panel that renders inferred SDOH factors with confidence scores,
 * inference chains, and distinction between inferred vs explicit facts.
 * Fetches from GET /api/inference/sdoh/{member_id}.
 *
 * Each inferred factor displays:
 *   - source text excerpt (up to 500 chars)
 *   - complete inference chain steps
 *   - confidence score (0.00-1.00)
 *   - visual origin indicator (INFERRED vs EXPLICIT)
 *
 * Requirements: 15.6, 15.8, 15.9
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
 * Renders the SDOH Inference Display into a container.
 *
 * @param {string} containerId - The DOM element ID to render into.
 * @param {string} memberId - The member ID to fetch SDOH inferences for.
 * @param {object} [options] - Optional configuration.
 * @returns {object} Control object with refresh() method.
 */
export function render(containerId, memberId, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`[SDOHInferenceDisplay] Container #${containerId} not found.`);
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
      const response = await fetch(`/api/inference/sdoh/${encodeURIComponent(memberId)}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      renderInferences(data);
    } catch (error) {
      showError(error.message);
    }
  }

  function buildShell() {
    return `
      <div class="sdoh-inference-display glass-card">
        <h2>🏠 SDOH Inference</h2>
        <p class="subtitle">Inferred &amp; explicit social determinants of health</p>
        <div class="sdoh-stats" id="sdoh-stats-${containerId}">
          <span style="font-size:0.68rem;color:var(--text-muted);">Loading SDOH factors...</span>
        </div>
        <div class="sdoh-content" id="sdoh-content-${containerId}"></div>
        <div class="sdoh-error" id="sdoh-error-${containerId}" style="display:none;"></div>
      </div>
    `;
  }

  function renderInferences(data) {
    const statsEl = document.getElementById(`sdoh-stats-${containerId}`);
    const contentEl = document.getElementById(`sdoh-content-${containerId}`);
    const errorEl = document.getElementById(`sdoh-error-${containerId}`);

    // Hide error
    if (errorEl) errorEl.style.display = 'none';

    const inferredFacts = data.inferred_facts || [];
    const explicitFacts = data.explicit_facts || [];
    const totalFacts = inferredFacts.length + explicitFacts.length;

    // Stats bar
    if (statsEl) {
      statsEl.innerHTML = `
        <div style="display:flex;gap:0.8rem;align-items:center;flex-wrap:wrap;padding:0.3rem 0;">
          <span style="font-size:0.7rem;font-weight:600;color:var(--text-main);">${totalFacts} factor${totalFacts !== 1 ? 's' : ''}</span>
          <span style="font-size:0.62rem;color:var(--accent-orange);">
            <span style="display:inline-flex;align-items:center;gap:0.15rem;background:rgba(249,115,22,0.1);padding:0.1rem 0.35rem;border-radius:10px;border:1px solid rgba(249,115,22,0.25);">🔮 ${inferredFacts.length} inferred</span>
          </span>
          <span style="font-size:0.62rem;color:var(--accent-green);">
            <span style="display:inline-flex;align-items:center;gap:0.15rem;background:rgba(34,197,94,0.1);padding:0.1rem 0.35rem;border-radius:10px;border:1px solid rgba(34,197,94,0.25);">📋 ${explicitFacts.length} explicit</span>
          </span>
        </div>
      `;
    }

    // Render content
    if (contentEl) {
      if (totalFacts === 0) {
        contentEl.innerHTML = `
          <div style="text-align:center;padding:1.5rem;color:var(--text-muted);font-size:0.75rem;">
            <p>No SDOH factors found for this member.</p>
          </div>
        `;
      } else {
        let html = '';

        // Inferred facts section
        if (inferredFacts.length > 0) {
          html += `
            <div class="sdoh-inferred-section" style="margin-bottom:0.8rem;">
              <div style="font-size:0.68rem;font-weight:700;color:var(--text-main);margin-bottom:0.4rem;display:flex;align-items:center;gap:0.3rem;">
                <span style="background:rgba(249,115,22,0.1);padding:0.1rem 0.4rem;border-radius:10px;border:1px solid rgba(249,115,22,0.25);font-size:0.6rem;color:var(--accent-orange);">🔮 INFERRED</span>
                <span>Inferred Factors</span>
              </div>
              ${inferredFacts.map(fact => renderInferredFactCard(fact)).join('')}
            </div>
          `;
        }

        // Explicit facts section
        if (explicitFacts.length > 0) {
          html += `
            <div class="sdoh-explicit-section">
              <div style="font-size:0.68rem;font-weight:700;color:var(--text-main);margin-bottom:0.4rem;display:flex;align-items:center;gap:0.3rem;">
                <span style="background:rgba(34,197,94,0.1);padding:0.1rem 0.4rem;border-radius:10px;border:1px solid rgba(34,197,94,0.25);font-size:0.6rem;color:var(--accent-green);">📋 EXPLICIT</span>
                <span>Explicit Factors</span>
              </div>
              ${explicitFacts.map(fact => renderExplicitFactCard(fact)).join('')}
            </div>
          `;
        }

        contentEl.innerHTML = html;
      }
    }

    previousContent = container.innerHTML;
  }

  function renderInferredFactCard(fact) {
    const confidence = fact.confidence;
    const confidenceDisplay = confidence != null ? confidence.toFixed(2) : 'N/A';
    const confidenceColor = getConfidenceColor(confidence);
    const category = fact.category || fact.type || 'unknown';
    const categoryIcon = getCategoryIcon(category);

    // Inference chain rendering
    const chain = fact.chain || {};
    const hops = chain.hops || [];
    const chainHtml = hops.length > 0 ? renderInferenceChain(hops) : '';

    // Source text (up to 500 chars)
    const sourceText = (fact.source_text || '').substring(0, 500);

    return `
      <div class="sdoh-inferred-card" style="background:var(--bg-secondary);border:1px solid var(--card-border);border-radius:8px;padding:0.5rem 0.6rem;margin-bottom:0.5rem;border-left:3px solid var(--accent-orange);">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.3rem;">
          <div style="display:flex;align-items:center;gap:0.3rem;">
            <span style="font-size:0.7rem;">${categoryIcon}</span>
            <span style="font-size:0.62rem;font-weight:600;color:var(--text-main);">${escapeHtml(fact.conclusion || '')}</span>
          </div>
          <div style="display:flex;align-items:center;gap:0.3rem;">
            <span style="font-size:0.6rem;font-weight:700;color:${confidenceColor};background:${confidenceColor}12;padding:0.08rem 0.3rem;border-radius:10px;border:1px solid ${confidenceColor}30;">
              ${confidenceDisplay}
            </span>
            <span style="font-size:0.5rem;color:var(--accent-orange);background:rgba(249,115,22,0.1);padding:0.08rem 0.25rem;border-radius:8px;border:1px solid rgba(249,115,22,0.2);font-weight:600;">
              INFERRED
            </span>
          </div>
        </div>
        <div style="font-size:0.58rem;color:var(--text-muted);margin-bottom:0.2rem;">
          <span style="background:var(--card-bg);padding:0.05rem 0.25rem;border-radius:3px;border:1px solid var(--card-border);">${escapeHtml(category)}</span>
          <span style="margin-left:0.3rem;">${escapeHtml(fact.type || '')}</span>
        </div>
        ${sourceText ? `
          <div style="font-size:0.63rem;color:var(--text-main);line-height:1.4;margin:0.3rem 0;padding:0.3rem;background:var(--card-bg);border-radius:4px;border:1px solid var(--card-border);">
            <span style="font-size:0.55rem;color:var(--text-muted);display:block;margin-bottom:0.15rem;">Source text:</span>
            ${escapeHtml(sourceText)}
          </div>
        ` : ''}
        ${chainHtml}
      </div>
    `;
  }

  function renderInferenceChain(hops) {
    return `
      <div class="inference-chain" style="margin-top:0.3rem;padding:0.3rem;background:rgba(59,130,246,0.03);border:1px solid rgba(59,130,246,0.12);border-radius:6px;">
        <div style="font-size:0.55rem;font-weight:600;color:var(--accent-blue);margin-bottom:0.2rem;">Inference Chain (${hops.length} hop${hops.length !== 1 ? 's' : ''}):</div>
        ${hops.map((hop, index) => `
          <div style="display:flex;align-items:flex-start;gap:0.3rem;padding:0.15rem 0;${index > 0 ? 'border-top:1px solid rgba(59,130,246,0.08);margin-top:0.1rem;padding-top:0.2rem;' : ''}">
            <span style="font-size:0.55rem;font-weight:700;color:var(--accent-blue);min-width:1.2rem;">H${hop.hop_number || index + 1}</span>
            <div style="flex:1;">
              <div style="font-size:0.58rem;color:var(--text-main);font-weight:500;">${escapeHtml(hop.intermediate_conclusion || '')}</div>
              ${hop.source_text ? `<div style="font-size:0.52rem;color:var(--text-muted);margin-top:0.08rem;font-style:italic;">${escapeHtml(hop.source_text.substring(0, 100))}</div>` : ''}
            </div>
            <span style="font-size:0.55rem;color:${getConfidenceColor(hop.confidence)};font-weight:600;">${hop.confidence != null ? hop.confidence.toFixed(2) : ''}</span>
          </div>
        `).join('')}
      </div>
    `;
  }

  function renderExplicitFactCard(fact) {
    const category = fact.category || fact.type || 'unknown';
    const categoryIcon = getCategoryIcon(category);

    return `
      <div class="sdoh-explicit-card" style="background:var(--bg-secondary);border:1px solid var(--card-border);border-radius:8px;padding:0.4rem 0.6rem;margin-bottom:0.4rem;border-left:3px solid var(--accent-green);">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div style="display:flex;align-items:center;gap:0.3rem;">
            <span style="font-size:0.7rem;">${categoryIcon}</span>
            <span style="font-size:0.62rem;font-weight:600;color:var(--text-main);">${escapeHtml(fact.conclusion || '')}</span>
          </div>
          <span style="font-size:0.5rem;color:var(--accent-green);background:rgba(34,197,94,0.1);padding:0.08rem 0.25rem;border-radius:8px;border:1px solid rgba(34,197,94,0.2);font-weight:600;">
            EXPLICIT
          </span>
        </div>
        <div style="font-size:0.58rem;color:var(--text-muted);margin-top:0.15rem;">
          <span style="background:var(--card-bg);padding:0.05rem 0.25rem;border-radius:3px;border:1px solid var(--card-border);">${escapeHtml(category)}</span>
        </div>
      </div>
    `;
  }

  function showError(message) {
    const errorEl = document.getElementById(`sdoh-error-${containerId}`);
    if (errorEl) {
      errorEl.style.display = 'block';
      errorEl.innerHTML = `
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;">
          <span style="color:var(--accent-red);font-size:0.72rem;font-weight:600;">⚠️ Data source unavailable</span>
          <p style="font-size:0.65rem;color:var(--text-muted);margin-top:0.2rem;">SDOH Inference: ${escapeHtml(message)}</p>
        </div>
      `;
      // Auto-dismiss after 10 seconds
      setTimeout(() => {
        if (errorEl) errorEl.style.display = 'none';
      }, 10000);
    }
    // Retain previous content — don't clear panel
  }

  function getConfidenceColor(confidence) {
    if (confidence == null) return 'var(--text-muted)';
    if (confidence >= 0.8) return 'var(--accent-green)';
    if (confidence >= 0.5) return 'var(--accent-blue)';
    if (confidence >= 0.3) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  }

  function getCategoryIcon(category) {
    const icons = {
      housing_instability: '🏚️',
      transportation_barriers: '🚗',
      medication_storage_limitations: '🧊',
      food_insecurity: '🍽️',
      caregiver_availability: '🤝',
      sdoh_factor: '🏠',
      medication_adherence_risk: '💊',
      care_access_barrier: '🚧',
    };
    return icons[category] || '📋';
  }

  return {
    refresh: fetchAndRender,
  };
}

/**
 * SDOHInferenceDisplay class for object-oriented usage.
 */
export class SDOHInferenceDisplay {
  /**
   * @param {string} containerId - DOM element ID to render into.
   * @param {string} memberId - Member ID to fetch SDOH data for.
   * @param {object} [options] - Configuration options.
   */
  constructor(containerId, memberId, options = {}) {
    this.containerId = containerId;
    this.memberId = memberId;
    this.options = options;
    this.control = null;
  }

  /**
   * Initialize and render the panel.
   * @returns {SDOHInferenceDisplay} this instance for chaining.
   */
  mount() {
    this.control = render(this.containerId, this.memberId, this.options);
    return this;
  }

  /** Refresh data from the API. */
  refresh() {
    if (this.control) this.control.refresh();
  }
}

export default { render, SDOHInferenceDisplay };
