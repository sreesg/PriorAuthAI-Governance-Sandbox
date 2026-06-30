/**
 * CausalGraphVisualization Panel
 *
 * Vanilla JS panel that renders the member active clinical state from the
 * Causal Ontology Graph as a visual graph with typed nodes and labeled directed
 * edges. Fetches from GET /api/graph/member/{member_id}.
 *
 * Node types: diagnosis, medication, sdoh_factor, policy_rule
 * Edge types: HAS_CONDITION, IS_PRESCRIBED, TRIGGERED_BY, GOVERNED_BY,
 *             EVIDENCED_BY, INFERRED_FROM
 *
 * Supports up to 200 nodes in the rendered view using a DOM-based grid layout.
 *
 * Requirements: 15.5, 15.8, 15.9
 */

/**
 * Node type definitions with display styling.
 */
const NODE_TYPES = {
  diagnosis: { icon: '🩺', color: 'var(--accent-blue)', label: 'Diagnosis' },
  medication: { icon: '💊', color: 'var(--accent-green)', label: 'Medication' },
  sdoh_factor: { icon: '🏠', color: 'var(--accent-orange)', label: 'SDOH Factor' },
  policy_rule: { icon: '📜', color: 'var(--accent-purple, #9333ea)', label: 'Policy Rule' },
  member: { icon: '👤', color: 'var(--text-main)', label: 'Member' },
  event: { icon: '📋', color: 'var(--accent-blue)', label: 'Event' },
  evidence_source: { icon: '📎', color: 'var(--text-muted)', label: 'Evidence' },
};

/**
 * Edge type definitions with display labels.
 */
const EDGE_TYPES = {
  HAS_CONDITION: { label: 'has condition', style: 'solid' },
  IS_PRESCRIBED: { label: 'is prescribed', style: 'solid' },
  TRIGGERED_BY: { label: 'triggered by', style: 'dashed' },
  GOVERNED_BY: { label: 'governed by', style: 'dotted' },
  EVIDENCED_BY: { label: 'evidenced by', style: 'dashed' },
  INFERRED_FROM: { label: 'inferred from', style: 'dotted' },
};

/**
 * Maximum number of nodes to render.
 */
const MAX_NODES = 200;

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
 * Renders the Causal Graph Visualization into a container.
 *
 * @param {string} containerId - The DOM element ID to render into.
 * @param {string} memberId - The member ID to fetch graph state for.
 * @param {object} [options] - Optional configuration.
 * @param {number} [options.maxNodes] - Maximum nodes to display (default: 200).
 * @returns {object} Control object with refresh() method.
 */
export function render(containerId, memberId, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`[CausalGraphViz] Container #${containerId} not found.`);
    return { refresh: () => {} };
  }

  const maxNodes = options.maxNodes || MAX_NODES;
  let previousContent = null;

  // Build initial shell
  container.innerHTML = buildShell();
  previousContent = container.innerHTML;

  // Initial fetch
  fetchAndRender();

  async function fetchAndRender() {
    try {
      const response = await fetch(`/api/graph/member/${encodeURIComponent(memberId)}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      renderGraph(data);
    } catch (error) {
      showError(error.message);
    }
  }

  function buildShell() {
    return `
      <div class="causal-graph-viz glass-card">
        <h2>🕸️ Causal Graph</h2>
        <p class="subtitle">Member active clinical state — nodes &amp; relationships</p>
        <div class="graph-legend" id="graph-legend-${containerId}">
          ${renderLegend()}
        </div>
        <div class="graph-stats" id="graph-stats-${containerId}">
          <span style="font-size:0.68rem;color:var(--text-muted);">Loading graph data...</span>
        </div>
        <div class="graph-content" id="graph-content-${containerId}"></div>
        <div class="graph-error" id="graph-error-${containerId}" style="display:none;"></div>
      </div>
    `;
  }

  function renderLegend() {
    const types = Object.entries(NODE_TYPES).filter(([key]) => key !== 'member' && key !== 'event' && key !== 'evidence_source');
    return `
      <div style="display:flex;gap:0.6rem;flex-wrap:wrap;padding:0.3rem 0;margin-bottom:0.4rem;">
        ${types.map(([key, def]) => `
          <span style="display:inline-flex;align-items:center;gap:0.2rem;font-size:0.6rem;color:var(--text-muted);background:var(--bg-secondary);padding:0.15rem 0.4rem;border-radius:12px;border:1px solid var(--card-border);">
            <span>${def.icon}</span>
            <span>${def.label}</span>
          </span>
        `).join('')}
      </div>
    `;
  }

  function renderGraph(data) {
    const statsEl = document.getElementById(`graph-stats-${containerId}`);
    const contentEl = document.getElementById(`graph-content-${containerId}`);
    const errorEl = document.getElementById(`graph-error-${containerId}`);

    // Hide error
    if (errorEl) errorEl.style.display = 'none';

    const nodes = (data.nodes || []).slice(0, maxNodes);
    const edges = data.edges || [];

    // Stats bar
    if (statsEl) {
      const nodesByType = {};
      nodes.forEach(n => {
        const type = (n.type || 'unknown').toLowerCase();
        nodesByType[type] = (nodesByType[type] || 0) + 1;
      });

      const typeStats = Object.entries(nodesByType)
        .map(([type, count]) => {
          const def = NODE_TYPES[type] || { icon: '❓', label: type };
          return `<span style="font-size:0.62rem;color:var(--text-muted);">${def.icon} ${count}</span>`;
        })
        .join('');

      statsEl.innerHTML = `
        <div style="display:flex;gap:0.8rem;align-items:center;flex-wrap:wrap;padding:0.3rem 0;">
          <span style="font-size:0.7rem;font-weight:600;color:var(--text-main);">${nodes.length} node${nodes.length !== 1 ? 's' : ''}</span>
          <span style="font-size:0.65rem;color:var(--text-muted);">${edges.length} edge${edges.length !== 1 ? 's' : ''}</span>
          ${typeStats}
          ${nodes.length >= maxNodes ? `<span style="font-size:0.6rem;color:var(--accent-orange);">⚠ Showing max ${maxNodes}</span>` : ''}
        </div>
      `;
    }

    // Render graph content
    if (contentEl) {
      if (nodes.length === 0) {
        contentEl.innerHTML = `
          <div style="text-align:center;padding:1.5rem;color:var(--text-muted);font-size:0.75rem;">
            <p>No graph data available for this member.</p>
          </div>
        `;
      } else {
        contentEl.innerHTML = `
          <div class="graph-nodes-section" style="margin-bottom:0.6rem;">
            <div style="font-size:0.68rem;font-weight:700;color:var(--text-main);margin-bottom:0.3rem;">Nodes</div>
            <div class="graph-nodes-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:0.4rem;">
              ${nodes.map(node => renderNodeCard(node)).join('')}
            </div>
          </div>
          <div class="graph-edges-section">
            <div style="font-size:0.68rem;font-weight:700;color:var(--text-main);margin-bottom:0.3rem;">Relationships</div>
            <div class="graph-edges-list" style="display:flex;flex-direction:column;gap:0.25rem;max-height:300px;overflow-y:auto;">
              ${edges.length > 0
                ? edges.map(edge => renderEdgeRow(edge, nodes)).join('')
                : '<span style="font-size:0.65rem;color:var(--text-muted);padding:0.3rem;">No relationships</span>'
              }
            </div>
          </div>
        `;
      }
    }

    previousContent = container.innerHTML;
  }

  function renderNodeCard(node) {
    const type = (node.type || 'unknown').toLowerCase();
    const def = NODE_TYPES[type] || { icon: '❓', color: 'var(--text-muted)', label: type };
    const label = node.label || node.id || 'Unknown';
    const props = node.properties || {};

    // Build a brief properties display
    const propEntries = Object.entries(props).slice(0, 3);
    const propsHtml = propEntries.length > 0
      ? propEntries.map(([k, v]) => `<span style="font-size:0.55rem;color:var(--text-muted);">${escapeHtml(k)}: ${escapeHtml(String(v).substring(0, 30))}</span>`).join('<br>')
      : '';

    return `
      <div class="graph-node-card" style="background:var(--bg-secondary);border:1px solid var(--card-border);border-radius:8px;padding:0.4rem 0.5rem;border-left:3px solid ${def.color};" title="${escapeHtml(node.id || '')}">
        <div style="display:flex;align-items:center;gap:0.3rem;margin-bottom:0.15rem;">
          <span style="font-size:0.75rem;">${def.icon}</span>
          <span style="font-size:0.65rem;font-weight:600;color:var(--text-main);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:120px;">
            ${escapeHtml(label)}
          </span>
        </div>
        <div style="font-size:0.55rem;color:var(--text-muted);background:var(--card-bg);padding:0.1rem 0.3rem;border-radius:3px;display:inline-block;margin-bottom:0.15rem;">
          ${escapeHtml(def.label)}
        </div>
        ${propsHtml ? `<div style="margin-top:0.15rem;line-height:1.3;">${propsHtml}</div>` : ''}
      </div>
    `;
  }

  function renderEdgeRow(edge, nodes) {
    const edgeType = (edge.type || '').toUpperCase();
    const edgeDef = EDGE_TYPES[edgeType] || { label: edge.type || 'related', style: 'solid' };

    // Resolve source/target labels
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);
    const sourceLabel = sourceNode ? (sourceNode.label || sourceNode.id) : (edge.source || '?');
    const targetLabel = targetNode ? (targetNode.label || targetNode.id) : (edge.target || '?');

    const borderStyle = edgeDef.style === 'dashed' ? 'border-style:dashed;' : edgeDef.style === 'dotted' ? 'border-style:dotted;' : '';

    return `
      <div class="graph-edge-row" style="display:flex;align-items:center;gap:0.3rem;padding:0.2rem 0.4rem;font-size:0.6rem;background:var(--bg-secondary);border:1px solid var(--card-border);${borderStyle}border-radius:6px;">
        <span style="color:var(--text-main);font-weight:500;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(sourceLabel)}">
          ${escapeHtml(String(sourceLabel).substring(0, 15))}
        </span>
        <span style="color:var(--text-muted);font-size:0.5rem;">→</span>
        <span style="color:var(--accent-blue);font-size:0.55rem;font-style:italic;background:rgba(59,130,246,0.08);padding:0.05rem 0.25rem;border-radius:3px;">
          ${escapeHtml(edgeDef.label)}
        </span>
        <span style="color:var(--text-muted);font-size:0.5rem;">→</span>
        <span style="color:var(--text-main);font-weight:500;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(targetLabel)}">
          ${escapeHtml(String(targetLabel).substring(0, 15))}
        </span>
        ${edge.label ? `<span style="color:var(--text-muted);font-size:0.5rem;margin-left:auto;">${escapeHtml(edge.label)}</span>` : ''}
      </div>
    `;
  }

  function showError(message) {
    const errorEl = document.getElementById(`graph-error-${containerId}`);
    if (errorEl) {
      errorEl.style.display = 'block';
      errorEl.innerHTML = `
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;">
          <span style="color:var(--accent-red);font-size:0.72rem;font-weight:600;">⚠️ Data source unavailable</span>
          <p style="font-size:0.65rem;color:var(--text-muted);margin-top:0.2rem;">Causal Graph: ${escapeHtml(message)}</p>
        </div>
      `;
      // Auto-dismiss after 10 seconds
      setTimeout(() => {
        if (errorEl) errorEl.style.display = 'none';
      }, 10000);
    }
    // Retain previous content — don't clear panel
  }

  return {
    refresh: fetchAndRender,
  };
}

/**
 * CausalGraphVisualization class for object-oriented usage.
 */
export class CausalGraphVisualization {
  /**
   * @param {string} containerId - DOM element ID to render into.
   * @param {string} memberId - Member ID to visualize.
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
   * @returns {CausalGraphVisualization} this instance for chaining.
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

export default { render, CausalGraphVisualization };
