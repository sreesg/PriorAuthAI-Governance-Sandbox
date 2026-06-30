/**
 * PanelErrorHandler — Shared non-blocking error handling utility for all panels.
 *
 * Provides consistent error display across all 6 frontend panels:
 *   - Shows error with data source name identification
 *   - Retains previous panel content (non-destructive)
 *   - Auto-dismisses error after 10 seconds
 *   - Uses glass-card CSS pattern
 *   - Does not interfere with other panels
 *
 * Requirements: 15.8, 15.9
 */

/**
 * Creates a PanelErrorHandler for a specific container.
 *
 * @param {string} containerId - The DOM element ID of the panel container.
 * @param {string} dataSourceName - Human-readable name of the data source (e.g., "BEACON Status", "Evidence Bundle").
 * @returns {object} Error handler with show(), dismiss(), and wrapFetch() methods.
 */
export function createPanelErrorHandler(containerId, dataSourceName) {
  let errorElement = null;
  let dismissTimer = null;

  /**
   * Show a non-blocking error message overlaid on the panel.
   * Retains the previous content underneath.
   *
   * @param {string} message - The error message to display.
   * @param {object} [options] - Optional configuration.
   * @param {number} [options.autoDismissMs] - Auto-dismiss timeout in ms (default: 10000).
   */
  function show(message, options = {}) {
    const autoDismissMs = options.autoDismissMs || 10000;
    const container = document.getElementById(containerId);
    if (!container) {
      console.warn(`[PanelErrorHandler] Container #${containerId} not found.`);
      return;
    }

    // Clear any existing error element and timer
    dismiss();

    // Create error overlay element
    errorElement = document.createElement('div');
    errorElement.className = 'panel-error-overlay';
    errorElement.setAttribute('role', 'alert');
    errorElement.setAttribute('aria-live', 'polite');
    errorElement.innerHTML = `
      <div class="panel-error-banner glass-card" style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;position:relative;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="color:var(--accent-red, #ef4444);font-size:0.72rem;font-weight:600;">⚠️ Data source unavailable</span>
          <button class="panel-error-dismiss" style="background:none;border:none;color:var(--text-muted, #9ca3af);cursor:pointer;font-size:0.8rem;padding:0.1rem 0.3rem;line-height:1;" aria-label="Dismiss error">&times;</button>
        </div>
        <p style="font-size:0.65rem;color:var(--text-muted, #9ca3af);margin-top:0.2rem;margin-bottom:0;">
          ${escapeHtml(dataSourceName)}: ${escapeHtml(message)}
        </p>
      </div>
    `;

    // Add dismiss button handler
    const dismissBtn = errorElement.querySelector('.panel-error-dismiss');
    if (dismissBtn) {
      dismissBtn.addEventListener('click', dismiss);
    }

    // Append error element to container (non-destructive)
    container.appendChild(errorElement);

    // Auto-dismiss after timeout
    dismissTimer = setTimeout(dismiss, autoDismissMs);
  }

  /**
   * Dismiss the current error message if one is showing.
   */
  function dismiss() {
    if (dismissTimer) {
      clearTimeout(dismissTimer);
      dismissTimer = null;
    }
    if (errorElement && errorElement.parentNode) {
      errorElement.parentNode.removeChild(errorElement);
      errorElement = null;
    }
  }

  /**
   * Wraps a fetch call with error handling.
   * On success, returns the parsed JSON response.
   * On failure, shows the error and returns null.
   *
   * @param {string} url - The URL to fetch.
   * @param {object} [fetchOptions] - Options to pass to fetch().
   * @returns {Promise<object|null>} Parsed JSON on success, null on failure.
   */
  async function wrapFetch(url, fetchOptions = {}) {
    try {
      const response = await fetch(url, fetchOptions);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      show(error.message || 'Unknown error');
      return null;
    }
  }

  /**
   * Check if an error is currently being displayed.
   * @returns {boolean} True if an error is visible.
   */
  function isShowing() {
    return errorElement !== null && errorElement.parentNode !== null;
  }

  return {
    show,
    dismiss,
    wrapFetch,
    isShowing,
  };
}

/**
 * Escape HTML entities to prevent XSS in error messages.
 * @param {string} text - Raw text to escape.
 * @returns {string} Escaped HTML-safe string.
 */
function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export default { createPanelErrorHandler };
