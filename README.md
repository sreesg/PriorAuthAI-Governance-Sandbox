# PriorAuthAI — Payer Clinical Governance Sandbox

An interactive demonstration of a **governed AI agent** for healthcare prior authorization. Built on SkillOpt & SkillJuror specifications, this sandbox shows how Skills, Rules, Hooks, and Progressive Disclosure work together to create a transparent, auditable clinical decision support system.

## What This Demonstrates

- **Progressive Disclosure** — The agent receives rules and policies *incrementally* at each pipeline stage, not all at once. This prevents context overload and enforces separation of concerns.
- **Declarative Governance** — Clinical directors edit markdown, not code. Policies compile to OPA Rego rules automatically.
- **Dynamic Skill Registration** — New validation skills can be created from plain English descriptions at runtime.
- **Full Audit Trail** — Every decision step is traced with timestamps, type badges, and expandable JSON payloads.
- **Clinical Conservatism** — The agent never auto-denies. Ambiguous cases escalate to human Medical Directors.

## Architecture

```
PDF Policy → Markdown Declaration → OPA Rego Rules → Agent Evaluation → Decision
                                                          ↑
                                    Skills + Hooks + Rules (disclosed per-stage)
```

## Quick Start

```bash
# Start the server
python3 server.py

# Open in browser
open http://localhost:8000
```

Requires Python 3.8+ (no external dependencies).

## Project Structure

| File | Purpose |
|------|---------|
| `index.html` | Main UI — 3-column governance console |
| `help.html` | Detailed architecture & usage guide |
| `app.js` | Frontend logic, animated progressive disclosure |
| `agent.js` | Agent pipeline with staged context disclosure |
| `skills.js` | Executable skill functions (compiled output) |
| `hooks.js` | Lifecycle event callbacks enforcing rules |
| `cases.js` | Preset clinical cases & mock data |
| `regoInterpreter.js` | Lightweight OPA Rego evaluator |
| `rules.rego` | Compiled OPA policy rules |
| `rules_declaration.md` | Human-readable policy declarations |
| `skills_declaration.md` | Human-readable skill contracts |
| `server.py` | MCP-style server: compile, generate, reset |
| `real_payer_policy_uhc.pdf` | Source payer policy document |
| `index.css` | Premium dark glassmorphic design system |

## Key Concepts

### Progressive Disclosure (Agent-Level)
The agent only receives context relevant to its current pipeline stage:
- Stage 1: PHI rules only
- Stage 2: Coverage + code validation rules
- Stage 3: Clinical guidelines (CPT-specific)
- Stage 4: OPA Rego evaluation rules
- Stage 5: Notice generation + conservatism rules

### Skills
Composable functions with defined input/output contracts. Can be created from plain English via the NL skill generator.

### Rules
Governance guardrails scoped to specific lifecycle stages. Enforced through hooks.

### Hooks
Event-driven callbacks that fire at pipeline transition points, connecting rules to the agent lifecycle.

## License

MIT
