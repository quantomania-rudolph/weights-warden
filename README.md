# Weights Warden

**Weights Warden** is a complete harness for **editing model weights in place** — attention heads, MLP blocks, row/column scales, checkpoint blends, and more — with **no gradient training**. An LLM agent (or your own script) proposes small edits; a **judged loop** accepts only changes that pass statistical and safety gates. Results land in a **hash-chained ledger**; held-out evaluation is operator-only.

Almost **model-agnostic**: transformers and custom PyTorch policies share the same tool surface once you wire a thin **backend** (`evaluate`, `forward`, `probe_obs`, `load_ckpt`). Out of the box it runs against the bundled toy GPT-2 and navigation demos when those checkpoints are present.

## What it does

| Area | Capabilities |
|---|---|
| **Read** | Weight summaries, diffs, baselines, prompt-pair checks, circuit tracing, grow/proposals |
| **Edit** | Head scaling, matrix ops, blends/reverts (with checkpoint refs), and architecture-mapped variants via `ModelMap` |
| **Judge** | Fixed proposal budget, paired-bootstrap accept rule, in-distribution drop cap, canary guard |
| **Train hook** | `HarnessHook` — snapshot mid-training, let an agent edit weights, your evaluator gates acceptance, auto-rollback on metric regression |
| **Agents** | MCP in Cursor/Claude Code, or `./h call TOOL '{json}'` from any shell agent |

## Quick start (Cursor / Claude Code)

1. Clone or copy this repo, open as workspace. `pip install -r requirements.txt` (torch, transformers, numpy, scipy).
2. Tell the agent: **"Read AGENTS.md and follow it."**
   - Cursor: `.cursor/mcp.json` + `AGENTS.md`. Claude Code: `.mcp.json` + `CLAUDE.md`. Enable the `weight-editing-harness` MCP server if prompted. Point `command` at the Python that has torch if `python3` is wrong.
   - Other agents: `./h call TOOL '{json}'` or `agent/PROMPT.md` + `agent/tool_spec.json`.

Operator-only: `./h finish` (held-out read for the best accepted edit). `./h status` anytime.

## Plug into an existing project (honest version)

**Not** literally drop-in with zero wiring — you implement one backend module and point `harness.env` at it:

```bash
# harness.env
HARNESS_PYTHONPATH=/path/to/dir/containing/my_backend
HARNESS_BACKEND=my_backend:MyBackend
HARNESS_MODE=llm   # or policy mode for RL-style models
```

Requirements:

- **Structure exposure** — editable tensors must match what `ops.py` expects, or you supply a **ModelMap** (`engine/any_model/spec_map.py`, `structure.py`).
- **Backend protocol** — see `engine/core/adapter.py` (Nav policy) and `engine/any_model/lm_backend.py` (transformers).
- **Adoption check** — run `engine/agent_tools/adoption_check.py` against your `model` + `backend` before trusting agent edits.
- **Training integration (optional)** — import `HarnessHook` from `engine/agent_tools/train_hook.py` in your loop; you own the evaluator and metrics.

After that, the same agent tools, ledger, and accept rules work on your checkpoint.

## Layout

| folder | what | who reads it |
|---|---|---|
| `AGENTS.md`, `CLAUDE.md`, `agent/` | process, tool list, prompts, schemas | agents |
| `runs/` | per-run ledger, notes, circuits, scratch work | agents write, operator reads |
| `engine/` | loop, `any_model` tools, MCP server, agent tools, nav sandboxes | operators |
| `models/` | bundled toy checkpoints (when shipped) | operators |
| `docs/` | specs and adoption protocol (when shipped) | operators |
| `harness.env` | backend, mode, budget, feature flags | operators |

## Tests

From repo root (bash): `PYTHON=<python with torch> scripts/run_tests.sh` when `scripts/` is present in your tree.

## Limits

Evidence in public releases is intentionally **toy-scale** until you run it on your model. Neuron-level reads are weak; component-level edits are stronger. Screen/sweep results are hints — only accepted `propose` calls count; only `finish` is final held-out.

## License

MIT — see [LICENSE](LICENSE).
