# Weights Warden

**Weights Warden** gives an LLM **overseer** (or any script) a full **read → localize → edit → judge** loop over a PyTorch model **without training**. Everything below is what the harness actually exposes through MCP and `./h call` — see `agent/TOOLS.md`, `agent/tool_spec.json`, and `engine/agent_tools/dispatch.py`. Tool lists depend on `HARNESS_MODE` in `harness.env` (`llm` for transformers/MLPs via a ModelMap; default for the Nav2D policy demo).

---

## What the overseer’s agent can do

### Automated weight-editing research

The harness is built for **agents that run their own science**: orient with `inventory` / `list_ops`, form hypotheses (`note`), try edits for free (`screen`, `sweep`), spend a **fixed budget** of charged `propose` calls, and resume after context loss (`state`). Every tool call and verdict appends to a **hash-chained ledger** (`runs/<run>/`). The agent must **not** call `finish` — that single held-out evaluation is **operator-only** (`./h finish`).

Enforced accept rule on each `propose` (see `AGENTS.md`): success must rise by ≥3 points with paired-bootstrap CI lower bound > 0, in-distribution drop ≤3 points, and canary must pass. `screen` / `sweep` use separate research data and **do not** count as proof.

Optional **mid-training** hook (`engine/agent_tools/train_hook.py`): snapshot weights, hand a request dir to an agent, gate edits with **your** evaluator, auto-rollback if a watched metric regresses.

---

### Attribution & activation patching (`engine/any_model/patching.py`)

On clean/corrupt prompt pairs (LLM mode) or policy state batches (Nav mode), the stack separates:

| Mechanism | What it is | Role |
|-----------|------------|------|
| **Node attribution patching** | First-order: \((a_\text{corrupt} - a_\text{clean}) \cdot \partial \text{metric}/\partial a\) | Ranks heads / MLP units / hidden units |
| **Exact node patching** | Re-run forward with corrupt activations inserted | Oracle for faithfulness and circuit scores |
| **Edge attribution (EAP)** | Gradient at each **reader** input × writer contribution | Shortlists residual edges (transformers) |
| **Exact edge patching** | Add \((w_\text{corrupt} - w_\text{clean})\) at one reader only | Validates edges; used in `grow` for upstream writers |

Nav2D mode adds **exact edge-removal** over the full residual graph (`circuit` tool → `engine/nav_circuits/eap_nav.py`): memorisation vs expert **contrast** on writer→reader edges and neuron attributions (`neuron_scan`).

---

### Circuit discovery & comparison (LLM mode: `engine/any_model/llm_cli.py`, `circuit.py`, `circuit_grow.py`)

| Tool | Grounded behaviour |
|------|-------------------|
| **`pairs_check`** | Validates clean/corrupt JSON pairs (length, logit preference, corruption hurts). |
| **`circuit`** | Exact patching over **every head and MLP unit**; keep nodes with strong positive importance, **t-stat ≥ 2**, sign agreement across pairs; reports **faithfulness** (`ablate_drop`, `keep_only_recovered`). `--edges` adds edge attribution checked against exact direct-path patching. Saves named circuits under the run. |
| **`grow`** | Starts from a selected circuit; greedy expansion (upstream writers via exact edge patching, high-importance nodes, optional **baseline** co-activation neighbours) until **sufficiency** (`keep_only` recovers `target` fraction of the clean–corrupt gap); prunes; reports a random control. |
| **`baseline`** | In-distribution activation map: dead / always-on units, **co-activation clusters** (`baseline_map.py`). |
| **`neighbors`** | Units correlated with a saved circuit; optional **verification** vs random units on sufficiency. |
| **`union` / `overlap`** | **Multiple circuits**: macro core across behaviours (`min_frac`); Jaccard, shared core, behaviour-specific nodes. |
| **`proposals`** | Turns circuit nodes into **`scale_head` / `scale_neurons` configs** (optionally `composite` stacks). |

A circuit that is **necessary** is not always **sufficient** — the README process in `AGENTS.md` expects `grow` before trusting edits.

---

### Inspection & “open the hood” reads

**LLM mode:** `inventory`, `weights` (sliced raw matrices ≤4000 values; `HARNESS_RAW_VIEW=0` = summary-only ablation arm), `diff` between checkpoints `0…9` / `final`.

**Nav2D mode** (`engine/agent_tools/read_tools.py`, `analyst_tools.py`):

| Tool | What you get |
|------|----------------|
| **`weights`** | Any matrix or parameter slice (biases, LayerNorm), with norms and values. |
| **`qkv`** | Per-head **Wq / Wk / Wv / Wo** norms, **QK circuit** (\(W_q^\top W_k\)) and **OV circuit** spectra, which input dims each projection reads, **`edit_handles`** (row/column slices in `qkv` / `out` matrices), optional raw dumps. |
| **`attention`** | Mean Q×K patterns by token type, entropy, example maps (`heldout` vs `train`). |
| **`activity`** | MLP hidden units: train vs held-out firing stats; most-different units. |
| **`heads`** | Declared `action_heads` (base/nudge/expert) or logit-lens / per-writer action decomposition. |
| **`neuron_scan`** | Label-free **contrast attribution** per MLP neuron (training labels vs expert on held-out). |
| **`circuit`** | **Exact** edge-level contrast on the Nav residual graph (~90s). |
| **`diff`** | Per-module relative weight change **and** mean action change between checkpoints. |
| **`probe`** (research) | Free teacher-forced L1 shift toward expert — not closed-loop success. |

---

### Weight & matrix editing ops (`engine/core/ops.py` + `engine/agent_tools/more_ops.py`)

`list_ops` prints every op, argument ranges, and matrix shapes for **this** model. Registered set depends on mode (`engine/agent_tools/run.py`):

**Heads & MLP units**

- `scale_head` — scale one attention head’s **output** projection columns (`gamma=0` ablates).
- `head_temperature` — scale **query** rows (attention logits; where the head looks).
- `scale_neurons` / `prune_neurons` — scale or zero MLP hidden units (Nav: fc2 columns; LLM: mapped `down_proj` / MLP lin).
- `head_transplant` — blend one head’s Q/K/V/O from another (**Nav / full opt-in** only).
- `scale_token_embed` — scale token embedding linears (Nav policy).

**Whole-matrix & SVD**

- `project_out_top_svd`, `rank_truncate`, `shrink_subspace` — remove or damp dominant singular directions of a named matrix.
- `revert_delta` — move **all** weights toward an earlier checkpoint `ckpt_000…009`.
- `blend_matrix` / `blend_slice` — partial checkpoint blend on one matrix or row/column subset.

**Fine-grained value edits** (caps relative to matrix RMS; see `more_ops.py`)

- `nudge_weights` — add deltas to individual entries along one row or column.
- `scale_slice` — multiply whole rows or columns by `gamma`.
- `bias_shift` — nudge bias vector entries.
- `rank1_update` — small structured \( \alpha \, u v^\top \) patch on a sub-block.

**Composition**

- `composite` — up to **four** sub-edits in one charged `propose` (validated independently).

LLM mode uses `llm_ops.enable()` plus matrix/value ops; Nav-only structured ops (e.g. `head_transplant`) are not registered when `HARNESS_MODE=llm`.

---

### Free vs charged evaluation loop

| Tool | Cost | Data | Returns |
|------|------|------|---------|
| **`screen`** | Free (capped/run) | Research episodes | Δsuccess + CI — **hint only** |
| **`sweep`** | Free (2–9 strengths) | Research episodes | Curve + best value; each point uses a screen |
| **`propose`** | **Charged** (budget default 6) | Selection set + ID/canary gates | Verdict, Δsuccess, CI, `id_drop`, `accepted` |
| **`status`** | Free | — | Budget left, ledger chain integrity |
| **`state`** | Free | — | Digest for resume: history, notes, best edit |
| **`note`** | Free | — | `hypothesis` / `result` / `decision` on the ledger |

Outputs can include a one-line **`caveat`** (`engine/any_model/caveats.py`) stating how far to trust the measurement.

---

### How to run it

**Cursor / Claude Code:** open the repo, `pip install -r requirements.txt`, tell the agent **“Read AGENTS.md and follow it.”** Enable the `weight-editing-harness` MCP server (`.cursor/mcp.json` / `.mcp.json`).

**Any shell agent:** `./h call TOOL '{"…"}'` or `agent/PROMPT.md` + `agent/tool_spec.json`.

**Operator:** `./h finish` for held-out read; `./h status` anytime.

---

### Plug into your own model (honest)

Not zero-config: implement a **backend** (`evaluate`, `forward`, `probe_obs`, `load_ckpt`) and set `harness.env`:

```bash
HARNESS_PYTHONPATH=/path/to/your/backend
HARNESS_BACKEND=my_backend:MyBackend
HARNESS_MODE=llm          # transformers / MLPs via ModelMap + structure.py
```

Run `engine/agent_tools/adoption_check.py` on your `model` + `backend` before agent edits. For training integration, wire `HarnessHook` with your evaluator.

---

### Repo layout

| Path | Role |
|------|------|
| `AGENTS.md`, `agent/` | Agent process, tools, schemas |
| `runs/` | Ledgers, notes, saved circuits (local) |
| `engine/` | Loop, ops, patching, MCP server, backends |
| `harness.env` | Backend, mode, budget, flags |

Toy `models/`, `docs/`, `scripts/` ship in full releases when present in your tree.

---

### Limits

Public evidence is **toy-scale** until you run on your checkpoint. Exact patching beats first-order edge scores for decisions. Neuron-level reads are weaker than head/block edits. Only **accepted `propose`** counts; only **`finish`** is final held-out.

### License

MIT — see [LICENSE](LICENSE).
