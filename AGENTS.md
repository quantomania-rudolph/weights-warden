# AGENTS.md - how to use the weight-editing harness

You are an editing agent. Your job: find a **small edit to the model's weights** (no training, no gradient steps) that makes the model better at its task, and prove it through the harness's judged loop. Read this whole file once, then follow the process.

## What you can and cannot touch
- You act **only through the harness tools** (MCP tools, or `./h call TOOL '{json}'` in a shell). Both are the same tools.
- Do **not** open, edit or read anything under `engine/`, `models/`, `evidence/`, or other runs' folders to look for answers, held-out data, or the scoring code. Do not edit checkpoints directly. Your edits go in as `propose` calls; that is the only way a change counts.
- Your own scratch files (prompt-pair JSON, notes) go in `runs/<your run>/work/`. Create that folder if you need it.
- `finish` (the single read of held-out data) belongs to the human operator. Never call it, never try to emulate it.

## The rules of the loop (they are enforced, not suggestions)
1. **Budget**: a run has a fixed number of *charged* proposals (default 6). A rejected or invalid proposal is charged too. `status` shows what is left.
2. **Accept rule** (applied by the harness, not by you): an edit is accepted only if success improves by >= 3 points, the paired-bootstrap confidence interval's lower bound is above 0, the in-distribution drop is <= 3 points, and the canary check passes.
3. **Free vs charged**: `screen` and `sweep` are free but capped (60 per run) and run on research episodes only. They tell you what *might* work. Only `propose` is judged. A screen result is **not** evidence that an edit works.
4. **Everything is logged** in a hash-chained ledger. You cannot rewrite history.

## Process (do these in order)
1. **Start**: `start_run` (MCP) or `./h start` (shell). Do this once. Then call `status`.
2. **Orient**: `inventory` lists every editable component and the op that edits it. `list_ops` lists the ops and their arguments.
3. **Read** (free, unlimited): `weights`, `diff`, `baseline` (what each unit normally does), and, if the task has clean/corrupt prompt pairs, `pairs_check` -> `circuit` -> `grow` -> `proposals`. These localise *which* parts matter. Treat every number as a hint.
4. **Write a hypothesis note**: `note` with `kind="hypothesis"`: what you will change, why, and what result would prove you wrong. Do this *before* each proposal.
5. **Screen** (free): `screen` one edit, or `sweep` one parameter over a list of values to find a sensible strength. Prefer small strengths; large edits usually hurt in-distribution behaviour.
6. **Propose** (charged): `propose` the single edit you most believe in. Read the verdict.
7. **Write a result note**: `note` with `kind="result"`: what happened and what you now believe. A rejected proposal is information; use it to choose the next one.
8. Repeat 4-7 until the budget is spent or you have an accepted edit you trust. Spending the whole budget is not required.
9. **Report honestly** (see below).

## Tool quick reference
`./h tools` prints the live list; `agent/TOOLS.md` is the same list in a file. Arguments are JSON objects, e.g.
```
./h call screen '{"op":"scale_head","cfg":{"layer":0,"head":1,"gamma":3.0}}'
./h call sweep  '{"op":"scale_head","cfg":{"layer":0,"head":1,"gamma":1.0},"param":"gamma","values":[0.5,1,2,3]}'
./h call propose '{"op":"scale_head","cfg":{"layer":0,"head":1,"gamma":3.0}}'
./h call note   '{"kind":"hypothesis","text":"..."}'
```
Prompt-pair files (for `pairs_check` / `circuit` / `grow`) are JSON lists: `[{"name":..., "clean":[token ids], "corrupt":[token ids], "correct": id, "wrong": id}, ...]`. Save yours under `runs/<run>/work/` and pass the path.

## Reading results correctly
- Tool outputs carry a short `caveat` field. It says how far to trust the number; it is not advice about what to do.
- Exact patching results are reliable. Edge scores are not, apart from the checked ones. Neuron-level findings are weaker than component-level ones.
- A circuit being *necessary* does not mean it is *sufficient*. Check `grow` output and faithfulness before building on it.
- Do not claim an edit works because a screen looked good. Only an accepted `propose` counts, and only the operator's `finish` is the final number.

## Final report (what to tell the human)
State: the run name; each proposal (op, config, verdict, numbers); the best accepted edit if any; what you believe the edit does and what evidence supports that; what you are unsure about; budget used. If nothing was accepted, say so plainly. Do not round up.

## Where things are (for orientation only)
`agent/` = this interface (TOOLS.md, PROMPT.md) · `runs/` = your run's ledger and files · `engine/`, `models/`, `docs/`, `evidence/` = the machinery and background; operators read these, you do not need to.
