"""MODEL-AGNOSTIC tool layer.  One neutral JSON tool spec + one dispatcher; converters for the common function-calling formats.  Nothing here imports or assumes any vendor SDK.
   spec = dispatch.TOOLS                      list of {name, description, parameters(JSON Schema)}
   dispatch.to_openai(spec) / to_anthropic(spec) / to_neutral(spec)     (Gemini, Mistral, local models use the same JSON-Schema parameters)
   dispatch.call(run, name, args) -> dict      runs the tool (subprocess through agent_tools/run.py, or in-process for `state`), returns JSON-serialisable result, never raises
The analyst can NOT call: finish (the single held-out read), revert of a committed run, anything touching HOLD/CANARY.  Those belong to the overseer."""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
PY = os.environ.get("HARNESS_PYTHON", sys.executable)
S = lambda **p: {"type": "object", "properties": p, "additionalProperties": False, "required": [k for k, v in p.items() if not v.get("optional")]}
_i = lambda d, **k: {"type": "integer", "description": d, **k}; _s = lambda d, **k: {"type": "string", "description": d, **k}
KINDS = ["enc_self", "dec_self", "dec_cross"]
TOOLS = [
    {"name": "inventory", "description": "List every editable component (feedforward banks, attention-head groups, embeddings, output head) with the op that edits it.", "parameters": S()},
    {"name": "neuron_scan", "description": "Rank MLP neurons of one block by label-free memorisation contrast (train labels vs expert behaviour on held-out scenarios). Larger = more memorisation-like.", "parameters": S(block=_s("enc0|enc1|dec0|dec1"), top=_i("how many", optional=True))},
    {"name": "circuit", "description": "EXACT edge-removal contrast over the residual graph (slow, ~90 s). Read the two damage columns, not only the contrast.", "parameters": S(top=_i("how many", optional=True))},
    {"name": "heads", "description": "How actions are produced: declared action heads (base/nudge/expert) if the model exposes them, else decoder readouts and the exact per-writer split.", "parameters": S(top=_i("how many", optional=True))},
    {"name": "diff", "description": "Per-module weight change and action change between two checkpoints (ints 0..9 or 'final').", "parameters": S(a=_s("checkpoint a"), b=_s("checkpoint b"), top=_i("how many", optional=True))},
    {"name": "weights", "description": "RAW weights of any matrix/parameter slice (<=4000 values). rows/cols as 'a:b'. If it answers raw_view DISABLED the run is the summary-only arm.", "parameters": S(matrix=_s("matrix or parameter name"), rows=_s("e.g. 0:4", optional=True), cols=_s("e.g. 0:8", optional=True))},
    {"name": "qkv", "description": "One attention head: Wq/Wk/Wv/Wo norms, QK and OV circuit spectra, most-read input dims, exact edit handles; dump=true adds raw slices.", "parameters": S(kind={"type": "string", "enum": KINDS}, layer=_i("layer"), head=_i("head"), dump={"type": "boolean", "optional": True})},
    {"name": "attention", "description": "Mean query x key attention of one head, entropy, example patterns.", "parameters": S(kind={"type": "string", "enum": KINDS}, layer=_i("layer"), head=_i("head"), set=_s("heldout|train", optional=True))},
    {"name": "activity", "description": "MLP unit activations on train vs held-out states; lists the units that behave most differently.", "parameters": S(kind={"type": "string", "enum": ["enc", "dec"]}, layer=_i("layer"))},
    {"name": "list_ops", "description": "The edit ops with every argument's allowed range for THIS model. Read it before proposing.", "parameters": S()},
    {"name": "probe", "description": "FREE research probe of one edit: teacher-forced L1 change on a research set (negative = closer to the expert). Not closed-loop success. 30 per run.", "parameters": S(op=_s("op name"), cfg={"type": "object", "description": "the op's arguments"})},
    {"name": "note", "description": "Write a research note to the ledger (do this before and after every charged evaluation).", "parameters": S(kind={"type": "string", "enum": ["hypothesis", "result", "decision"]}, text=_s("the note"))},
    {"name": "screen", "description": "FREE (uncharged) trial of one edit on research episodes, separate from the data the judge uses. Returns d_success with CI. Capped per run; use it to pick what to spend a charged proposal on.", "parameters": S(op=_s("op name"), cfg={"type": "object", "description": "the op's arguments"})},
    {"name": "sweep", "description": "FREE strength sweep: one op, one numeric parameter (e.g. gamma, or a dotted path like ops.1.cfg.gamma for a composite), 2-9 values. Returns the curve on research episodes and the best value. Each value uses one screen.", "parameters": S(op=_s("op name"), cfg={"type": "object", "description": "the op's arguments (the swept parameter's value is replaced)"}, param=_s("parameter name or dotted path"), values={"type": "array", "items": {"type": "number"}, "description": "2-9 values"})},
    {"name": "propose", "description": "CHARGED evaluation of one edit against the unedited model on the selection set (rejected proposals are charged too). Returns d_success, CI, id_drop, accepted.", "parameters": S(op=_s("op name"), cfg={"type": "object", "description": "the op's arguments"})},
    {"name": "status", "description": "Budget left and ledger integrity.", "parameters": S()},
    {"name": "state", "description": "Compact digest of this run (budget, best charged edit, history, recent probes, your notes). Use it to resume after a context reset.", "parameters": S()},
]


_n = lambda d, **k: {"type": "number", "description": d, **k}; _b = lambda d, **k: {"type": "boolean", "description": d, **k}
TOOLS_LLM = [    # HARNESS_MODE=llm : transformers / MLPs through a ModelMap (any_model).  Same propose/status/state/list_ops/note as the Nav set.
    {"name": "inventory", "description": "Every editable component (blocks, heads, MLP banks, matrices with guessed roles) and the op that edits it.", "parameters": S()},
    {"name": "pairs_check", "description": "Quality-check clean/corrupt prompt pairs in a JSON file (equal length, model prefers `correct` on clean, corruption hurts). Returns which pairs are usable and why others are not.", "parameters": S(file=_s("path to pairs.json"), ckpt=_s("checkpoint to analyse: 0..9 or final (default: the target)", optional=True))},
    {"name": "circuit", "description": "Find the circuit for one behaviour: exact patching of every head and MLP unit over the usable pairs, t-stat + sign-agreement selection, faithfulness. edges=true adds edge attribution checked against exact edge patching. Saves it under `name`.", "parameters": S(file=_s("path to pairs.json"), name=_s("name to save the circuit under"), top_frac=_n("share of total importance to cover (default 0.8)", optional=True), edges=_b("also compute top edges", optional=True), ckpt=_s("checkpoint to analyse (default: the target)", optional=True))},
    {"name": "grow", "description": "(optional arg `baseline`: name of a baseline map whose co-active neighbours are offered as extra growth candidates.) Grow a circuit until keeping ONLY it clean recovers `target` of the behaviour (necessary -> sufficient). Adds upstream writers (exact edge patching), whole MLP banks and single nodes, then prunes. Saves it under `name`.", "parameters": S(file=_s("path to pairs.json"), name=_s("name to save under"), target=_n("sufficiency target (default 0.8)", optional=True), top_frac=_n("seed coverage (default 0.8)", optional=True), baseline=_s("baseline map name", optional=True))},
    {"name": "baseline", "description": "Map what lights up on NORMAL in-distribution prompts: per-layer dead / always-on units and clusters of units that co-activate. Correlational, not causal; read it as a starting map.", "parameters": S(name=_s("name to save under"), n=_i("number of prompts (default 512)", optional=True), where=_s("last (decision position) or all", optional=True), min_corr=_n("cluster threshold (default 0.6)", optional=True))},
    {"name": "neighbors", "description": "Units that co-activate with a saved circuit (candidate connective tissue). With `file` (pairs) the candidates are verified: informative only if adding them beats the same number of random units.", "parameters": S(name=_s("baseline name"), circuit=_s("saved circuit name"), file=_s("pairs.json for verification", optional=True), min_corr=_n("default 0.5", optional=True), k=_i("how many candidates (default 32)", optional=True))},
    {"name": "union", "description": "Macro circuit: nodes present in at least min_frac of several saved circuits (build the circuits from different related prompt sets).", "parameters": S(names=_s("comma-separated circuit names"), min_frac=_n("fraction of circuits (default 0.6)", optional=True))},
    {"name": "overlap", "description": "Compare saved circuits: Jaccard, nodes shared by all behaviours, nodes specific to each.", "parameters": S(names=_s("comma-separated circuit names"))},
    {"name": "proposals", "description": "Turn a saved circuit's nodes into ready-to-propose edit configs (edit ONLY circuit nodes).", "parameters": S(name=_s("circuit name"), gamma=_n("scale factor for the proposed edits (default 0)", optional=True))},
    {"name": "weights", "description": "RAW weights of any matrix/parameter slice (<=4000 values). rows/cols as 'a:b'.", "parameters": S(matrix=_s("matrix or parameter name"), rows=_s("a:b", optional=True), cols=_s("a:b", optional=True))},
    {"name": "diff", "description": "Per-matrix relative weight change between two checkpoints (ints 0..9 or 'final').", "parameters": S(a=_s("checkpoint a"), b=_s("checkpoint b"), top=_i("how many", optional=True))},
    {"name": "list_ops", "description": "The edit ops with every argument's allowed range for THIS model. Read it before proposing.", "parameters": S()},
    {"name": "note", "description": "Write a research note to the ledger (before and after every charged evaluation).", "parameters": S(kind={"type": "string", "enum": ["hypothesis", "result", "decision"]}, text=_s("the note"))},
    {"name": "screen", "description": "FREE (uncharged) trial of one edit on research episodes, separate from the data the judge uses. Returns d_success with CI. Capped per run; use it to pick what to spend a charged proposal on.", "parameters": S(op=_s("op name"), cfg={"type": "object", "description": "the op's arguments"})},
    {"name": "sweep", "description": "FREE strength sweep: one op, one numeric parameter (e.g. gamma, or a dotted path like ops.1.cfg.gamma for a composite), 2-9 values. Returns the curve on research episodes and the best value. Each value uses one screen.", "parameters": S(op=_s("op name"), cfg={"type": "object", "description": "the op's arguments (the swept parameter's value is replaced)"}, param=_s("parameter name or dotted path"), values={"type": "array", "items": {"type": "number"}, "description": "2-9 values"})},
    {"name": "propose", "description": "CHARGED evaluation of one edit against the unedited model on the selection set (rejected proposals are charged too). Returns d_success, CI, id_drop, accepted.", "parameters": S(op=_s("op name"), cfg={"type": "object", "description": "the op's arguments"})},
    {"name": "status", "description": "Budget left and ledger integrity.", "parameters": S()},
    {"name": "state", "description": "Compact digest of this run (budget, best charged edit, history, your notes). Use it to resume after a context reset.", "parameters": S()},
]


def active():
    """the tool list for the current mode (HARNESS_MODE=llm -> TOOLS_LLM)."""
    return TOOLS_LLM if os.environ.get("HARNESS_MODE") == "llm" else TOOLS


def _cmd(run, name, a):
    """argv (after agent_tools/run.py) for a tool call."""
    if name in ("pairs_check", "circuit", "grow", "union", "baseline", "neighbors", "overlap", "proposals", "note") and os.environ.get("HARNESS_MODE") == "llm":
        v = ["tools", run, name]
        for k in ("file", "name", "names", "top_frac", "gamma", "ckpt", "kind", "text", "target", "min_frac", "n", "where", "min_corr", "k", "circuit", "baseline"):
            if a.get(k) is not None: v += [f"--{k}", str(a[k])]
        return v + (["--edges"] if a.get("edges") else [])
    if name in ("inventory", "heads", "circuit", "neuron_scan", "diff", "weights", "qkv", "attention", "activity"):
        v = ["tools", run, name]
        for k in ("block", "top", "a", "b", "matrix", "rows", "cols", "kind", "layer", "head", "set"):
            if a.get(k) is not None: v += [("--set" if k == "set" else f"--{k}"), str(a[k])]
        return v + (["--dump"] if a.get("dump") else [])
    if name == "list_ops": return ["loop", "ops", run]
    if name == "status": return ["loop", "status", run]
    if name == "screen": return ["loop", "screen", run, "--op", a["op"], "--cfg", json.dumps(a["cfg"])]
    if name == "sweep": return ["loop", "sweep", run, "--op", a["op"], "--cfg", json.dumps(a["cfg"]), "--param", a["param"], "--values", ",".join(str(float(v)) for v in a["values"])]
    if name == "propose": return ["loop", "propose", run, "--op", a["op"], "--cfg", json.dumps(a["cfg"])]
    if name == "probe": return ["research", run, "probe", "--op", a["op"], "--cfg", json.dumps(a["cfg"])]
    if name == "note": return ["research", run, "note", "--kind", a["kind"], "--text", a["text"]]
    raise KeyError(name)


def call(run, name, args=None, timeout=900):
    args = args or {}
    spec = active()
    if name not in {t["name"] for t in spec}: return {"error": f"unknown tool {name!r}; available: {[t['name'] for t in spec]}"}
    missing = [k for k in next(t for t in spec if t["name"] == name)["parameters"]["required"] if k not in args]
    if missing: return {"error": f"missing arguments {missing}"}
    if name == "state":
        import research_state; return research_state.digest(run)
    try:
        p = subprocess.run([PY, os.path.join(HERE, "run.py")] + _cmd(run, name, args), capture_output=True, text=True, timeout=timeout, env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"})
    except subprocess.TimeoutExpired: return {"error": "tool timed out"}
    out = "\n".join(l for l in p.stdout.splitlines() if "Warning" not in l and "warn" not in l.lower()).strip()
    if p.returncode != 0 or out.startswith("ERROR"): return {"error": (out or p.stderr.strip().splitlines()[-1] if p.stderr.strip() else out)[:1500]}
    try: res = json.loads(out[out.index("[") if out.startswith("[") else out.index("{"):])
    except Exception: return {"text": out[:6000]}
    if name in ("screen", "sweep") and os.environ.get("HARNESS_MODE") == "llm":            # these run through loop.py, so the LM-mode caveat is attached here (same switch: HARNESS_CAVEATS=0)
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "any_model")); import caveats; res = caveats.attach(name, res)
    return res


def to_neutral(spec=TOOLS): return [{"name": t["name"], "description": t["description"], "parameters": _clean(t["parameters"])} for t in spec]
def to_openai(spec=TOOLS): return [{"type": "function", "function": t} for t in to_neutral(spec)]
def to_anthropic(spec=TOOLS): return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in to_neutral(spec)]


def _clean(schema):
    """strip the non-standard 'optional' marker (it only fed the required list)."""
    sch = json.loads(json.dumps(schema))
    for v in sch.get("properties", {}).values(): v.pop("optional", None)
    return sch


if __name__ == "__main__":
    print("tool specs are generated by `./h docs` (writes agent/TOOLS.md and agent/tool_spec.json)")
