"""Edit-op registry for the core ACT-like Nav2D policy (module names fixed by docs/SPEC.md).  Same shape as lab3/lm_ops.py:
REGISTRY[name] = {"schema","doc","fn"}; fn(model, cfg, ctx) -> NEW model (deepcopy, input never mutated).
validate_cfg(name, cfg, model=None) raises ValueError on anything outside the schema (unknown/missing/extra params, bad types, out of range);
with `model` it also checks the model-specific ranges (number of layers/heads/neurons, existing matrix names).
apply(name, cfg, model, ctx=None) validates against the model, runs, and rejects NaN/inf weights.
ctx may carry {"ref_models": {ckpt_index: model}} (needed by revert_delta; ckpt_index 0..9 = ckpt_000..ckpt_009).
describe_ops(model=None, ref_ckpts=None) prints the analyst-facing listing: EVERY argument with its allowed values / range
(lesson from the lab3 experiment: never leave the analyst to guess a choice list).

Matrix names (the `matrix` parameter; 2-D weights):  tok_self tok_goal tok_obs type_emb query_emb head and, for layer i,
  enc{i}.qkv enc{i}.out enc{i}.fc1 enc{i}.fc2     (encoder self-attn qkv / out, MLP)
  dec{i}.sqkv dec{i}.sout                          (decoder self-attention)
  dec{i}.cq dec{i}.ckv dec{i}.cout                 (decoder cross-attention q / kv / out)
  dec{i}.fc1 dec{i}.fc2                            (decoder MLP)
Head kinds (scale_head): enc_self, dec_self, dec_cross.  Neuron kinds: enc, dec (fc2 input columns of that block's MLP).
"""
import copy
import torch
import adapter

MAX_LAYERS, MAX_HEADS, MAX_FF, N_CKPT = 8, 16, 4096, 10
MAX_COMPOSITE = 4          # sub-edits per composite proposal (added 2026-11-05: single-op proposals could not express the multi-block neuron edit that wins on the planted policy)
HEAD_KINDS = ["enc_self", "dec_self", "dec_cross"]
MLP_KINDS = ["enc", "dec"]
TOKENS = ["obs", "goal", "self"]
_PROBE_N = 256

REGISTRY = {}


def register(name, schema, doc):
    def deco(fn):
        if name in REGISTRY:
            raise ValueError(f"edit op {name!r} already registered")
        REGISTRY[name] = {"schema": schema, "doc": doc, "fn": fn}
        return fn
    return deco


def _is_int(v): return isinstance(v, int) and not isinstance(v, bool)


# ------------------------------------------------------------------ model structure helpers
def n_heads_of(model, attn):
    for obj in (attn, model):
        for a in ("n_heads", "num_heads", "nhead"):
            v = getattr(obj, a, None)
            if isinstance(v, int):
                return v
    raise ValueError("cannot find n_heads on the attention module or the model")


def attn_module(model, kind, layer):
    if kind == "enc_self": return model.enc_layers[layer].attn
    if kind == "dec_self": return model.dec_layers[layer].self_attn
    if kind == "dec_cross": return model.dec_layers[layer].cross_attn
    raise ValueError(f"bad head kind {kind}")


def mlp_module(model, kind, layer):
    return (model.enc_layers if kind == "enc" else model.dec_layers)[layer].mlp


def n_layers(model, kind):
    return len(model.enc_layers) if kind in ("enc", "enc_self") else len(model.dec_layers)


def matrix_table(model):
    """ordered {name: parameter tensor (2-D)} for THIS model."""
    t = {"tok_self": model.tok_self.weight, "tok_goal": model.tok_goal.weight, "tok_obs": model.tok_obs.weight,
         "type_emb": model.type_emb.weight, "query_emb": model.query_emb, "head": model.head.weight}
    for i, l in enumerate(model.enc_layers):
        t.update({f"enc{i}.qkv": l.attn.qkv.weight, f"enc{i}.out": l.attn.out.weight, f"enc{i}.fc1": l.mlp.fc1.weight, f"enc{i}.fc2": l.mlp.fc2.weight})
    for i, l in enumerate(model.dec_layers):
        t.update({f"dec{i}.sqkv": l.self_attn.qkv.weight, f"dec{i}.sout": l.self_attn.out.weight, f"dec{i}.cq": l.cross_attn.q.weight,
                  f"dec{i}.ckv": l.cross_attn.kv.weight, f"dec{i}.cout": l.cross_attn.out.weight, f"dec{i}.fc1": l.mlp.fc1.weight, f"dec{i}.fc2": l.mlp.fc2.weight})
    return t


def matrix_names(model=None):
    if model is not None:
        return list(matrix_table(model))
    names = ["tok_self", "tok_goal", "tok_obs", "type_emb", "query_emb", "head"]
    for i in range(MAX_LAYERS):
        names += [f"enc{i}.{k}" for k in ("qkv", "out", "fc1", "fc2")] + [f"dec{i}.{k}" for k in ("sqkv", "sout", "cq", "ckv", "cout", "fc1", "fc2")]
    return names


def model_dims(model):
    d = {"n_enc": len(model.enc_layers), "n_dec": len(model.dec_layers),
         "n_heads": {"enc_self": n_heads_of(model, model.enc_layers[0].attn), "dec_self": n_heads_of(model, model.dec_layers[0].self_attn),
                     "dec_cross": n_heads_of(model, model.dec_layers[0].cross_attn)},
         "d_ff": {"enc": model.enc_layers[0].mlp.fc1.out_features, "dec": model.dec_layers[0].mlp.fc1.out_features}}
    return d


# ------------------------------------------------------------------ validation
def validate_cfg(name, cfg, model=None):
    if name not in REGISTRY:
        raise ValueError(f"unknown edit op {name!r}; known: {sorted(REGISTRY)}")
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dict")
    schema = REGISTRY[name]["schema"]
    extra = set(cfg) - set(schema)
    if extra:
        raise ValueError(f"unexpected parameters for {name}: {sorted(extra)}")
    for k, s in schema.items():
        if k not in cfg:
            if s.get("required", True):
                raise ValueError(f"{name}: missing parameter {k!r}")
            continue
        v, t = cfg[k], s["type"]
        lo, hi = s.get("min", -1e18), s.get("max", 1e18)
        if t == "int_list":
            if not isinstance(v, list) or not v or not all(_is_int(i) for i in v):
                raise ValueError(f"{name}.{k} must be a non-empty list of ints")
            if len(set(v)) != len(v):
                raise ValueError(f"{name}.{k} has duplicates")
            if any(i < lo or i > hi for i in v):
                raise ValueError(f"{name}.{k} out of range [{lo},{hi}]")
        elif t in ("float", "int"):
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and (t == "float" or _is_int(v)) and v == v and abs(v) != float("inf")
            if not ok or v < lo or v > hi:
                raise ValueError(f"{name}.{k} must be a {t} in [{lo},{hi}], got {v!r}")
        elif t == "float_list":      # added 2026-11-05 (used by the opt-in value-level ops); finite numbers, length cap
            if not isinstance(v, list) or not v or len(v) > s.get("max_len", 64) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf") for x in v):
                raise ValueError(f"{name}.{k} must be a non-empty list of at most {s.get('max_len', 64)} finite numbers")
            if any(x < lo or x > hi for x in v):
                raise ValueError(f"{name}.{k} out of range [{lo},{hi}]")
        elif t == "str":
            ch = (s["choices"](model) if model is not None else s["choices"]()) if callable(s["choices"]) else s["choices"]      # model-aware so generic (non-Nav) models validate against THEIR matrix names
            if not isinstance(v, str) or v not in ch:
                raise ValueError(f"{name}.{k} must be one of {ch}")
        elif t == "op_list":      # composite proposal: a short list of {"op","cfg"} sub-edits, each fully validated; no nesting
            if not isinstance(v, list) or not 1 <= len(v) <= MAX_COMPOSITE:
                raise ValueError(f"{name}.{k} must be a list of 1..{MAX_COMPOSITE} sub-edits")
            for i, st in enumerate(v):
                if not isinstance(st, dict) or set(st) != {"op", "cfg"}:
                    raise ValueError(f"{name}.{k}[{i}] must be exactly {{'op': NAME, 'cfg': {{...}}}}")
                if st["op"] == "composite":
                    raise ValueError("composite edits cannot be nested")
                validate_cfg(st["op"], st["cfg"], model)
        else:
            raise ValueError(f"bad schema type {t}")
    if model is not None:
        _check_model_ranges(name, cfg, model)


RANGE_HOOKS = {}      # op name -> fn(cfg, model) raising ValueError; lets opt-in ops add model-dependent checks without editing this function


def _check_model_ranges(name, cfg, model):
    if name in RANGE_HOOKS: RANGE_HOOKS[name](cfg, model)
    try: dm = model_dims(model)
    except AttributeError: dm = None          # generic (non-NavPolicy) model: structure-specific ops are not registered, so dm is not needed
    if "matrix" in cfg:
        tab = matrix_table(model)
        if cfg["matrix"] not in tab:
            raise ValueError(f"{name}.matrix {cfg['matrix']!r} does not exist in this model; existing: {list(tab)}")
        mn = min(tab[cfg["matrix"]].shape)
        for key in ("k", "rank"):
            if key in cfg and cfg[key] > mn:
                raise ValueError(f"{name}.{key}={cfg[key]} exceeds min dimension {mn} of {cfg['matrix']}")
    if dm is not None and name == "scale_head":
        kind = cfg["kind"]; nl = n_layers(model, kind)
        if not 0 <= cfg["layer"] < nl:
            raise ValueError(f"scale_head.layer must be in 0..{nl - 1} for kind {kind}")
        if not 0 <= cfg["head"] < dm["n_heads"][kind]:
            raise ValueError(f"scale_head.head must be in 0..{dm['n_heads'][kind] - 1}")
    if dm is not None and name in ("scale_neurons", "prune_neurons"):
        kind = cfg["kind"]; nl = n_layers(model, kind)
        if not 0 <= cfg["layer"] < nl:
            raise ValueError(f"{name}.layer must be in 0..{nl - 1} for kind {kind}")
        if name == "scale_neurons" and max(cfg["neurons"]) >= dm["d_ff"][kind]:
            raise ValueError(f"scale_neurons.neurons must be in 0..{dm['d_ff'][kind] - 1}")


def check_finite(model, where=""):
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise ValueError(f"non-finite weights in {n} {where}")


def apply(name, cfg, model, ctx=None):
    """Validate (incl. model-specific ranges), apply on a deepcopy, verify finite weights. Never mutates `model`."""
    validate_cfg(name, cfg, model)
    out = REGISTRY[name]["fn"](model, cfg, ctx or {})
    check_finite(out, f"after {name}")
    return out


def apply_chain(steps, model, ctx=None):
    for name, cfg in steps:
        model = apply(name, cfg, model, ctx)
    return model


# ------------------------------------------------------------------ ops
_MAT = {"type": "str", "choices": matrix_names}


def _svd_edit(model, matrix, f):
    m = copy.deepcopy(model)
    with torch.no_grad():
        W = matrix_table(m)[matrix]
        U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
        W.copy_(((U * f(S)) @ Vh).to(W.dtype))
    return m


@register("project_out_top_svd", {"matrix": _MAT, "k": {"type": "int", "min": 1, "max": 256}},
          "Remove the top-k singular directions of the named matrix (W <- W - sum_{i<k} s_i u_i v_i^T).")
def _project_out_top_svd(model, cfg, ctx):
    k = cfg["k"]
    return _svd_edit(model, cfg["matrix"], lambda S: torch.cat([torch.zeros(min(k, len(S))).double(), S[k:]]))


@register("rank_truncate", {"matrix": _MAT, "rank": {"type": "int", "min": 1, "max": 256}},
          "Keep only the top-`rank` singular directions of the named matrix (low-rank approximation).")
def _rank_truncate(model, cfg, ctx):
    r = cfg["rank"]
    return _svd_edit(model, cfg["matrix"], lambda S: torch.cat([S[:r], torch.zeros(max(len(S) - r, 0)).double()]))


@register("shrink_subspace", {"matrix": _MAT, "k": {"type": "int", "min": 1, "max": 256}, "gamma": {"type": "float", "min": 0.0, "max": 4.0}},
          "Multiply the top-k singular values of the named matrix by gamma (gamma<1 shrinks that subspace, 0 = project_out_top_svd).")
def _shrink_subspace(model, cfg, ctx):
    k, g = cfg["k"], float(cfg["gamma"])
    return _svd_edit(model, cfg["matrix"], lambda S: torch.cat([S[:k] * g, S[k:]]))


@register("scale_head", {"kind": {"type": "str", "choices": HEAD_KINDS}, "layer": {"type": "int", "min": 0, "max": MAX_LAYERS - 1},
                         "head": {"type": "int", "min": 0, "max": MAX_HEADS - 1}, "gamma": {"type": "float", "min": 0.0, "max": 10.0}},
          "Multiply head `head` of layer `layer` of the given attention kind by gamma (its columns of that attention's output projection; 0 = ablate).")
def _scale_head(model, cfg, ctx):
    m = copy.deepcopy(model); at = attn_module(m, cfg["kind"], cfg["layer"]); H = n_heads_of(m, at)
    dh = at.out.in_features // H; h = cfg["head"]
    with torch.no_grad():
        at.out.weight[:, h * dh:(h + 1) * dh] *= float(cfg["gamma"])
    return m


@register("scale_neurons", {"kind": {"type": "str", "choices": MLP_KINDS}, "layer": {"type": "int", "min": 0, "max": MAX_LAYERS - 1},
                            "neurons": {"type": "int_list", "min": 0, "max": MAX_FF - 1}, "gamma": {"type": "float", "min": 0.0, "max": 10.0}},
          "Multiply the output weights (columns of mlp.fc2) of the listed MLP neurons of one encoder/decoder block by gamma.")
def _scale_neurons(model, cfg, ctx):
    m = copy.deepcopy(model); idx = torch.tensor(cfg["neurons"]).long()
    with torch.no_grad():
        mlp_module(m, cfg["kind"], cfg["layer"]).fc2.weight[:, idx] *= float(cfg["gamma"])
    return m


def neuron_act_norms(model, kind, layer):
    """Per-neuron L2 norm of the post-activation MLP hidden over a FIXED batch of TRAIN-layout observations (adapter.probe_obs('ID', 256))."""
    obs = adapter.probe_obs("ID", _PROBE_N)
    mlp = mlp_module(model, kind, layer); acts = []
    h = mlp.fc2.register_forward_pre_hook(lambda mod, inp: acts.append(inp[0].detach()))
    was = model.training; model.eval()
    try:
        with torch.no_grad():
            adapter.forward(model, obs)
    finally:
        h.remove(); model.train(was)
    return acts[0].reshape(-1, acts[0].shape[-1]).norm(dim=0)


@register("prune_neurons", {"kind": {"type": "str", "choices": MLP_KINDS}, "layer": {"type": "int", "min": 0, "max": MAX_LAYERS - 1},
                            "frac": {"type": "float", "min": 0.0, "max": 1.0}, "gamma": {"type": "float", "min": 0.0, "max": 10.0},
                            "which": {"type": "str", "choices": ["lowest", "highest"], "required": False}},
          "Rank the block's MLP neurons by activation norm on a fixed batch of TRAIN observations; scale the output weights of the `frac` lowest-norm "
          "(or highest, which='highest') neurons by gamma (gamma=0 prunes them).")
def _prune_neurons(model, cfg, ctx):
    m = copy.deepcopy(model); kind, L = cfg["kind"], cfg["layer"]
    n = neuron_act_norms(m, kind, L); n_sel = int(round(cfg["frac"] * len(n)))
    idx = torch.argsort(n, descending=(cfg.get("which", "lowest") == "highest"))[:n_sel]
    with torch.no_grad():
        mlp_module(m, kind, L).fc2.weight[:, idx] *= float(cfg["gamma"])
    return m


@register("revert_delta", {"frac": {"type": "float", "min": -1.0, "max": 1.0}, "ref_ckpt": {"type": "int", "min": 0, "max": N_CKPT - 1}},
          "Move ALL weights toward the EARLIER checkpoint ckpt_{ref_ckpt:03d} of the same training run: W <- W + frac*(W_ref - W). frac in (0,1] reverts, "
          "negative frac extrapolates away from the reference.")
def _revert_delta(model, cfg, ctx):
    refs = (ctx or {}).get("ref_models") or {}
    if cfg["ref_ckpt"] not in refs:
        raise ValueError(f"revert_delta: ctx['ref_models'] has no model for ref_ckpt={cfg['ref_ckpt']} (have {sorted(refs)})")
    ref = refs[cfg["ref_ckpt"]]; f = float(cfg["frac"])
    m = copy.deepcopy(model); rp = dict(ref.named_parameters())
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(f * (rp[n].to(p.dtype) - p))
    return m


@register("scale_token_embed", {"which": {"type": "str", "choices": TOKENS}, "gamma": {"type": "float", "min": 0.0, "max": 10.0}},
          "Multiply the token-embedding Linear (weight and bias) of one input token type (obs = the 5 obstacle tokens, goal, self) by gamma; the type embedding is untouched.")
def _scale_token_embed(model, cfg, ctx):
    m = copy.deepcopy(model); lin = {"obs": m.tok_obs, "goal": m.tok_goal, "self": m.tok_self}[cfg["which"]]
    with torch.no_grad():
        lin.weight *= float(cfg["gamma"])
        if lin.bias is not None:
            lin.bias *= float(cfg["gamma"])
    return m


ARG_TEXT_HOOK = None      # optional fn(op_name, arg, req_suffix, model) -> str|None for models without the Nav encoder/decoder structure


# ------------------------------------------------------------------ analyst-facing listing (ALWAYS shows allowed values of every argument)
def _arg_text(name, k, s, model, ref_ckpts):
    t = s["type"]; req = "" if s.get("required", True) else " (optional)"
    if t == "op_list":
        return f"    {k}{req}: list of 1..{MAX_COMPOSITE} items, each exactly {{\"op\": <one of the op names above except composite>, \"cfg\": {{...that op's arguments...}}}}"
    if t == "str":
        ch = s["choices"]
        if callable(ch):
            ch = ch(model)
        return f"    {k}{req}: one of {', '.join(ch)}"
    lo, hi = s.get("min"), s.get("max")
    dm = None
    if model is not None:
        try: dm = model_dims(model)
        except AttributeError:
            dm = None            # non-Nav model: ARG_TEXT_HOOK (set by any_model/llm_ops.py) supplies model-true ranges
            if ARG_TEXT_HOOK is not None:
                txt = ARG_TEXT_HOOK(name, k, req, model)
                if txt: return txt
    if dm is not None:
        if k == "layer":
            if name == "scale_head":
                hi = f"enc_self:{dm['n_enc'] - 1} dec_self:{dm['n_dec'] - 1} dec_cross:{dm['n_dec'] - 1}"
            else:
                hi = f"enc:{dm['n_enc'] - 1} dec:{dm['n_dec'] - 1}"
            return f"    {k}{req}: int, 0..(max by kind: {hi})"
        if k == "head":
            return f"    {k}{req}: int, 0..(max by kind: " + " ".join(f"{a}:{b - 1}" for a, b in dm["n_heads"].items()) + ")"
        if k == "neurons":
            return f"    {k}{req}: non-empty list of distinct ints, each in 0..(enc:{dm['d_ff']['enc'] - 1} dec:{dm['d_ff']['dec'] - 1})"
        if k in ("k", "rank"):
            return f"    {k}{req}: int in {lo}..{hi} and <= the smaller dimension of the matrix (see shapes below)"
    if k == "ref_ckpt":
        have = f"; checkpoints available in this run: {sorted(ref_ckpts)}" if ref_ckpts is not None else ""
        return f"    {k}{req}: int in {lo}..{hi} (ckpt_000..ckpt_009; 9 = latest earlier checkpoint){have}"
    if t == "int_list":
        return f"    {k}{req}: non-empty list of distinct ints in {lo}..{hi}"
    if t == "float_list":
        return f"    {k}{req}: non-empty list of at most {s.get('max_len', 64)} finite numbers, each in [{lo}, {hi}]"
    return f"    {k}{req}: {t} in [{lo}, {hi}]"


@register("composite", {"ops": {"type": "op_list", "min": 1, "max": MAX_COMPOSITE}},
          "Apply up to 4 of the edits above in order as ONE proposal (one evaluation charged). Each sub-edit is {\"op\": NAME, \"cfg\": {...its arguments...}} and is validated like a normal proposal; composite cannot be nested. Use it to stack edits across blocks or to extend your best earlier edit (repeat its sub-edits and add more).")
def _composite(model, cfg, ctx):
    return apply_chain([(st["op"], st["cfg"]) for st in cfg["ops"]], model, ctx)


def describe_ops(model=None, ref_ckpts=None):
    L = ["EDIT OPS (propose one per evaluation; every argument is REQUIRED unless marked optional; values outside the ranges below are REJECTED and charged):"]
    for name, e in REGISTRY.items():
        L.append(f"- {name}: {e['doc']}")
        for k, s in e["schema"].items():
            L.append(_arg_text(name, k, s, model, ref_ckpts))
    if model is not None:
        L.append("MATRIX SHAPES (name [out,in]): " + " ".join(f"{n}{list(w.shape)}" for n, w in matrix_table(model).items()))
        try:
            dm = model_dims(model)
            L.append(f"MODEL: {dm['n_enc']} encoder layers, {dm['n_dec']} decoder layers, heads per attention {dm['n_heads']}, MLP hidden {dm['d_ff']}")
        except AttributeError:
            if any(type(m).__name__ == "Conv1D" for m in model.modules()):
                L.append("NOTE: HF Conv1D matrices (GPT-2 style) are stored [in,out], i.e. TRANSPOSED relative to nn.Linear: for them axis 'col' selects an OUTPUT unit and axis 'row' an INPUT unit.")
    return "\n".join(L)
