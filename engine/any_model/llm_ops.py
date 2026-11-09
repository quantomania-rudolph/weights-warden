"""Head / neuron ops for ANY architecture ModelMap supports (GPT-2, Llama-style incl. GQA, plain MLP) - same op names the Nav toy uses, so prompts and loops are unchanged.
enable() registers (replacing the Nav-specific versions, see generic_model.enable_generic): scale_head, scale_neurons, prune_neurons, head_temperature.  All edit a deep copy; all validated against THIS model's layer/head/unit counts; gamma=1 is the exact identity.
For an MLP (no heads): scale_neurons/prune_neurons act on hidden layer `layer`; head ops are not registered."""
import os, sys, copy, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import ops as OPS, structure as S, generic_model as GM

_SAVED = {}
_SPEC = {"spec": None}          # optional declarative ModelMap (dict/path) bound by enable(spec=...)


def _tm(model): return S.discover(model, _SPEC["spec"])


def _check_layer(tm, layer, what):
    n = tm.n_layers if tm.kind == "transformer" else tm.n_hidden
    if not 0 <= layer < n: raise ValueError(f"{what}.layer must be in 0..{n - 1} for this model")


def _chk_head(cfg, model):
    tm = _tm(model)
    if tm.kind != "transformer": raise ValueError("this model has no attention heads")
    _check_layer(tm, cfg["layer"], "head op")
    if cfg.get("_needs_q") and tm.q_layout != "contiguous": raise ValueError("head_temperature unavailable: fused qkv layout not declared contiguous")
    if not 0 <= cfg["head"] < tm.n_heads: raise ValueError(f"head must be in 0..{tm.n_heads - 1}")


def _chk_temp(cfg, model):
    _chk_head(cfg, model)
    if _tm(model).q_layout != "contiguous": raise ValueError("head_temperature unavailable: fused qkv layout not declared contiguous")


def _chk_neurons(cfg, model):
    tm = _tm(model); _check_layer(tm, cfg["layer"], "neuron op"); n = tm.d_ff if tm.kind == "transformer" else tm.widths[cfg["layer"]]
    if max(cfg["neurons"]) >= n: raise ValueError(f"neurons must be in 0..{n - 1} for layer {cfg['layer']}")


def _scale_head(model, cfg, ctx):
    m = copy.deepcopy(model); tm = _tm(m)
    with torch.no_grad(): tm.o_head(tm.blocks[cfg["layer"]], cfg["head"]).mul_(float(cfg["gamma"]))
    return m


def _scale_neurons(model, cfg, ctx):
    m = copy.deepcopy(model); tm = _tm(m); u = list(cfg["neurons"])
    with torch.no_grad():
        if tm.kind == "mlp": W = tm.lin[cfg["layer"] + 1].weight; W[:, u] = W[:, u] * float(cfg["gamma"])
        else: W, ix = tm.down_units(tm.blocks[cfg["layer"]], u); W[ix] = W[ix] * float(cfg["gamma"])
    return m


def _prune_neurons(model, cfg, ctx):
    """remove units completely: zero their outgoing weights AND their incoming weights/bias."""
    m = copy.deepcopy(model); tm = _tm(m); u = list(cfg["neurons"])
    with torch.no_grad():
        if tm.kind == "mlp": tm.lin[cfg["layer"] + 1].weight[:, u] = 0; tm.lin[cfg["layer"]].weight[u, :] = 0; tm.lin[cfg["layer"]].bias[u] = 0
        else:
            b = tm.blocks[cfg["layer"]]; W, ix = tm.down_units(b, u); W[ix] = 0
            for w, ix, bias in tm.up_units(b, u):
                w[ix] = 0
                if bias is not None: bias[u] = 0
    return m


def _head_temperature(model, cfg, ctx):
    m = copy.deepcopy(model); tm = _tm(m); w, b = tm.q_head(tm.blocks[cfg["layer"]], cfg["head"]); g = float(cfg["gamma"])
    with torch.no_grad():
        w.mul_(g)
        if b is not None: b.mul_(g)
    return m


def _arg_text(name, k, req, model):
    """model-true ranges for the analyst listing (the Nav version talks about encoder/decoder kinds, which do not exist here)."""
    tm = _tm(model)
    if k == "layer":
        n = tm.n_layers if tm.kind == "transformer" else tm.n_hidden
        return f"    {k}{req}: int, 0..{n - 1}" + ("" if tm.kind == "transformer" else " (hidden layer index)")
    if k == "head" and tm.kind == "transformer": return f"    {k}{req}: int, 0..{tm.n_heads - 1}"
    if k == "neurons":
        n = tm.d_ff if tm.kind == "transformer" else max(tm.widths)
        return f"    {k}{req}: non-empty list of distinct ints, each in 0..{n - 1}" + ("" if tm.kind == "transformer" else " (must fit the chosen layer's width)")
    return None


def enable(spec=None):
    _SPEC["spec"] = spec; OPS.ARG_TEXT_HOOK = _arg_text
    if _SAVED: return
    GM.enable_generic()
    L = {"type": "int", "min": 0, "max": 255}; H = {"type": "int", "min": 0, "max": 255}; N = {"type": "int_list", "min": 0, "max": 262143}
    spec = [("scale_head", {"layer": L, "head": H, "gamma": {"type": "float", "min": 0.0, "max": 10.0}}, _scale_head, _chk_head, "Multiply attention head `head` of block `layer` by gamma (scales the weights that read that head's output; gamma 0 removes the head, 1 is the identity)."),
            ("scale_neurons", {"layer": L, "neurons": N, "gamma": {"type": "float", "min": 0.0, "max": 10.0}}, _scale_neurons, _chk_neurons, "Multiply the OUTPUT weights of the listed MLP hidden units of block/hidden-layer `layer` by gamma."),
            ("prune_neurons", {"layer": L, "neurons": N}, _prune_neurons, _chk_neurons, "Remove the listed MLP hidden units entirely (zero their incoming weights/bias and outgoing weights)."),
            ("head_temperature", {"layer": L, "head": H, "gamma": {"type": "float", "min": 0.25, "max": 4.0}}, _head_temperature, _chk_temp, "Scale one head's QUERY projection by gamma: multiplies its attention logits (sharper/softer) without changing what it writes.")]
    for name, sch, fn, chk, doc in spec: OPS.register(name, sch, doc)(fn); OPS.RANGE_HOOKS[name] = chk; _SAVED[name] = True


def disable():
    for k in list(_SAVED): OPS.REGISTRY.pop(k, None); OPS.RANGE_HOOKS.pop(k, None)
    _SAVED.clear(); _SPEC["spec"] = None; OPS.ARG_TEXT_HOOK = None; GM.restore()
