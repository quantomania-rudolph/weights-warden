"""OPT-IN extra op (same bridge pattern as nav_sandbox/ridge_op: off unless enable() is called, so old runs/prompts are byte-identical).
head_temperature: scale the QUERY projection (weight rows AND bias) of ONE attention head by gamma => that head's attention logits scale by gamma (gamma<1 flatter/softer, gamma>1 sharper).
It changes WHERE a head looks without changing what it writes, which scale_head (out-projection columns) cannot do."""
import os, sys, copy, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import ops as OPS


def _head_temperature(model, cfg, ctx):
    m = copy.deepcopy(model); at = OPS.attn_module(m, cfg["kind"], cfg["layer"]); H = OPS.n_heads_of(m, at); h = int(cfg["head"]); g = float(cfg["gamma"])
    lin = at.qkv if cfg["kind"] in ("enc_self", "dec_self") else at.q; d = lin.weight.shape[1]; dh = d // H; s = slice(h * dh, (h + 1) * dh)       # q rows come first in qkv
    with torch.no_grad(): lin.weight[s] *= g; lin.bias[s] *= g
    return m


def enable_head_temperature():
    if "head_temperature" in OPS.REGISTRY: return
    OPS.register("head_temperature", {"kind": {"type": "str", "choices": OPS.HEAD_KINDS}, "layer": {"type": "int", "min": 0, "max": OPS.MAX_LAYERS - 1}, "head": {"type": "int", "min": 0, "max": OPS.MAX_HEADS - 1},
                 "gamma": {"type": "float", "min": 0.25, "max": 4.0}}, "Scale one attention head's query projection by gamma: multiplies that head's attention logits (temperature), changing where it looks but not what it writes.")(_head_temperature)


# ---------------------------------------------------------------- value-level edits (not just scale/zero)
MAX_ENTRIES = 64; REL_CAP = 2.0           # each |delta| <= REL_CAP x the matrix RMS weight: small, local nudges only


def _slice_check(name):
    def chk(cfg, model):
        tab = OPS.matrix_table(model)
        if cfg["matrix"] not in tab: raise ValueError(f"{name}.matrix {cfg['matrix']!r} not in this model; existing: {list(tab)}")
        W = tab[cfg["matrix"]]; ax = 0 if cfg["axis"] == "row" else 1; other = W.shape[1 - ax]
        if not 0 <= cfg["index"] < W.shape[ax]: raise ValueError(f"{name}.index must be in 0..{W.shape[ax] - 1} for axis {cfg['axis']} of {cfg['matrix']} {tuple(W.shape)}")
        if max(cfg["positions"]) >= other: raise ValueError(f"{name}.positions must be in 0..{other - 1}")
        if len(cfg["positions"]) != len(cfg["deltas"]): raise ValueError(f"{name}: positions and deltas must have equal length")
        cap = REL_CAP * float(W.detach().pow(2).mean().sqrt())
        if max(abs(x) for x in cfg["deltas"]) > cap: raise ValueError(f"{name}: |delta| must be <= {cap:.4f} ({REL_CAP} x RMS of {cfg['matrix']})")
    return chk


def _nudge_weights(model, cfg, ctx):
    m = copy.deepcopy(model); W = OPS.matrix_table(m)[cfg["matrix"]]; i = cfg["index"]; pos = torch.as_tensor(cfg["positions"]); d = torch.as_tensor(cfg["deltas"], dtype=W.dtype)
    with torch.no_grad():
        if cfg["axis"] == "row": W[i, pos] += d
        else: W[pos, i] += d
    return m


def _scale_slice(model, cfg, ctx):
    m = copy.deepcopy(model); W = OPS.matrix_table(m)[cfg["matrix"]]
    with torch.no_grad():
        for i in cfg["indices"]:
            if cfg["axis"] == "row": W[i, :] *= float(cfg["gamma"])
            else: W[:, i] *= float(cfg["gamma"])
    return m


def _scale_slice_check(cfg, model):
    tab = OPS.matrix_table(model)
    if cfg["matrix"] not in tab: raise ValueError(f"scale_slice.matrix {cfg['matrix']!r} not in this model; existing: {list(tab)}")
    W = tab[cfg["matrix"]]; n = W.shape[0 if cfg["axis"] == "row" else 1]
    if max(cfg["indices"]) >= n or len(cfg["indices"]) > 256: raise ValueError(f"scale_slice.indices must be in 0..{n - 1} (at most 256)")


def enable_value_ops():
    if "nudge_weights" in OPS.REGISTRY: return
    _AX = {"type": "str", "choices": ["row", "col"]}
    OPS.register("nudge_weights", {"matrix": OPS._MAT, "axis": _AX, "index": {"type": "int", "min": 0, "max": 4095}, "positions": {"type": "int_list", "min": 0, "max": 4095},
                 "deltas": {"type": "float_list", "min": -1.0, "max": 1.0, "max_len": MAX_ENTRIES}},
                 f"ADD small amounts to individual weight values: along one row (or column) `index` of a matrix, add deltas[j] to the entry at positions[j] (<= {MAX_ENTRIES} entries, each |delta| <= {REL_CAP} x the matrix's RMS weight). "
                 "Use it for fine value changes (e.g. nudge one neuron's input weights in fc1 = axis row; one neuron's output weights in fc2 = axis col).")(_nudge_weights)
    OPS.RANGE_HOOKS["nudge_weights"] = _slice_check("nudge_weights")
    OPS.register("scale_slice", {"matrix": OPS._MAT, "axis": _AX, "indices": {"type": "int_list", "min": 0, "max": 4095}, "gamma": {"type": "float", "min": 0.0, "max": 3.0}},
                 "Multiply whole rows (axis row) or columns (axis col) of ANY matrix by gamma (up to 256 of them): e.g. rows of fc1 = a neuron's INPUT weights, rows of qkv = specific q/k/v units, columns of an out-projection = a head's output.")(_scale_slice)
    OPS.RANGE_HOOKS["scale_slice"] = _scale_slice_check


# ---------------------------------------------------------------- more value-level edits
def bias_names(model=None):
    if model is not None: return [n for n, p in model.named_parameters() if n.endswith("bias")]
    names = []
    for i in range(OPS.MAX_LAYERS):
        names += [f"enc_layers.{i}.{a}" for a in ("attn.qkv.bias", "attn.out.bias", "mlp.fc1.bias", "mlp.fc2.bias")]
        names += [f"dec_layers.{i}.{a}" for a in ("self_attn.qkv.bias", "self_attn.out.bias", "cross_attn.q.bias", "cross_attn.kv.bias", "cross_attn.out.bias", "mlp.fc1.bias", "mlp.fc2.bias")]
    return names + ["head.bias", "tok_self.bias", "tok_goal.bias", "tok_obs.bias"]


def _bias_check(cfg, model):
    P = dict(model.named_parameters())
    if cfg["bias"] not in P: raise ValueError(f"bias_shift.bias {cfg['bias']!r} not in this model; existing: {bias_names(model)}")
    b = P[cfg["bias"]]
    if max(cfg["positions"]) >= b.numel() or len(cfg["positions"]) != len(cfg["deltas"]): raise ValueError(f"bias_shift.positions must be in 0..{b.numel() - 1} and match deltas in length")
    cap = REL_CAP * max(float(b.detach().pow(2).mean().sqrt()), 0.05)
    if max(abs(x) for x in cfg["deltas"]) > cap: raise ValueError(f"bias_shift: |delta| must be <= {cap:.4f}")


def _bias_shift(model, cfg, ctx):
    m = copy.deepcopy(model); b = dict(m.named_parameters())[cfg["bias"]]
    with torch.no_grad(): b[torch.as_tensor(cfg["positions"])] += torch.as_tensor(cfg["deltas"], dtype=b.dtype)
    return m


def _r1_check(cfg, model):
    tab = OPS.matrix_table(model)
    if cfg["matrix"] not in tab: raise ValueError(f"rank1_update.matrix {cfg['matrix']!r} not in this model; existing: {list(tab)}")
    W = tab[cfg["matrix"]]
    if max(cfg["rows"]) >= W.shape[0] or max(cfg["cols"]) >= W.shape[1]: raise ValueError(f"rank1_update: rows must be < {W.shape[0]} and cols < {W.shape[1]}")
    if len(cfg["rows"]) != len(cfg["u"]) or len(cfg["cols"]) != len(cfg["v"]): raise ValueError("rank1_update: rows/u and cols/v must match in length")
    cap = REL_CAP * float(W.detach().pow(2).mean().sqrt()); big = cfg["alpha"] * max(abs(x) for x in cfg["u"]) * max(abs(x) for x in cfg["v"])
    if big > cap: raise ValueError(f"rank1_update: largest added entry alpha*max|u|*max|v|={big:.4f} exceeds the cap {cap:.4f} ({REL_CAP} x RMS of {cfg['matrix']})")


def _rank1_update(model, cfg, ctx):
    m = copy.deepcopy(model); W = OPS.matrix_table(m)[cfg["matrix"]]; r, c = torch.as_tensor(cfg["rows"]), torch.as_tensor(cfg["cols"])
    u, v = torch.as_tensor(cfg["u"], dtype=W.dtype), torch.as_tensor(cfg["v"], dtype=W.dtype)
    with torch.no_grad(): W[r[:, None], c[None, :]] += float(cfg["alpha"]) * u[:, None] * v[None, :]
    return m


def _blend_matrix(model, cfg, ctx):
    m = copy.deepcopy(model); ref = ctx["ref_models"][int(cfg["ref_ckpt"])]; W, R = OPS.matrix_table(m)[cfg["matrix"]], OPS.matrix_table(ref)[cfg["matrix"]]
    with torch.no_grad(): W.add_(float(cfg["frac"]) * (R - W))
    return m


def enable_more_value_ops():
    if "bias_shift" in OPS.REGISTRY: return
    _AX = {"type": "str", "choices": ["row", "col"]}
    OPS.register("bias_shift", {"bias": {"type": "str", "choices": bias_names}, "positions": {"type": "int_list", "min": 0, "max": 4095}, "deltas": {"type": "float_list", "min": -1.0, "max": 1.0, "max_len": MAX_ENTRIES}},
                 f"ADD small amounts to individual entries of one bias vector (positions[j] += deltas[j]; <= {MAX_ENTRIES} entries, |delta| <= {REL_CAP} x the bias RMS, floor 0.05). Biases shift a unit's operating point without changing its input weights.")(_bias_shift)
    OPS.RANGE_HOOKS["bias_shift"] = _bias_check
    OPS.register("rank1_update", {"matrix": OPS._MAT, "rows": {"type": "int_list", "min": 0, "max": 4095}, "cols": {"type": "int_list", "min": 0, "max": 4095},
                                  "u": {"type": "float_list", "min": -1.0, "max": 1.0, "max_len": 32}, "v": {"type": "float_list", "min": -1.0, "max": 1.0, "max_len": 32}, "alpha": {"type": "float", "min": -1.0, "max": 1.0}},
                 "Add alpha * outer(u, v) to the sub-block of a matrix at the given rows x cols (<= 32 each): a small structured low-rank nudge that changes how a few inputs map to a few outputs together. Largest added entry is capped at 2 x the matrix RMS.")(_rank1_update)
    OPS.RANGE_HOOKS["rank1_update"] = _r1_check
    OPS.register("blend_matrix", {"matrix": OPS._MAT, "ref_ckpt": {"type": "int", "min": 0, "max": OPS.N_CKPT - 1}, "frac": {"type": "float", "min": -0.5, "max": 1.0}},
                 "Move ONE matrix a fraction of the way toward its value in an earlier checkpoint (frac 1 = fully revert that matrix; negative = extrapolate away). revert_delta does this for the whole model; this is per matrix.")(_blend_matrix)
    OPS.register("blend_slice", {"matrix": OPS._MAT, "axis": _AX, "indices": {"type": "int_list", "min": 0, "max": 4095}, "ref_ckpt": {"type": "int", "min": 0, "max": OPS.N_CKPT - 1}, "frac": {"type": "float", "min": -0.5, "max": 1.0}},
                 "Move chosen rows/columns of ONE matrix (up to 256; e.g. specific neurons' fc1 rows or fc2 columns, or one head's slice) toward their values in an earlier checkpoint. frac 1 = restore them to that checkpoint. A surgical version of revert_delta.")(_blend_slice)
    OPS.RANGE_HOOKS["blend_slice"] = _blend_slice_check


# ---------------------------------------------------------------- structured, component-specific edits
def _head_views(m, kind, layer, h):
    """list of (weight_or_bias_view) that together define ONE head: its Q, K, V rows (+biases) and its output-projection columns."""
    at = OPS.attn_module(m, kind, layer); H = OPS.n_heads_of(m, at); d = at.out.in_features; dh = d // H; s = slice(h * dh, (h + 1) * dh)
    if kind in ("enc_self", "dec_self"): W, b = at.qkv.weight, at.qkv.bias; v = [W[s], W[d:2 * d][s], W[2 * d:][s], b[s], b[d:2 * d][s], b[2 * d:][s]]
    else: v = [at.q.weight[s], at.kv.weight[:d][s], at.kv.weight[d:][s], at.q.bias[s], at.kv.bias[:d][s], at.kv.bias[d:][s]]
    return v + [at.out.weight[:, s]]


def _transplant_check(cfg, model):
    kind = cfg["kind"]; nl = OPS.n_layers(model, kind); H = OPS.model_dims(model)["n_heads"][kind]
    for k in ("layer", "src_layer"):
        if not 0 <= cfg[k] < nl: raise ValueError(f"head_transplant.{k} must be in 0..{nl - 1} for kind {kind}")
    for k in ("head", "src_head"):
        if not 0 <= cfg[k] < H: raise ValueError(f"head_transplant.{k} must be in 0..{H - 1}")
    if (cfg["layer"], cfg["head"]) == (cfg["src_layer"], cfg["src_head"]): raise ValueError("head_transplant: source and destination are the same head")


def _head_transplant(model, cfg, ctx):
    m = copy.deepcopy(model); f = float(cfg["frac"]); dst = _head_views(m, cfg["kind"], cfg["layer"], cfg["head"]); src = [t.detach().clone() for t in _head_views(m, cfg["kind"], cfg["src_layer"], cfg["src_head"])]
    with torch.no_grad():
        for a, b in zip(dst, src): a.mul_(1 - f).add_(f * b)
    return m


def _blend_slice_check(cfg, model):
    tab = OPS.matrix_table(model)
    if cfg["matrix"] not in tab: raise ValueError(f"blend_slice.matrix {cfg['matrix']!r} not in this model; existing: {list(tab)}")
    n = tab[cfg["matrix"]].shape[0 if cfg["axis"] == "row" else 1]
    if max(cfg["indices"]) >= n or len(cfg["indices"]) > 256: raise ValueError(f"blend_slice.indices must be in 0..{n - 1} (at most 256)")


def _blend_slice(model, cfg, ctx):
    m = copy.deepcopy(model); ref = ctx["ref_models"][int(cfg["ref_ckpt"])]; W, R = OPS.matrix_table(m)[cfg["matrix"]], OPS.matrix_table(ref)[cfg["matrix"]]; idx = torch.as_tensor(cfg["indices"])
    with torch.no_grad():
        if cfg["axis"] == "row": W[idx] += float(cfg["frac"]) * (R[idx] - W[idx])
        else: W[:, idx] += float(cfg["frac"]) * (R[:, idx] - W[:, idx])
    return m


def enable_structured_ops():
    if "head_transplant" in OPS.REGISTRY: return
    _AX = {"type": "str", "choices": ["row", "col"]}
    OPS.register("head_transplant", {"kind": {"type": "str", "choices": OPS.HEAD_KINDS}, "layer": {"type": "int", "min": 0, "max": OPS.MAX_LAYERS - 1}, "head": {"type": "int", "min": 0, "max": OPS.MAX_HEADS - 1},
                                     "src_layer": {"type": "int", "min": 0, "max": OPS.MAX_LAYERS - 1}, "src_head": {"type": "int", "min": 0, "max": OPS.MAX_HEADS - 1}, "frac": {"type": "float", "min": 0.0, "max": 1.0}},
                 "Replace a fraction of one attention head (its Q, K, V projections, biases and output columns) with another head of the SAME kind: frac 1 = full copy of the source head into the destination. Use it to overwrite a suspect head with a healthy one, or to merge two heads.")(_head_transplant)
    OPS.RANGE_HOOKS["head_transplant"] = _transplant_check


def enable():
    enable_head_temperature(); enable_value_ops(); enable_more_value_ops(); enable_structured_ops()
