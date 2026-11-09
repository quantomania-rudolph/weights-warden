"""DECLARATIVE ModelMaps.  Adding an architecture = filling ~12 fields (a JSON spec), not writing code.  Three ways to get a spec:
  1. propose_spec(model)  - the harness inspects the module tree (shapes + names + config) and PROPOSES a spec, with warnings for anything it could not establish;
  2. an agent / human edits the proposal after reading `inspect(model)` (module paths, shapes);
  3. acclimate.py then PROVES or REJECTS the spec on the live model (hook-equivalence of every op, exact residual decomposition, sequential-residual check, attribution-vs-exact).
A spec the acclimation cannot verify never reaches the editing agent.
SPEC: {family, blocks_path, n_heads, n_kv_heads, head_dim, d_model, final_norm, q_layout: 'contiguous'|'unknown',
       block: {attn_norm, mlp_norm, o, o_conv, q, q_conv, qkv_fused, down, down_conv, ups: [[path, conv1d?], ...]}}   (paths inside `block` are relative to each block module; blocks_path/final_norm are relative to the model)"""
import json, torch, torch.nn as nn
import structure as S


def _lin_like(m): return isinstance(m, nn.Linear) or (type(m).__name__ == "Conv1D" and hasattr(m, "weight"))
def _dims(m): w = m.weight; return (w.shape[1], w.shape[0]) if isinstance(m, nn.Linear) else (w.shape[0], w.shape[1])      # (in, out) for both


def inspect(model, depth=4, max_rows=80):
    """module tree with shapes - what an agent reads before writing/repairing a spec."""
    rows = []
    for n, m in model.named_modules():
        if n.count(".") > depth or not n: continue
        sh = tuple(m.weight.shape) if hasattr(m, "weight") and hasattr(m.weight, "shape") else ""
        rows.append(f"{n:60s} {type(m).__name__:22s} {sh}")
    return rows[:max_rows], {k: getattr(model.config, k) for k in ("hidden_size", "n_embd", "d_model", "num_attention_heads", "n_head", "num_heads", "num_key_value_heads", "head_dim", "intermediate_size", "ffn_dim", "num_hidden_layers") if getattr(getattr(model, "config", None), k, None) is not None} if hasattr(model, "config") else {}


def propose_spec(model, n_heads=None, n_kv_heads=None, head_dim=None):
    warns = []; cfg = getattr(model, "config", None); g = lambda *ns: next((getattr(cfg, n) for n in ns if cfg is not None and getattr(cfg, n, None) is not None), None)
    stacks = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 1 and len({type(c).__name__ for c in m}) == 1 and sum(p.numel() for p in m.parameters()) > 0]
    if not stacks: raise S.Unsupported("no repeated block stack (ModuleList of identical blocks) found")
    bpath, blocks = max(stacks, key=lambda nm: sum(p.numel() for p in nm[1].parameters()))
    if len(stacks) > 1 and sum(1 for n, m in stacks if sum(p.numel() for p in m.parameters()) > 0.5 * sum(p.numel() for p in blocks.parameters())) > 1: warns.append("several comparable block stacks (encoder-decoder?): using the largest; check blocks_path")
    H = n_heads or g("num_attention_heads", "n_head", "num_heads", "attention_heads"); d = g("hidden_size", "n_embd", "d_model")
    if H is None or d is None: raise S.Unsupported("cannot read n_heads / d_model from config: pass n_heads=... (and d_model via config)")
    Hkv = n_kv_heads or g("num_key_value_heads") or H; dh = head_dim or g("head_dim") or d // H; blk = blocks[0]
    lin = [(n, m) for n, m in blk.named_modules() if n and _lin_like(m)]; attn = [(n, m) for n, m in lin if "attn" in n.lower() or "attention" in n.lower()]; mlp = [(n, m) for n, m in lin if (n, m) not in attn]
    norms = [n for n, m in blk.named_modules() if n and ("norm" in type(m).__name__.lower() or type(m).__name__.lower().startswith("ln"))]
    if len(norms) != 2: raise S.Unsupported(f"expected 2 norm layers per block (pre-norm attn + mlp), found {norms}: post-norm / extra-norm architectures need a hand-written spec")
    o = [(n, m) for n, m in attn if _dims(m) == (H * dh, d)]
    if not o: raise S.Unsupported(f"no attention output projection with shape ({H * dh} -> {d}) among {[n for n, _ in attn]}")
    o_path, om = o[-1]; fused = [(n, m) for n, m in attn if _dims(m) == (d, (H + 2 * Hkv) * dh) and (n, m) != (o_path, om)]
    if fused: qn, qm = fused[0]; qfused = True; warns.append("fused qkv: layout cannot be established from shapes, so q_layout='unknown' and head_temperature is DISABLED. If you know it is [q|k|v] contiguous (GPT-2 style) set q_layout='contiguous' in the spec")
    else:
        qs = [(n, m) for n, m in attn if _dims(m) == (d, H * dh) and n != o_path]
        if not qs: raise S.Unsupported("no query projection found")
        qn, qm = qs[0]; qfused = False
    dn = [(n, m) for n, m in mlp if _dims(m)[1] == d and _dims(m)[0] != d]
    if not dn: raise S.Unsupported(f"no MLP down projection (-> {d}) among {[n for n, _ in mlp]}")
    down_path, dm = dn[-1]; dff = _dims(dm)[0]; ups = [(n, m) for n, m in mlp if _dims(m)[1] == dff and n != down_path]
    if not ups: raise S.Unsupported("no MLP up/gate projection found")
    out_norm = [n for n, m in model.named_modules() if n and not n.startswith(bpath) and "norm" in type(m).__name__.lower()]
    if not out_norm: raise S.Unsupported("no final norm layer found outside the block stack")
    spec = {"family": "auto:" + type(model).__name__, "blocks_path": bpath, "n_heads": H, "n_kv_heads": Hkv, "head_dim": dh, "d_model": d, "final_norm": out_norm[-1], "q_layout": "unknown" if qfused else "contiguous",
            "block": {"attn_norm": norms[0], "mlp_norm": norms[1], "o": o_path, "o_conv": not isinstance(om, nn.Linear), "q": qn, "q_conv": not isinstance(qm, nn.Linear), "qkv_fused": qfused,
                      "down": down_path, "down_conv": not isinstance(dm, nn.Linear), "ups": [[n, not isinstance(m, nn.Linear)] for n, m in ups]}}
    return spec, warns


def from_spec(model, spec):
    b = spec["block"]; blocks = model.get_submodule(spec["blocks_path"]); t = S.TMap(); t.family = spec.get("family", "custom"); t.n_heads = spec["n_heads"]; t.n_kv_heads = spec.get("n_kv_heads", t.n_heads)
    t.head_dim, t.d_model = spec["head_dim"], spec["d_model"]; t.q_layout = spec.get("q_layout", "contiguous"); G = lambda blk, p: blk.get_submodule(p)
    t.blocks = [S.Blk(mod=blk, attn_norm=G(blk, b["attn_norm"]), mlp_norm=G(blk, b["mlp_norm"]), o=G(blk, b["o"]), o_conv=b.get("o_conv", False), q=G(blk, b["q"]), q_conv=b.get("q_conv", False), qkv_fused=b.get("qkv_fused", False),
                       down=G(blk, b["down"]), down_conv=b.get("down_conv", False), ups=[(G(blk, p), c) for p, c in b["ups"]]) for blk in blocks]
    t.final_norm = model.get_submodule(spec["final_norm"]); t.n_layers = len(t.blocks); w = t.blocks[0].down.weight; t.d_ff = w.shape[0] if t.blocks[0].down_conv else w.shape[1]; return t
