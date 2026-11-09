"""ModelMap: discover the structure of an arbitrary transformer (or MLP) so head / neuron ops and patching work on it WITHOUT per-model code.
Supported out of the box (tested): GPT-2 family (HF Conv1D, fused c_attn), Llama-style families (q/k/v/o_proj, gate/up/down_proj, RMSNorm; Llama, Qwen2, Mistral, Gemma-like, with GQA), plain nn.Sequential MLPs.
Anything else raises Unsupported with the module names it saw - never a silent wrong answer.  Register a new family with register_family(name, detect_fn, build_fn).
CONVENTIONS: nn.Linear weight is (out, in); GPT-2 Conv1D weight is (in, out).  `o_head(blk, h)` etc. return WRITABLE views so ops can scale/nudge them in place on a copy."""
import torch, torch.nn as nn


class Unsupported(Exception): pass


class Blk:
    def __init__(self, **kw): self.__dict__.update(kw)


class TMap:
    kind = "transformer"; q_layout = "contiguous"
    def head_slice(self, h): return slice(h * self.head_dim, (h + 1) * self.head_dim)
    def o_head(self, b, h):        # view of the output-projection weights that read head h
        W = b.o.weight; s = self.head_slice(h); return W[s, :] if b.o_conv else W[:, s]
    def down_units(self, b, u):    # (weight, index) of the weights that read MLP hidden units u -> residual.  NOTE: list indexing COPIES, so callers must write with W[idx] = ..., never W[idx].mul_()
        return b.down.weight, ((u, slice(None)) if b.down_conv else (slice(None), u))
    def up_units(self, b, u):      # list of (weight, weight_index, bias|None) that WRITE MLP hidden units u
        return [(mod.weight, ((slice(None), u) if conv else (u, slice(None))), mod.bias) for mod, conv in b.ups]
    def q_head(self, b, h):        # (weight_view, bias_view|None) producing head h's queries
        s = self.head_slice(h)
        if b.qkv_fused:            # fused [q|k|v] with q FIRST and contiguous per head (GPT-2 Conv1D: output columns; nn.Linear: rows).  Interleaved layouts (GPT-NeoX) are NOT this: q_layout must say so
            if self.q_layout != "contiguous": raise Unsupported("fused qkv layout not declared contiguous: head_temperature disabled")
            return (b.q.weight[:, s], b.q.bias[s]) if b.q_conv else (b.q.weight[s, :], b.q.bias[s])
        return b.q.weight[s, :], (None if b.q.bias is None else b.q.bias[s])


class MLPMap:
    kind = "mlp"
    def __init__(self, model):
        self.lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(self.lin) < 2: raise Unsupported("MLP needs >= 2 nn.Linear layers")
        self.n_hidden = len(self.lin) - 1; self.widths = [l.out_features for l in self.lin[:-1]]


_FAMILIES = []


def register_family(name, detect, build): _FAMILIES.append((name, detect, build))


def _cfg(model, *names, default=None):
    c = getattr(model, "config", None)
    for n in names:
        if c is not None and getattr(c, n, None) is not None: return getattr(c, n)
    return default


def _gpt2_detect(m): b = getattr(m, "transformer", m); return hasattr(b, "h") and len(b.h) and hasattr(b.h[0], "attn") and hasattr(b.h[0].attn, "c_attn")
def _gpt2_build(m):
    base = getattr(m, "transformer", m); t = TMap(); t.family = "gpt2"; t.base = base; H = _cfg(m, "n_head", "num_attention_heads"); d = _cfg(m, "n_embd", "hidden_size"); t.n_heads = t.n_kv_heads = H; t.head_dim = d // H; t.d_model = d
    t.blocks = [Blk(mod=b, attn_norm=b.ln_1, mlp_norm=b.ln_2, o=b.attn.c_proj, o_conv=True, q=b.attn.c_attn, q_conv=True, qkv_fused=True, down=b.mlp.c_proj, down_conv=True, ups=[(b.mlp.c_fc, True)], mlp=b.mlp) for b in base.h]
    t.final_norm = base.ln_f; t.n_layers = len(t.blocks); t.d_ff = t.blocks[0].down.weight.shape[0]; return t


def _llama_detect(m):
    b = getattr(m, "model", m); L = getattr(b, "layers", None)
    return L is not None and len(L) and hasattr(L[0], "self_attn") and hasattr(L[0].self_attn, "q_proj") and hasattr(L[0], "mlp") and hasattr(L[0].mlp, "down_proj")
def _llama_build(m):
    base = getattr(m, "model", m); t = TMap(); t.family = "llama-like"; t.base = base; H = _cfg(m, "num_attention_heads"); d = _cfg(m, "hidden_size"); t.n_heads = H; t.n_kv_heads = _cfg(m, "num_key_value_heads", default=H)
    t.head_dim = _cfg(m, "head_dim", default=d // H); t.d_model = d
    if base.layers[0].self_attn.o_proj.weight.shape[1] != H * t.head_dim: raise Unsupported("o_proj input width != n_heads*head_dim")
    ups = lambda l: [(x, False) for x in (getattr(l.mlp, "gate_proj", None), l.mlp.up_proj) if x is not None]
    t.blocks = [Blk(mod=l, attn_norm=l.input_layernorm, mlp_norm=l.post_attention_layernorm, o=l.self_attn.o_proj, o_conv=False, q=l.self_attn.q_proj, q_conv=False, qkv_fused=False, down=l.mlp.down_proj, down_conv=False, ups=ups(l), mlp=l.mlp) for l in base.layers]
    t.final_norm = base.norm; t.n_layers = len(t.blocks); t.d_ff = t.blocks[0].down.weight.shape[1]; return t


register_family("gpt2", _gpt2_detect, _gpt2_build); register_family("llama-like", _llama_detect, _llama_build)


def discover(model, spec=None):
    """spec: a dict or a path to a JSON ModelMap spec (see spec_map.py) - overrides auto-detection."""
    if spec is not None:
        import spec_map; return spec_map.from_spec(model, spec if isinstance(spec, dict) else __import__("json").load(open(spec)))
    for name, det, build in _FAMILIES:
        try:
            if det(model): return build(model)
        except Unsupported: raise
        except Exception: continue
    lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(lin) >= 2 and not any("attn" in n or "attention" in n for n, _ in model.named_modules()): return MLPMap(model)       # NOTE: Linear order must match forward order - acclimate() verifies this with a hook equivalence test
    names = [n for n, _ in model.named_modules()][:25]
    raise Unsupported(f"unknown architecture; first modules: {names}. Add a family with structure.register_family(). Matrix-level generic ops still work (generic_model.py).")
