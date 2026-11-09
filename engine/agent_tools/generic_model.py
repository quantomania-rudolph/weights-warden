"""GENERIC MODE: use the model-agnostic part of the op library on ANY torch.nn.Module (MLP, CNN/ResNet backbones, GPT-2/Llama/Qwen via transformers, ViT, ...).
What works generically (needs only a name -> 2-D weight table): project_out_top_svd, rank_truncate, shrink_subspace, revert_delta, nudge_weights, scale_slice, rank1_update, bias_shift, blend_matrix, blend_slice, composite.
What needs a ModelMap (structure knowledge): scale_head, scale_neurons, prune_neurons, scale_token_embed, head_transplant, head_temperature, the EAP circuit finder, the attention/QKV views.
  -> in generic mode those structure-specific ops are UNREGISTERED (the agent cannot call them) and guess_roles() tells the agent which rows/cols to hit with scale_slice/nudge_weights instead.
CONVENTIONS: nn.Linear weight = (out, in)  => axis row = an OUTPUT unit (e.g. one head's q rows, one MLP neuron's input weights).  GPT-2 Conv1D weight = (in, out) => axis col = an output unit.
LIMITS: edits deep-copy the model (fine up to ~1B params; larger needs an in-place apply/undo log); evaluation for LLMs is a benchmark, not a closed-loop episode: you must supply an evaluator with the core/adapter.py shape."""
import os, re, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import ops as OPS

STRUCTURE_OPS = ["scale_head", "scale_neurons", "prune_neurons", "scale_token_embed", "head_transplant", "head_temperature"]
_ROLES = [(r"(c_attn|qkv|query_key_value|in_proj)", "attn_qkv_fused"), (r"(q_proj|\.query\b|\.q\b|wq\b|to_q)", "attn_q"), (r"(k_proj|\.key\b|\.k\b|wk\b|to_k)", "attn_k"), (r"(v_proj|\.value\b|\.v\b|wv\b|to_v)", "attn_v"),
          (r"(o_proj|out_proj|attn\.c_proj|\.dense\b|wo\b|to_out)", "attn_out"), (r"(gate_proj|up_proj|fc1|c_fc|linear1|w1|w3|dense_h_to_4h)", "ffn_in"), (r"(down_proj|fc2|mlp\.c_proj|linear2|w2|dense_4h_to_h)", "ffn_out"),
          (r"(embed|wte|wpe|tok_|emb)", "embedding"), (r"(lm_head|head|classifier|output_proj)", "output_head")]


def generic_matrix_table(model):
    return {n: p for n, p in model.named_parameters() if p.ndim == 2}


def guess_roles(model):
    """{matrix_name: (role, shape)} by name patterns; 'unknown' when nothing matches (inspect with weights/qkv before editing)."""
    out = {}
    for n, p in generic_matrix_table(model).items():
        r = next((role for pat, role in _ROLES if re.search(pat, n)), "unknown"); out[n] = (r, tuple(p.shape))
    return out


_SAVED = {}


def enable_generic():
    """switch the (process-global) op library to generic mode. returns the list of ops removed. restore() undoes it."""
    if _SAVED: return []
    _SAVED["matrix_table"] = OPS.matrix_table; _SAVED["ops"] = {k: OPS.REGISTRY[k] for k in STRUCTURE_OPS if k in OPS.REGISTRY}; _SAVED["hooks"] = {k: OPS.RANGE_HOOKS[k] for k in STRUCTURE_OPS if k in OPS.RANGE_HOOKS}
    OPS.matrix_table = generic_matrix_table
    for k in _SAVED["ops"]: del OPS.REGISTRY[k]
    for k in _SAVED["hooks"]: del OPS.RANGE_HOOKS[k]
    return list(_SAVED["ops"])


def restore():
    if not _SAVED: return
    OPS.matrix_table = _SAVED["matrix_table"]; OPS.REGISTRY.update(_SAVED["ops"]); OPS.RANGE_HOOKS.update(_SAVED["hooks"]); _SAVED.clear()
