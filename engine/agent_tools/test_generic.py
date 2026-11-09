import os, sys, copy, torch, torch.nn as nn, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import generic_model as GM, more_ops, ops as OPS

@pytest.fixture(autouse=True)
def _generic():
    keep = set(OPS.REGISTRY); hooks = set(OPS.RANGE_HOOKS); more_ops.enable(); GM.enable_generic(); yield
    GM.restore()
    for k in set(OPS.REGISTRY) - keep: del OPS.REGISTRY[k]
    for k in set(OPS.RANGE_HOOKS) - hooks: del OPS.RANGE_HOOKS[k]

def test_structure_ops_are_unregistered_and_generic_ops_work_on_a_50_neuron_mlp():
    assert not any(k in OPS.REGISTRY for k in GM.STRUCTURE_OPS) and "scale_slice" in OPS.REGISTRY
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(10, 50), nn.GELU(), nn.Linear(50, 3)); x = torch.randn(8, 10); W0 = m[0].weight.detach().clone()
    e = OPS.apply("scale_slice", {"matrix": "0.weight", "axis": "row", "indices": [3, 4], "gamma": 0.0}, m); assert float(e[0].weight[3].abs().sum()) == 0 and torch.equal(e[0].weight[5], W0[5]) and not torch.allclose(e(x), m(x))
    e = OPS.apply("nudge_weights", {"matrix": "2.weight", "axis": "col", "index": 7, "positions": [0, 2], "deltas": [0.05, -0.05]}, m); assert int((e[2].weight != m[2].weight).sum()) == 2
    e = OPS.apply("rank_truncate", {"matrix": "0.weight", "rank": 2}, m); assert torch.linalg.matrix_rank(e[0].weight) <= 2
    e = OPS.apply("bias_shift", {"bias": "0.bias", "positions": [1], "deltas": [0.01]}, m); assert abs(float(e[0].bias[1] - m[0].bias[1]) - 0.01) < 1e-7
    OPS.apply("composite", {"ops": [{"op": "scale_slice", "cfg": {"matrix": "0.weight", "axis": "row", "indices": [0], "gamma": 0.5}}, {"op": "rank1_update", "cfg": {"matrix": "2.weight", "rows": [0], "cols": [0], "u": [0.1], "v": [0.1], "alpha": 0.1}}]}, m)
    for op, c in (("scale_slice", {"matrix": "nope.weight", "axis": "row", "indices": [0], "gamma": 0.5}), ("scale_slice", {"matrix": "0.weight", "axis": "row", "indices": [60], "gamma": 0.5})):
        with pytest.raises(ValueError): OPS.validate_cfg(op, c, m)

def test_tiny_gpt2_generic_edit_and_role_guess():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0); m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=50, n_positions=16)).eval(); ids = torch.randint(0, 50, (2, 8)); roles = GM.guess_roles(m)
    assert roles["transformer.h.0.attn.c_attn.weight"][0] == "attn_qkv_fused" and roles["transformer.h.0.mlp.c_fc.weight"][0] == "ffn_in" and roles["transformer.h.0.mlp.c_proj.weight"][0] in ("ffn_out", "attn_out")
    base = m(ids).logits.detach(); e = OPS.apply("scale_slice", {"matrix": "transformer.h.0.attn.c_attn.weight", "axis": "col", "indices": list(range(0, 16)), "gamma": 0.0}, m)        # Conv1D: cols = output units -> zero head-0's q
    assert not torch.allclose(e(ids).logits.detach(), base) and torch.equal(m(ids).logits.detach(), base)                                                                   # original untouched
    OPS.apply("rank1_update", {"matrix": "transformer.h.1.mlp.c_fc.weight", "rows": [0, 1], "cols": [0, 1], "u": [0.1, 0.1], "v": [0.1, 0.1], "alpha": 0.1}, m)
    OPS.apply("rank_truncate", {"matrix": "transformer.h.0.mlp.c_fc.weight", "rank": 4}, m)
