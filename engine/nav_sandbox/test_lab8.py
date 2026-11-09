import os, sys, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import fit_teacher as FT

def test_golden_ridge_recovers_exact_linear_teacher():
    torch.manual_seed(0); X = torch.randn(500, 16); Wt = torch.randn(8, 16); W = FT.ridge_to_teacher(torch.zeros(8, 16), X, X @ Wt.T, lam=1e-8)
    assert torch.allclose(W, Wt, atol=1e-3)

def test_property_large_lambda_stays_at_original_weights():
    torch.manual_seed(1); X = torch.randn(200, 8); W0 = torch.randn(4, 8); W = FT.ridge_to_teacher(W0, X, torch.randn(200, 4), lam=1e6)
    assert torch.allclose(W, W0, atol=1e-3)

def test_poison_guard_refuses_an_overfit_fit():
    torch.manual_seed(2); d = 40; W0 = torch.zeros(4, d); Xf = torch.randn(30, d); Yf = torch.randn(30, 4)       # pure noise targets, more features than samples: fits train, fails held-out
    Xh = torch.randn(200, d); Yh = torch.randn(200, 4); r = FT.fit_with_guard(W0, Xf, Yf, Xh, Yh, lam=1e-8)
    assert not r["applied"] and torch.equal(r["W"], W0)

def test_golden_guard_accepts_a_real_linear_teacher():
    torch.manual_seed(3); Wt = torch.randn(4, 12); Xf, Xh = torch.randn(400, 12), torch.randn(400, 12); r = FT.fit_with_guard(torch.zeros(4, 12), Xf, Xf @ Wt.T, Xh, Xh @ Wt.T, lam=1e-6)
    assert r["applied"] and r["held_rmse"] < 1e-2

def test_leak_research_layouts_disjoint_from_every_other_split():
    import research as R, env as E
    L = R.research_layouts(); keys = {E.layout_key(l) for l in L}; assert len(keys) == 100
    for name, ks in E.split_keys().items(): assert not (keys & ks), name

def test_bridge_ridge_op_is_off_by_default_and_registers_on_enable():
    import ops as OPS
    assert "ridge_head" not in OPS.REGISTRY
    import ridge_op; ridge_op.enable(); assert "ridge_head" in OPS.REGISTRY
    try: OPS.validate_cfg("ridge_head", {"lam": -1.0, "max_ratio": 1.5}); assert False
    except ValueError: pass

def test_golden_ridge_head_with_self_as_teacher_changes_nothing_much():
    import ridge_op, policy as P, copy, torch; ridge_op.enable()
    m = P.NavPolicy(); out = ridge_op._ridge_head(m, {"lam": 1.0, "max_ratio": 3.0}, {"ref_models": {0: copy.deepcopy(m)}})
    import research as R; o = R.research_obs(200)
    assert float((out(o) - m(o)).abs().max()) < 1e-2

def test_head_temperature_gamma1_is_identity_and_changes_logits():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent_tools")); import more_ops, ops as OPS, policy as P; more_ops.enable(); m = P.NavPolicy(); o = torch.rand(4, 21)
    e1 = OPS.apply("head_temperature", {"kind": "enc_self", "layer": 0, "head": 1, "gamma": 1.0}, m); e2 = OPS.apply("head_temperature", {"kind": "dec_cross", "layer": 0, "head": 1, "gamma": 2.0}, m)
    assert torch.allclose(e1(o), m(o)) and not torch.allclose(e2(o), m(o))
    try: OPS.validate_cfg("head_temperature", {"kind": "enc_self", "layer": 0, "head": 0, "gamma": 9.0}); assert False
    except ValueError: pass

def test_value_ops_golden_poison_and_composite():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent_tools")); import more_ops, ops as OPS, policy as P; more_ops.enable(); m = P.NavPolicy()
    W0 = OPS.matrix_table(m)["enc0.fc1"].detach().clone(); e = OPS.apply("nudge_weights", {"matrix": "enc0.fc1", "axis": "row", "index": 3, "positions": [0, 5], "deltas": [0.01, -0.02]}, m)
    W1 = OPS.matrix_table(e)["enc0.fc1"]; d = (W1 - W0); assert abs(d[3, 0] - 0.01) < 1e-7 and abs(d[3, 5] + 0.02) < 1e-7 and int((d != 0).sum()) == 2 and torch.equal(OPS.matrix_table(m)["enc0.fc1"], W0)
    e2 = OPS.apply("scale_slice", {"matrix": "enc0.fc1", "axis": "row", "indices": [1, 2], "gamma": 0.5}, m); assert torch.allclose(OPS.matrix_table(e2)["enc0.fc1"][1], W0[1] * 0.5) and torch.equal(OPS.matrix_table(e2)["enc0.fc1"][0], W0[0])
    bad = [{"matrix": "enc0.fc1", "axis": "row", "index": 3, "positions": [0], "deltas": [0.9]},            # over the relative cap
           {"matrix": "enc0.fc1", "axis": "row", "index": 99999, "positions": [0], "deltas": [0.001]},      # index out of range (schema or model check)
           {"matrix": "enc0.fc1", "axis": "row", "index": 3, "positions": [0, 1], "deltas": [0.001]},      # length mismatch
           {"matrix": "enc0.fc1", "axis": "row", "index": 3, "positions": [0], "deltas": [float("nan")]}]
    for c in bad:
        try: OPS.validate_cfg("nudge_weights", c, m); assert False, c
        except ValueError: pass
    OPS.apply("composite", {"ops": [{"op": "nudge_weights", "cfg": {"matrix": "enc0.fc1", "axis": "row", "index": 3, "positions": [0], "deltas": [0.01]}}, {"op": "scale_slice", "cfg": {"matrix": "dec0.fc2", "axis": "col", "indices": [4], "gamma": 0.0}}]}, m)

def test_writer_decomposition_is_exact_and_declared_heads_compare():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent_tools")); import action_heads as AH, policy as P, torch.nn as nn
    m = P.NavPolicy(); o = torch.rand(6, 21); dec, err = AH.writer_decomposition(m, o); assert err < 1e-4
    ro = AH.decoder_readouts(m, o); assert torch.allclose(ro["after_D2"], m(o), atol=1e-5)
    class Nudged(nn.Module):
        def __init__(s, base): super().__init__(); s.base = base; s.nudge = nn.Linear(21, 8 * 2)
        def action_heads(s, x): b = s.base(x); n = s.nudge(x).view(-1, 8, 2) * 0.1; return {"base": b, "nudge": n, "final": b + n}
    n = Nudged(m); y = m(o).detach() + 0.05; r = AH.compare_heads(n, o, y, o, y)
    assert "_nudge" in r["heldout_vs_expert"] and 0 <= r["heldout_vs_expert"]["_nudge"]["fraction_of_states_helped"] <= 1

def test_more_value_ops_and_read_tools():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent_tools")); import more_ops, read_tools as RT, ops as OPS, policy as P, copy; more_ops.enable(); m = P.NavPolicy()
    b0 = dict(m.named_parameters())["head.bias"].detach().clone(); e = OPS.apply("bias_shift", {"bias": "head.bias", "positions": [1], "deltas": [0.05]}, m); b1 = dict(e.named_parameters())["head.bias"]
    assert abs(float(b1[1] - b0[1]) - 0.05) < 1e-7 and float(b1[0]) == float(b0[0])
    W0 = OPS.matrix_table(m)["enc0.fc2"].detach().clone(); e = OPS.apply("rank1_update", {"matrix": "enc0.fc2", "rows": [0, 1], "cols": [2, 3, 4], "u": [1.0, 0.5], "v": [0.2, 0.1, -0.1], "alpha": 0.01}, m)
    d = OPS.matrix_table(e)["enc0.fc2"] - W0; assert abs(float(d[1, 3]) - 0.01 * 0.5 * 0.1) < 1e-8 and int((d != 0).sum()) == 6
    ref = {0: P.NavPolicy()}; e = OPS.apply("blend_matrix", {"matrix": "enc0.fc2", "ref_ckpt": 0, "frac": 1.0}, m, ctx={"ref_models": ref}); assert torch.allclose(OPS.matrix_table(e)["enc0.fc2"], OPS.matrix_table(ref[0])["enc0.fc2"])
    for op, c in (("bias_shift", {"bias": "head.bias", "positions": [0], "deltas": [0.9]}), ("rank1_update", {"matrix": "enc0.fc2", "rows": [0], "cols": [0], "u": [1.0], "v": [1.0], "alpha": 1.0}), ("rank1_update", {"matrix": "enc0.fc2", "rows": [0, 1], "cols": [0], "u": [0.1], "v": [0.1], "alpha": 0.1})):
        try: OPS.validate_cfg(op, c, m); assert False, op
        except ValueError: pass
    # read tools: shapes, consistency with the model, caps
    r = RT.read_weights(m, "enc0.qkv", "0:2", "0:3"); assert r["slice_shape"] == [2, 3] and np.allclose(r["values"], OPS.matrix_table(m)["enc0.qkv"][:2, :3].detach().numpy(), atol=1e-4)
    try: RT.read_weights(m, "enc0.qkv"); assert False
    except ValueError: pass
    q = RT.qkv_view(m, "enc_self", 0, 1); assert q["d_head"] == 32 and len(q["QK_circuit_top_singular_values"]) == 6 and "raw" not in q
    a = RT.attention_view(m, torch.rand(20, 21), "dec_cross", 0, 2); assert np.allclose(np.array(a["mean_attention"]).sum(-1), 1.0, atol=5e-3)
    ac = RT.neuron_activity(m, torch.rand(20, 21), torch.rand(20, 21), "dec", 1); assert ac["n_units"] == 512 and len(ac["units_most_different_A_vs_B"]) == 12

def test_structured_ops_head_transplant_and_blend_slice():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent_tools")); import more_ops, ops as OPS, policy as P; more_ops.enable(); m = P.NavPolicy()
    e = OPS.apply("head_transplant", {"kind": "dec_cross", "layer": 0, "head": 1, "src_layer": 1, "src_head": 2, "frac": 1.0}, m); v0, v1, vm = more_ops._head_views(e, "dec_cross", 0, 1), more_ops._head_views(e, "dec_cross", 1, 2), more_ops._head_views(m, "dec_cross", 0, 0)
    assert all(torch.equal(a, b) for a, b in zip(v0, v1)); assert all(torch.equal(a, b) for a, b in zip(more_ops._head_views(e, "dec_cross", 0, 0), vm))          # other heads untouched
    h = OPS.apply("head_transplant", {"kind": "enc_self", "layer": 0, "head": 0, "src_layer": 1, "src_head": 3, "frac": 0.0}, m); assert torch.equal(OPS.matrix_table(h)["enc0.qkv"], OPS.matrix_table(m)["enc0.qkv"])
    for c in ({"kind": "enc_self", "layer": 0, "head": 0, "src_layer": 0, "src_head": 0, "frac": 1.0}, {"kind": "enc_self", "layer": 0, "head": 9, "src_layer": 0, "src_head": 1, "frac": 1.0}):
        try: OPS.validate_cfg("head_transplant", c, m); assert False
        except ValueError: pass
    ref = {3: P.NavPolicy()}; W0 = OPS.matrix_table(m)["dec0.fc2"].detach().clone(); R = OPS.matrix_table(ref[3])["dec0.fc2"]
    b = OPS.apply("blend_slice", {"matrix": "dec0.fc2", "axis": "col", "indices": [5, 6], "ref_ckpt": 3, "frac": 1.0}, m, ctx={"ref_models": ref}); Wb = OPS.matrix_table(b)["dec0.fc2"]
    assert torch.allclose(Wb[:, 5], R[:, 5]) and torch.allclose(Wb[:, 7], W0[:, 7])

def test_raw_view_toggle_hides_values_but_keeps_summaries():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent_tools")); import read_tools as RT, policy as P; m = P.NavPolicy()
    os.environ["HARNESS_RAW_VIEW"] = "0"
    try:
        r = RT.read_weights(m, "enc0.qkv", "0:2", "0:3"); assert "values" not in r and "DISABLED" in r["raw_view"] and "rms" in r; assert "DISABLED" in RT.qkv_view(m, "enc_self", 0, 0, dump=True)["raw"]
    finally: os.environ.pop("HARNESS_RAW_VIEW")
    assert "values" in RT.read_weights(m, "enc0.qkv", "0:2", "0:3")
