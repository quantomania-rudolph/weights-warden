import os, sys, copy, numpy as np, torch, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import structure as S, baseline_map as BM, lm_backend as B
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(B.default_root(), "policy_s0", "ckpt_009.pt")), reason="run any_model/lm_backend.py train first")


def _load(which=9): return B.LMBackend().load_ckpt(0, which)
def _tokens(n=400): return B.LMBackend().probe_obs("ID", n)


def test_record_matches_independent_hooks():
    m = _load(); tm = S.discover(m); x = _tokens(64); tab = BM.record(m, tm, x, "last"); cap = {}
    hs = [m.transformer.h[0].mlp.act.register_forward_hook(lambda mod, i, o: cap.__setitem__("n", o.detach())), m.transformer.h[0].attn.c_proj.register_forward_pre_hook(lambda mod, a: cap.__setitem__("h", a[0].detach()))]
    with torch.no_grad(): m(x)
    for h in hs: h.remove()
    assert torch.allclose(tab[("neuron", 0)], cap["n"][:, -1], atol=1e-6)
    assert torch.allclose(tab[("head", 0)], cap["h"][:, -1].reshape(64, 4, 16).norm(dim=-1), atol=1e-5)


def test_stats_flag_planted_dead_always_on_and_constant_units_without_mutating_model():
    m = _load(); sd0 = {k: v.clone() for k, v in m.state_dict().items()}; m2 = copy.deepcopy(m)
    with torch.no_grad(): b = m2.transformer.h[0].mlp.c_fc.bias; b[3] = -60.0; b[4] = 60.0                    # 3: never fires, 4: always on
    tm = S.discover(m2); tab = BM.record(m2, tm, _tokens(), "last"); st = BM.stats(tab)[("neuron", 0)]
    assert bool(st["dead"][3]) and bool(st["always_on"][4]) and not bool(st["dead"][4]); assert not bool(st["always_on"][3])
    assert all(torch.equal(sd0[k], v) for k, v in m.state_dict().items())                                   # the original model was never touched
    s1 = BM.record(m, S.discover(m), _tokens(), "last"); s2 = BM.record(m, S.discover(m), _tokens(), "last"); assert all(torch.equal(s1[k], s2[k]) for k in s1)       # deterministic


def test_duplicate_neuron_clusters_together_and_is_the_top_neighbor():
    m = copy.deepcopy(_load()); w = m.transformer.h[0].mlp.c_fc
    live = None
    with torch.no_grad():
        tab0 = BM.record(m, S.discover(m), _tokens(), "last"); st0 = BM.stats(tab0)[("neuron", 0)]; live = [int(i) for i in torch.nonzero(~st0["dead"] & ~st0["always_on"]).flatten()][:2]
        i, j = live; w.weight[:, j] = w.weight[:, i]; w.bias[j] = w.bias[i]                                      # j becomes an exact copy of i (Conv1D: columns are output units)
    tm = S.discover(m); tab = BM.record(m, tm, _tokens(), "last"); st = BM.stats(tab); co = BM.coactivation(tab, st, min_corr=0.9)
    ni, nj = co["nodes"].index(("neuron", 0, i)), co["nodes"].index(("neuron", 0, j)); assert co["corr"][ni, nj] > 0.999 and co["cluster"][ni] == co["cluster"][nj]
    nb = BM.neighbors(co, [("neuron", 0, i)], min_corr=0.9); assert nb and nb[0]["node"] == ["neuron", 0, j] and nb[0]["max_abs_corr"] > 0.999
    assert ("neuron", 0, i) not in [tuple(r["node"]) for r in nb]                                            # circuit nodes are never their own neighbours


def test_verify_neighbors_compares_against_equal_sized_random_group():
    import test_grow as T, circuit_grow as G
    m, tm, x, cor, mf, agg, seed = T._setup("final"); ev = G._Eval(m, tm, x, cor, mf); circ = [(n["kind"], n["layer"], n["index"]) for n in seed]
    tab = BM.record(m, tm, _tokens(), "last"); co = BM.coactivation(tab, BM.stats(tab)); nb = [tuple(r["node"]) for r in BM.neighbors(co, circ, 0.3, 16)]
    v = BM.verify_neighbors(ev, circ, nb, n_random=4); assert v["k"] == len(nb) and {"gain_coactive", "gain_random_mean", "gain_random_std", "informative"} <= set(v)
    g = BM.coactive_groups(co, circ, 0.3, (4, 8)); assert all(len(v_) <= 8 for v_ in g.values()) and not any(set(v_) & set(circ) for v_ in g.values())
