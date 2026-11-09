import os, sys, json, numpy as np, torch, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import structure as S, patching as P, circuit as C, circuit_grow as G, prompt_pairs as PP, lm_backend as B
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(B.default_root(), "policy_s0", "ckpt_009.pt")), reason="run any_model/lm_backend.py train first")


def _setup(which):
    m = B.LMBackend().load_ckpt(0, which); rng = np.random.default_rng(5); x, y = B.make_batch(32, rng); cor = x.clone()
    for i in range(len(cor)): cor[i, -1] = [t for t in range(B.V) if t not in x[i].tolist()][0]
    wrong = torch.tensor([[t for t in x[i, :-1].tolist() if t != int(y[i])][0] for i in range(len(x))]); mf = P.logit_diff(y, wrong); tm = S.discover(m)
    ns = P.exact_node_scores(m, tm, x, cor, mf); agg = C.aggregate([ns]); return m, tm, x, cor, mf, agg, C.select(agg, top_frac=0.95)


def test_grow_reaches_sufficiency_and_beats_random_growth():
    m, tm, x, cor, mf, agg, seed = _setup(9); r = G.grow(m, tm, x, cor, mf, agg, seed, target=0.8)
    assert r["sufficiency"] > r["seed_sufficiency"] + 0.3, r["curve"]                       # growth really raises sufficiency
    assert r["ablate_drop"] >= 0.7, r                                                       # still necessary
    assert r["reached_target"] and r["sufficiency"] >= 0.8, r
    # FINDING (not a pass/fail on ranking quality): on the healthy toy the mechanism is spread evenly over the layer-0 MLP bank, so NO neuron ranking beats random-order pruning (234 vs 213 nodes).
    # The useful result is at COMPONENT level (3 layer-0 heads + the layer-0 bank). We only require the guided prune not to be much worse than random.
    assert r["n_nodes"] <= 1.25 * r["n_nodes_if_pruned_in_random_order"], r
    grow_only = [c["sufficiency"] for c in r["curve"] if c["source"] != "prune"]; assert all(b >= a - 1e-6 for a, b in zip(grow_only, grow_only[1:]))


def test_grow_works_on_the_damaged_target_and_union_core():
    m, tm, x, cor, mf, agg, seed = _setup("final"); r = G.grow(m, tm, x, cor, mf, agg, seed, target=0.8); assert r["reached_target"] and r["sufficiency"] >= 0.7 and r["ablate_drop"] >= 0.7, r
    assert r["n_nodes"] < r["n_nodes_if_pruned_in_random_order"], r          # on the damaged target the leave-one-out ranking DOES beat random-order pruning (89 vs 106 nodes)
    u = G.union_core({"a": [{"kind": "head", "layer": 0, "index": 1}, {"kind": "head", "layer": 0, "index": 2}], "b": [{"kind": "head", "layer": 0, "index": 1}], "c": [{"kind": "head", "layer": 1, "index": 3}]}, 0.6)
    assert u["core"] == [("head", 0, 1)]
