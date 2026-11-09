import os, sys, copy, random, numpy as np, torch, torch.nn as nn, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import ops as OPS, structure as S, patching as P, circuit as C, acclimate as A, llm_ops, prompt_pairs as PP

V, N = 8, 5                       # vocab, distinct tokens per sequence (small enough that a 2-layer model learns induction in ~2000 steps)


def make_batch(n, rng):
    seqs, q, tgt, wrong, cor = [], [], [], [], []
    for _ in range(n):
        t = rng.permutation(V); seen, unused = t[:N], t[N:]; j = rng.integers(0, N - 1); seq = list(seen) + [seen[j]]; c = list(seen) + [int(rng.choice(unused))]
        w = int(rng.choice([x for x in seen if x != seen[j + 1]])); seqs.append(seq); cor.append(c); tgt.append(int(seen[j + 1])); wrong.append(w)
    return torch.tensor(seqs), torch.tensor(cor), torch.tensor(tgt), torch.tensor(wrong)


def accuracy(model, rng, n=400):
    x, _, y, _ = make_batch(n, rng)
    with torch.no_grad(): out = model(x); out = out.logits if hasattr(out, "logits") else out
    return float((out[:, -1].argmax(-1) == y).float().mean())


def train_induction(model, steps=2000, seed=0):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(model.parameters(), 2e-3)
    for s in range(steps):
        x, _, y, _ = make_batch(128, rng); out = model(x); out = out.logits if hasattr(out, "logits") else out
        loss = nn.functional.cross_entropy(out[:, -1], y); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@pytest.fixture(scope="module")
def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0); m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=64, n_head=4, vocab_size=V, n_positions=16, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)); cache = os.environ.get("LAB10_CACHE")
    if cache and os.path.exists(cache): m.load_state_dict(torch.load(cache)); m.eval()
    else:
        train_induction(m)
        if cache: torch.save(m.state_dict(), cache)
    acc = accuracy(m, np.random.default_rng(99)); assert acc > 0.8, f"toy induction model did not train (acc {acc})"; return m


def pairs(rng, n=16):
    x, c, y, w = make_batch(n, rng); return x, c, P.logit_diff(y, w)


def test_acclimation_passes_on_trained_gpt2_and_enables_every_tool(gpt2):
    clean, corrupt, mf = pairs(np.random.default_rng(1)); rep = A.acclimate(gpt2, clean, corrupt, mf)
    assert rep["n_fail"] == 0, rep["rows"]; assert {"scale_head", "scale_neurons", "head_temperature", "edge_circuit", "node_circuit_exact", "matrix_ops"} <= set(rep["verified_tools"])


def test_circuit_finder_recovers_the_induction_circuit_and_editing_it_beats_random_heads(gpt2):
    tm = S.discover(gpt2); clean, corrupt, mf = pairs(np.random.default_rng(2), 32); ns = P.exact_node_scores(gpt2, tm, clean, corrupt, mf); agg = C.aggregate([ns])
    assert float(ns["clean_metric"].mean()) > float(ns["corrupt_metric"].mean()) + 1.0            # the behaviour exists and the corruption removes it
    circ = C.select({k: v for k, v in agg.items() if k[0] == "head"}, top_frac=0.9); assert len(circ) >= 2
    f = C.faithfulness(gpt2, tm, clean, corrupt, mf, circ); assert f["ablate_drop"] > 0.5, f
    top = [(n["layer"], n["index"]) for n in circ[:2]]; heads = [(l, h) for l in range(2) for h in range(4)]; rng = np.random.default_rng(5); test_acc = lambda m: accuracy(m, np.random.default_rng(7))
    llm_ops.enable()
    try:
        ed = gpt2
        for l, h in top: ed = OPS.apply("scale_head", {"layer": l, "head": h, "gamma": 0.0}, ed)
        acc_circuit = test_acc(ed); base = test_acc(gpt2)
        others = [x for x in heads if x not in top]; acc_other = float(np.mean([test_acc(OPS.apply("scale_head", {"layer": l, "head": h, "gamma": 0.0}, OPS.apply("scale_head", {"layer": l2, "head": h2, "gamma": 0.0}, gpt2))) for (l, h), (l2, h2) in zip(others[0::2], others[1::2])]))     # random PAIRS of non-circuit heads
    finally: llm_ops.disable()
    assert acc_circuit < base - 0.2 and acc_circuit < acc_other, (base, acc_circuit, acc_other)       # editing the FOUND circuit hurts the behaviour more than editing pairs of other heads


def test_first_order_attribution_agrees_with_exact_patching_on_trained_model(gpt2):
    tm = S.discover(gpt2); clean, corrupt, mf = pairs(np.random.default_rng(3), 32); ns = P.node_scores(gpt2, tm, clean, corrupt, mf); ca = P.corrupt_acts_of(gpt2, tm, corrupt)
    flat = sorted(((float(v[:, i].mean()), k, i) for k, v in ns["effect"].items() if k[0] == "head" for i in range(v.shape[1])), key=lambda r: -abs(r[0])); ex = [float(P.exact_node_effect(gpt2, tm, clean, ca, k, i, mf).mean()) for _, k, i in flat]
    # honest expectation: first-order attribution gets the SIGN and rough ordering of strong nodes but under-estimates saturated heads, so it is only used to SHORTLIST units; heads are ranked by exact patching
    assert A._spearman([f[0] for f in flat], ex) > 0.3 and all((a[0] > 0) == (b > 0) for a, b in zip(flat[:1], ex[:1])) and min(ex) < -1.0


def test_circuit_overlap_and_edit_proposals():
    a = [{"kind": "head", "layer": 1, "index": 0}, {"kind": "head", "layer": 0, "index": 1}]; b = [{"kind": "head", "layer": 1, "index": 0}, {"kind": "neuron", "layer": 1, "index": 5}]
    o = C.overlap({"A": a, "B": b}); assert o["core_shared_by_all"] == [("head", 1, 0)] and o["specific"]["A"] == [("head", 0, 1)] and o["jaccard"]["A"]["B"] == round(1 / 3, 3)
    props = C.edit_proposals(o["specific"]["B"] + o["specific"]["A"]); assert {"op": "scale_head", "cfg": {"layer": 0, "head": 1, "gamma": 0.0}} in props and any(p["op"] == "scale_neurons" for p in props)


def test_pair_quality_check_rejects_bad_pairs(gpt2):
    x, c, y, w = make_batch(1, np.random.default_rng(4)); good = {"name": "good", "clean": x[0], "corrupt": c[0], "correct": int(y[0]), "wrong": int(w[0])}
    bad_len = {**good, "name": "len", "corrupt": c[0][:-1]}; same = {**good, "name": "same", "wrong": int(y[0])}; flat = {**good, "name": "flat", "corrupt": x[0]}
    keep, rep = PP.check(gpt2, [good, bad_len, same, flat]); r = {x_["name"]: x_ for x_ in rep}
    assert r["good"]["ok"] and not r["len"]["ok"] and not r["same"]["ok"] and not r["flat"]["ok"] and [p["name"] for p in keep] == ["good"]


def test_llama_family_with_gqa_acclimates():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0); m = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=V, max_position_embeddings=16)).eval()
    clean, corrupt, mf = pairs(np.random.default_rng(1), 8); rep = A.acclimate(m, clean, corrupt, mf); tm = S.discover(m)
    assert tm.family == "llama-like" and tm.n_kv_heads == 2 and tm.n_heads == 4 and rep["n_fail"] == 0, rep["rows"]; assert {"scale_head", "scale_neurons", "head_temperature", "edge_circuit"} <= set(rep["verified_tools"])


def test_plain_mlp_50_neurons_acclimates():
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(10, 50), nn.GELU(), nn.Linear(50, 3)).eval(); clean, corrupt = torch.randn(16, 10), torch.randn(16, 10)
    rep = A.acclimate(m, clean, corrupt, lambda o: o[:, 0] - o[:, 1]); assert rep["n_fail"] == 0, rep["rows"]; assert {"scale_neurons", "prune_neurons", "node_circuit_exact", "matrix_ops"} <= set(rep["verified_tools"]) and "scale_head" not in rep["verified_tools"]


def test_unknown_architecture_fails_loudly_not_silently():
    m = nn.Sequential(nn.Conv1d(1, 4, 3), nn.ReLU(), nn.Conv1d(4, 4, 3))
    with pytest.raises(S.Unsupported): S.discover(m)


# ---------------------------------------------------------------- model maps for architectures with no built-in family
def _rand_pairs(n=6): return pairs(np.random.default_rng(1), n)


def test_propose_spec_reproduces_builtin_maps_for_gpt2_and_llama():
    import spec_map as SM
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0); g = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=V, n_positions=16)).eval(); l = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=V)).eval()
    for m in (g, l):
        spec, warns = SM.propose_spec(m); a, b = SM.from_spec(m, spec), S.discover(m)
        assert (a.n_layers, a.n_heads, a.n_kv_heads, a.head_dim, a.d_model, a.d_ff) == (b.n_layers, b.n_heads, b.n_kv_heads, b.head_dim, b.d_model, b.d_ff)
        for x, y in zip(a.blocks, b.blocks): assert x.o is y.o and x.down is y.down and x.attn_norm is y.attn_norm and x.mlp_norm is y.mlp_norm and x.q is y.q and [u[0] for u in x.ups] == [u[0] for u in y.ups]


def test_auto_proposed_map_for_an_architecture_without_a_builtin_family_acclimates_and_edits_correctly():
    import spec_map as SM
    from transformers import OPTConfig, OPTForCausalLM
    torch.manual_seed(0); m = OPTForCausalLM(OPTConfig(hidden_size=32, ffn_dim=64, num_hidden_layers=2, num_attention_heads=4, vocab_size=V, max_position_embeddings=16, word_embed_proj_dim=32)).eval()
    with pytest.raises(S.Unsupported): S.discover(m)                                   # no built-in family: auto-detection must refuse...
    spec, warns = SM.propose_spec(m); assert spec["block"]["o"].endswith("out_proj") and spec["block"]["down"] == "fc2" and spec["q_layout"] == "contiguous"      # ...but the proposer builds a map from shapes
    clean, corrupt, mf = _rand_pairs(); rep = A.acclimate(m, clean, corrupt, mf, spec=spec); assert rep["n_fail"] == 0, rep["rows"]
    assert {"scale_head", "scale_neurons", "head_temperature", "edge_circuit", "node_circuit_exact", "matrix_ops"} <= set(rep["verified_tools"])


def test_fused_interleaved_and_parallel_residual_architecture_is_downgraded_not_trusted():
    import spec_map as SM
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(0); m = GPTNeoXForCausalLM(GPTNeoXConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, vocab_size=V, max_position_embeddings=16, use_parallel_residual=True)).eval()
    spec, warns = SM.propose_spec(m); assert spec["q_layout"] == "unknown" and warns
    clean, corrupt, mf = _rand_pairs(); rep = A.acclimate(m, clean, corrupt, mf, spec=spec); v = set(rep["verified_tools"]); assert rep["n_fail"] == 0, rep["rows"]
    assert {"scale_head", "scale_neurons", "node_circuit_exact", "matrix_ops"} <= v and "head_temperature" not in v and "edge_circuit" not in v           # what cannot be proven is switched off


def test_proposer_refuses_post_norm_or_unknown_structures():
    import spec_map as SM
    with pytest.raises(S.Unsupported): SM.propose_spec(nn.Sequential(nn.Conv1d(1, 4, 3), nn.ReLU()))
