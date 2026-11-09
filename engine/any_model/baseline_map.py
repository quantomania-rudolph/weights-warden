"""BASELINE ACTIVITY MAP: what lights up on NORMAL data, and which units move together.  A starting map of the model before any circuit is chosen.
  record()      activations of every head (output norm) and MLP unit on a sample of normal prompts (research data only; never CANARY/HOLD) -> table per layer
  stats()       per unit: mean, std, firing rate, dead (never fires), always-on, constant.  Dead/constant units are useless as circuit members; always-on ones are 'background'.
  coactivation() correlation between units across prompts + hierarchical clusters (units that behave as one group).
  neighbors()   units that co-activate with a given circuit = candidate 'connective tissue' (ranked by max |corr| to any circuit node).
  verify_neighbors()  the candidates are only PROPOSALS: add the top-k to the circuit and measure the sufficiency gain (circuit_grow._Eval) against the SAME number of RANDOM units.
HONEST LIMITS: activity is correlational, not causal (a busy unit may do nothing); heads are summarised by their output NORM, which hides direction; correlations are computed at one position
(default: the last / decision position) over N prompts, so N must be >> units shown.  Use the map to propose, patching to decide."""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import patching as P


def record(model, tm, tokens, where="last", batch=256):
    """{(kind,layer): (N, n_units) float tensor}.  kind 'head' = output norm of the head, 'neuron'/'unit' = the activation.  where: 'last' (decision position) or 'all' (every position)."""
    acc = {}
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            rec, _ = P._run(model, tm, tokens[i:i + batch], False)
            for key in rec.acts:
                a = rec.A(key).detach().float()
                if key[0] == "head": a = a.reshape(*a.shape[:-1], tm.n_heads, tm.head_dim).norm(dim=-1)            # (B,T,H) head output norm
                a = a[:, -1] if (where == "last" and a.ndim == 3) else (a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a)
                acc.setdefault(key, []).append(a.cpu())
    return {k: torch.cat(v) for k, v in acc.items()}


def stats(table, rel_thr=0.05):
    """per-layer unit statistics.  'active' = |a| > rel_thr * (99th percentile of |a| over the layer)."""
    out = {}
    for key, a in table.items():
        thr = rel_thr * float(torch.quantile(a.abs().flatten()[:200000], 0.99)) + 1e-12; rate = (a.abs() > thr).float().mean(0); std = a.std(0)
        is_head = key[0] == "head"                                                    # a head's output NORM is positive and continuous: 'always on' / 'dead' by threshold are meaningless, so they are reported False for heads
        out[key] = {"mean": a.mean(0), "std": std, "rate": rate, "dead": (rate < 1e-3) & (not is_head), "always_on": (rate > 0.99) & (not is_head), "constant": std < 1e-6, "thr": thr}
    return out


def summary(st):
    return {f"{k[0]}.{k[1]}": {"units": int(len(s["mean"])), "dead": int(s["dead"].sum()), "always_on": int(s["always_on"].sum()), "constant": int(s["constant"].sum()),
                               "median_rate": round(float(s["rate"].median()), 3), "top_mean": [int(i) for i in s["mean"].abs().argsort(descending=True)[:5]]} for k, s in st.items()}


def coactivation(table, st, min_corr=0.6, max_units=2000):
    """correlation over the non-constant units + average-linkage clusters at distance 1-min_corr.  returns {'nodes','corr','cluster'}  (nodes are (kind,layer,idx))."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    nodes, cols = [], []
    for key, a in table.items():
        ok = ~st[key]["constant"]
        for i in torch.nonzero(ok).flatten().tolist(): nodes.append((key[0], key[1], i)); cols.append(a[:, i])
    X = torch.stack(cols, 1); var = X.std(0)
    if X.shape[1] > max_units: keep = var.argsort(descending=True)[:max_units].tolist(); nodes = [nodes[j] for j in keep]; X = X[:, keep]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); C = (X.T @ X / (len(X) - 1)).clamp(-1, 1).numpy(); np.fill_diagonal(C, 1.0)
    D = 1 - np.abs(C); np.fill_diagonal(D, 0); lab = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - min_corr, criterion="distance")
    return {"nodes": nodes, "corr": C, "cluster": lab, "n_samples": int(len(X))}


def clusters_summary(co, top=8):
    from collections import defaultdict
    g = defaultdict(list)
    for n, c in zip(co["nodes"], co["cluster"]): g[int(c)].append(n)
    big = sorted(g.values(), key=len, reverse=True)[:top]
    return [{"size": len(m), "layers": sorted({(k, l) for k, l, _ in m}), "members_head": [list(x) for x in m[:6]]} for m in big]


def neighbors(co, circuit, min_corr=0.5, top=40):
    """units NOT in the circuit ranked by max |corr| with any circuit node (only circuit nodes present in the map count)."""
    idx = {n: j for j, n in enumerate(co["nodes"])}; cj = [idx[n] for n in circuit if n in idx]
    if not cj: return []
    score = np.abs(co["corr"][:, cj]).max(1); inc = set(circuit)
    rows = [(float(score[j]), co["nodes"][j]) for j in np.argsort(-score) if co["nodes"][j] not in inc and score[j] >= min_corr]
    return [{"node": list(n), "max_abs_corr": round(s, 3)} for s, n in rows[:top]]


def verify_neighbors(ev, circuit, cand_nodes, n_random=8, seed=0):
    """sufficiency gain from adding the candidate group vs the SAME number of random units (never in circuit). ev = circuit_grow._Eval."""
    rng = np.random.default_rng(seed); circuit = list(circuit); base = ev.sufficiency(circuit); k = len(cand_nodes)
    pool = [(kk[0], kk[1], i) for kk, idx in ev.full.items() if kk[0] in ("head", "neuron", "unit") for i in idx if (kk[0], kk[1], i) not in set(circuit)]
    g = ev.sufficiency(circuit + list(cand_nodes)) - base
    rg = [ev.sufficiency(circuit + [pool[j] for j in rng.choice(len(pool), min(k, len(pool)), replace=False)]) - base for _ in range(n_random)]
    return {"k": k, "base_sufficiency": round(base, 3), "gain_coactive": round(g, 3), "gain_random_mean": round(float(np.mean(rg)), 3), "gain_random_std": round(float(np.std(rg)), 3),
            "informative": bool(g > np.mean(rg) + 2 * np.std(rg) and g > 0.02)}


def coactive_groups(co, circuit, min_corr=0.5, ks=(8, 32)):
    """candidate GROUPS for circuit_grow.grow(extra_groups=...): the top-k co-active neighbours of the circuit."""
    nb = [tuple(r["node"]) for r in neighbors(co, circuit, min_corr, top=max(ks))]
    return {f"coactive_top{k}": nb[:k] for k in ks if len(nb) >= min(k, 2)}
