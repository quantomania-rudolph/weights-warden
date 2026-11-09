"""nav_circuits circuit report for a Nav2D policy: memorisation-vs-generalisation CONTRAST at edge, node and neuron level, with validation numbers shown to the reader.
EDGES/NODES use EXACT patching (all ~109 edges cost ~110 forwards; first-order EAP was validated and FAILS at this granularity for whole-edge removal, see nav_circuits/eap_nav.py notes):
  A(e) = damage to the L1 fit of the TRAINED labels on TRAIN states when edge e is removed ; B(e) = damage to the L1 fit of EXPERT labels on held-out-layout (VAL_SEEN) states
  contrast = A - B  (large: needed to reproduce training labels, not needed for expert behaviour on new layouts => memorisation-like; B large too => shared/generalising)
NEURONS use first-order EAP (validated vs exact ablation, Spearman reported).
ORACLE columns (planted run only, for validating the tool, NOT shown to an analyst): node damage on planted vs clean samples.
usage: python nav_circuits/circuit_report.py RUNDIR OUTDIR [--oracle]"""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "core")); sys.path.insert(0, os.path.join(ROOT, "nav_planted"))
import eap_nav as EN, policy as P, planted_metrics as PM
from scipy.stats import spearmanr
torch.set_num_threads(1)


def node_edges(edges): 
    d = {}
    for w, r in edges: d.setdefault(w, []).append((w, r))
    return d


@torch.no_grad()
def node_damage(m, obs, y, nodes, ne):
    base = EN.metric(EN.run(m, obs)[0], y); return {n: (EN.metric(EN.run(m, obs, frozenset(ne[n]))[0], y) - base) for n in nodes}      # per-sample


def main(run, outdir, oracle):
    os.makedirs(outdir, exist_ok=True); m = P.load_policy(os.path.join(run, "final.pt")); rng = np.random.default_rng(0)
    obs, clean, planted, mask = PM.build(); idx = rng.choice(len(obs), 4000, replace=False); o, yt, yc, mk = torch.from_numpy(obs[idx]), torch.from_numpy(planted[idx]), torch.from_numpy(clean[idx]), mask[idx]
    import adapter; vo = adapter.probe_obs("VAL", 2000); vy = torch.as_tensor(np.asarray(adapter.backend().expert_chunk(vo.numpy())), dtype=torch.float32)
    _, c = EN.run(m, o[:8], record=True); edges = list(dict.fromkeys(c["edges"])); ne = node_edges(edges); nodes = list(ne)
    exA = EN.exact_edge_scores(m, o, yt, edges); exB = EN.exact_edge_scores(m, vo, vy, edges)
    ea, na, _ = EN.eap(m, o, yt); eb, nb, _ = EN.eap(m, vo, vy)
    # validation numbers an analyst can see
    keys = edges; v_edge = float(spearmanr([ea[k] - eb[k] for k in keys], [exA[k] - exB[k] for k in keys])[0])
    rows = np.random.default_rng(1).choice(1024, 40, replace=False); W = 256 if m.enc_layers[0].mlp.fc2.in_features == 256 else m.enc_layers[0].mlp.fc2.in_features
    blocks = list(na); exn = []
    for g in rows:
        b, j = blocks[g // W], g % W; kind, i = ("enc", int(b[1])) if b[0] == "E" else ("dec", int(b[1])); mod = (m.enc_layers if kind == "enc" else m.dec_layers)[i].mlp
        import copy; e = copy.deepcopy(m)
        with torch.no_grad(): (e.enc_layers if kind == "enc" else e.dec_layers)[i].mlp.fc2.weight[:, j] *= 0
        with torch.no_grad(): exn.append(float((EN.metric(e(o), yt).mean() - EN.metric(m(o), yt).mean()) - (EN.metric(e(vo), vy).mean() - EN.metric(m(vo), vy).mean())))
    v_neu = float(spearmanr([na[blocks[g // W]][g % W] - nb[blocks[g // W]][g % W] for g in rows], exn)[0])
    contrast = {f"{w}->{r}": (exA[(w, r)], exB[(w, r)], exA[(w, r)] - exB[(w, r)]) for (w, r) in edges}
    nodeA, nodeB = {}, {}
    dA = node_damage(m, o, yt, nodes, ne); dB = node_damage(m, vo, vy, nodes, ne)
    for n in nodes: nodeA[n], nodeB[n] = float(dA[n].mean()), float(dB[n].mean())
    nc = {n: nodeA[n] - nodeB[n] for n in nodes}
    res = {"validation": {"edge_first_order_vs_exact_spearman_contrast": v_edge, "neuron_first_order_vs_exact_spearman_contrast_n40": v_neu}, "node": {n: [nodeA[n], nodeB[n], nc[n]] for n in nodes}, "edge": contrast,
           "neuron_contrast": {b: (na[b] - nb[b]).tolist() for b in na}}
    if oracle:
        pm = np.where(mk)[0]; cm = np.where(~mk)[0]; dp = node_damage(m, o[pm], yt[pm], nodes, ne); dc = node_damage(m, o[cm], yc[cm], nodes, ne)
        orc = {n: float(dp[n].mean() - dc[n].mean()) for n in nodes}; res["oracle_node"] = orc
        res["validation"]["node_contrast_vs_oracle_spearman"] = float(spearmanr([nc[n] for n in nodes], [orc[n] for n in nodes])[0])
        t = lambda d, k: set(sorted(d, key=lambda n: -d[n])[:k]); res["validation"]["node_top8_overlap_with_oracle"] = len(t(nc, 8) & t(orc, 8)); res["validation"]["n_nodes"] = len(nodes)
    json.dump(res, open(os.path.join(outdir, "circuit.json"), "w"), indent=1)
    # digest (<=4000 chars)
    L = ["CIRCUIT CONTRAST (needed-for-training-labels minus needed-for-expert-behaviour-on-new-layouts; damage in L1 units x1000; higher contrast = more memorisation-like).",
         f"validation: first-order EAP vs exact on edges Spearman={v_edge:.2f} (low: do not trust first-order at edge level); neuron first-order vs exact Spearman={v_neu:.2f} (n=40)."]
    L.append("TOP NODES by contrast (node = a head or MLP block; removing it from every reader):  node | A(train) | B(held-out) | contrast")
    for n in sorted(nodes, key=lambda n: -nc[n])[:10]: L.append(f"  {n:8s} {1000*nodeA[n]:8.1f} {1000*nodeB[n]:8.1f} {1000*nc[n]:8.1f}")
    L.append("TOP EDGES by contrast:  writer->reader | A | B | contrast")
    for k, v in sorted(contrast.items(), key=lambda kv: -kv[1][2])[:12]: L.append(f"  {k:18s} {1000*v[0]:7.1f} {1000*v[1]:7.1f} {1000*v[2]:7.1f}")
    L.append("TOP NEURONS by first-order contrast (block, index): " + ", ".join(f"{b}:{j}" for b, j in sorted(((b, int(j)) for b in na for j in range(len(na[b]))), key=lambda t: -(na[t[0]][t[1]] - nb[t[0]][t[1]]))[:30]))
    L.append("naming: E{l}.h{h} = encoder layer l self-attn head h (op scale_head kind=enc_self); D{l}.sh{h} = decoder self-attn (dec_self); D{l}.xh{h} = decoder cross-attn (dec_cross); E/D{l}.mlp = MLP block (op scale_neurons kind=enc|dec).")
    open(os.path.join(outdir, "circuit_contrast.txt"), "w").write("\n".join(L)[:4000]); print(json.dumps(res["validation"])); print("\n".join(L)[:2500])


if __name__ == "__main__": main(sys.argv[1], sys.argv[2], "--oracle" in sys.argv)
