"""From per-pair attribution scores to CIRCUITS: aggregate over prompt pairs, keep nodes that matter AND agree in sign across examples, measure FAITHFULNESS with exact patching,
compare circuits across behaviours (shared core vs behaviour-specific), and turn the behaviour-specific part into edit proposals ("edit only those")."""
import torch, patching as P


def aggregate(score_list):
    """score_list: list of node_scores() dicts (one per pair-batch). importance = -effect (positive = needed for the clean behaviour). returns {key: {'importance': (n,), 'consistency': (n,)}}"""
    out = {}
    for k in score_list[0]["effect"]:
        imp = -torch.cat([s["effect"][k] for s in score_list], 0); mean = imp.mean(0)
        n = imp.shape[0]; t = mean / (imp.std(0, unbiased=True) / n ** 0.5 + 1e-9) if n > 1 else torch.zeros_like(mean)
        out[k] = {"importance": mean, "consistency": (torch.sign(imp) == torch.sign(mean)).float().mean(0), "tstat": t}
    return out


def select(agg, top_frac=0.8, min_consistency=0.5, min_tstat=2.0, max_nodes=64):
    """smallest set of reliably-important nodes (mean importance > 0 with t-statistic >= min_tstat over examples, sign agreeing in >= min_consistency of them) covering `top_frac` of the total positive importance mass (capped at max_nodes).
    (t-statistic, not sign agreement alone: a head can matter a lot on average while its per-example effect varies, e.g. depending on where the matched token sits.)"""
    rows = [(float(a["importance"][i]), k, i, float(a["consistency"][i])) for k, a in agg.items() for i in range(len(a["importance"])) if a["importance"][i] > 0 and a["consistency"][i] >= min_consistency and a["tstat"][i] >= min_tstat]
    rows.sort(key=lambda r: -r[0]); total = sum(r[0] for r in rows) or 1.0; sel, acc = [], 0.0
    for r in rows:
        if acc / total >= top_frac or len(sel) >= max_nodes: break
        sel.append({"kind": r[1][0], "layer": r[1][1], "index": r[2], "importance": round(r[0], 4), "consistency": round(r[3], 2)}); acc += r[0]
    return sel


def nodes_dict(circuit):
    d = {}
    for n in circuit: d.setdefault((n["kind"], n["layer"]), []).append(n["index"])
    return d


@torch.no_grad()
def faithfulness(model, tm, clean, corrupt, metric_fn, circuit):
    """exact patching. ablate_drop = fraction of the clean-vs-corrupt metric gap LOST when the circuit's nodes get corrupt activations (want high);
    keep_only_recovered = fraction of the gap kept when EVERYTHING ELSE gets corrupt activations (want high)."""
    ca = P.corrupt_acts_of(model, tm, corrupt); c, k = metric_fn(model(clean)).mean(), metric_fn(model(corrupt)).mean(); gap = float(c - k) if abs(float(c - k)) > 1e-9 else 1e-9
    nd = nodes_dict(circuit); ab = P.patched_metric(model, tm, clean, ca, nd, metric_fn).mean()
    full = {key: list(range(a.shape[-1] // tm.head_dim if key[0] == "head" else a.shape[-1])) for key, a in ca.items()} if tm.kind == "transformer" else {key: list(range(a.shape[-1])) for key, a in ca.items()}
    rest = {key: [i for i in idx if i not in set(nd.get(key, []))] for key, idx in full.items()}; ko = P.patched_metric(model, tm, clean, ca, {k_: v for k_, v in rest.items() if v}, metric_fn).mean()
    return {"clean_metric": round(float(c), 3), "corrupt_metric": round(float(k), 3), "ablate_drop": round(float((c - ab) / gap), 3), "keep_only_recovered": round(float((ko - k) / gap), 3), "n_nodes": len(circuit)}


def overlap(circuits):
    """circuits: {behaviour: circuit list}. returns jaccard matrix, the core shared by ALL behaviours, and each behaviour's specific nodes."""
    sets = {n: {(x["kind"], x["layer"], x["index"]) for x in c} for n, c in circuits.items()}; names = list(sets)
    jac = {a: {b: round(len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])), 3) for b in names} for a in names}; core = set.intersection(*sets.values()) if sets else set()
    spec = {n: sorted(sets[n] - set.union(*(sets[m] for m in names if m != n))) if len(names) > 1 else sorted(sets[n]) for n in names}
    return {"jaccard": jac, "core_shared_by_all": sorted(core), "specific": spec}


def edit_proposals(nodes, kind_gamma=0.0, max_per_op=24):
    """turn a set of (kind, layer, index) nodes into charged-edit proposals for the op library (one op per head; neurons grouped per layer). Compose up to 4 with `composite`."""
    props, neu = [], {}
    for kind, layer, idx in nodes:
        if kind == "head": props.append({"op": "scale_head", "cfg": {"layer": layer, "head": idx, "gamma": kind_gamma}})
        elif kind in ("neuron", "unit"): neu.setdefault(layer, []).append(idx)
    for layer, ids in sorted(neu.items()): props.append({"op": "scale_neurons", "cfg": {"layer": layer, "neurons": sorted(ids)[:max_per_op], "gamma": kind_gamma}})
    return props
