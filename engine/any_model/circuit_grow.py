"""GROW a circuit until it is SUFFICIENT, not just necessary - and merge circuits from several behaviours.
WHY: select() keeps only nodes that reliably matter (necessary): corrupting them kills the behaviour, but keeping ONLY them clean recovers little (`keep_only_recovered`, ~0.06 on the toy).
The behaviour is spread over more parts.  grow() adds nodes until keeping only the circuit clean recovers `target` of the clean-vs-corrupt gap.
HOW (cheap: exact patching of a batch of short prompts, no training):
  1. seed = the select()ed circuit.
  2. candidates = (a) upstream WRITERS of the seed's readers, ranked by EXACT direct-path edge patching (first-order edge scores only shortlist; they are unreliable for whole edges),
     (b) the highest-importance remaining heads/neurons from the exact node scores (shortlist `pool`).
  3. greedy: each round add the candidate whose addition raises sufficiency most; stop at `target`, when nothing helps, or at `max_nodes`.
  4. report the curve, the source of every added node ('edge' or 'importance'), final sufficiency and ablate_drop, plus a RANDOM-growth control of the same size.
union_core(): nodes appearing in >= min_frac of several behaviours' circuits = the macro/shared circuit; the rest is behaviour-specific.
Honest limits: sufficiency here is w.r.t. corrupt-activation patching on the supplied pairs; a bigger circuit shares more parts with other behaviours, so edit it only after the side-effect gates (ID drop, canary)."""
import os, sys, random, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import patching as P, circuit as C


class _Eval:
    def __init__(self, model, tm, clean, corrupt, mf):
        self.m, self.tm, self.clean, self.mf = model, tm, clean, mf; self.ca = P.corrupt_acts_of(model, tm, corrupt)
        with torch.no_grad(): self.c, self.k = float(mf(model(clean)).mean()), float(mf(model(corrupt)).mean())
        self.gap = (self.c - self.k) if abs(self.c - self.k) > 1e-9 else 1e-9
        self.full = {key: list(range(a.shape[-1] // tm.head_dim if key[0] == "head" else a.shape[-1])) for key, a in self.ca.items()}

    def _nd(self, nodes):
        d = {}
        for kind, l, i in nodes: d.setdefault((kind, l), []).append(i)
        return d

    def sufficiency(self, nodes):
        """fraction of the clean-vs-corrupt gap kept when EVERYTHING EXCEPT `nodes` gets corrupt activations."""
        keep = self._nd(nodes); rest = {k: [i for i in idx if i not in set(keep.get(k, []))] for k, idx in self.full.items()}
        ko = float(P.patched_metric(self.m, self.tm, self.clean, self.ca, {k: v for k, v in rest.items() if v}, self.mf).mean()); return (ko - self.k) / self.gap

    def ablate_drop(self, nodes):
        return (self.c - float(P.patched_metric(self.m, self.tm, self.clean, self.ca, self._nd(nodes), self.mf).mean())) / self.gap


def edge_candidates(model, tm, clean, corrupt, mf, seed, agg, max_writers=6, shortlist=12, min_rel=0.2):
    """upstream writers of the seed nodes' readers, verified by EXACT direct-path patching. returns [(node, exact_effect)] sorted by |effect|."""
    readers = {("r", "attn_in", l) if k == "head" else ("r", "mlp_in", l) for k, l, _ in seed}
    es = P.edge_scores(model, tm, clean, corrupt, mf); rows = []
    for (wk, rk), v in es.items():
        if rk not in readers or wk[0] == "embed": continue
        v = v.detach(); v = v.reshape(v.shape[0], -1) if v.ndim > 1 else v[:, None]
        for h in range(v.shape[1]): rows.append((abs(float(v[:, h].mean())), wk, rk, h))
    rows.sort(key=lambda r: -r[0]); out = []
    for _, wk, rk, h in rows[:shortlist]:
        ex = float(P.exact_edge_effect(model, tm, clean, corrupt, wk, rk, mf, head=h if wk[0] == "head" else None).mean())
        if wk[0] == "head": out.append((("head", wk[1], h), ex))
        else:                                                                              # an MLP writer: take its most important neurons
            imp = agg.get(("neuron", wk[1])); 
            if imp is not None: out += [(("neuron", wk[1], int(i)), ex) for i in torch.argsort(-imp["importance"])[:3].tolist()]
    out.sort(key=lambda r: -abs(r[1])); top = abs(out[0][1]) if out else 0
    return [(n, e) for n, e in out if abs(e) >= min_rel * top][:max_writers]


def grow(model, tm, clean, corrupt, mf, agg, seed, target=0.8, max_nodes=64, pool=40, use_edges=True, seed_rng=0, prune=True, extra_groups=None):
    """greedy growth over single nodes AND whole components (an MLP bank / all heads of a layer): behaviours are often spread over many tiny neurons, so single-node greedy is far too slow.
    After the target is reached, `prune` drops the least important members of any added bank in chunks while sufficiency stays >= target, so the final circuit is as small as the evidence allows."""
    ev = _Eval(model, tm, clean, corrupt, mf); cur = [(n["kind"], n["layer"], n["index"]) for n in seed]; src = {n: "seed" for n in cur}
    imp = lambda n: float(agg[(n[0], n[1])]["importance"][n[2]]) if (n[0], n[1]) in agg else 0.0
    ranked = sorted(((float(a["importance"][i]), (k[0], k[1], i)) for k, a in agg.items() for i in range(len(a["importance"])) if a["importance"][i] > 0), key=lambda r: -r[0])
    singles = []
    if use_edges and cur:
        for n, e in edge_candidates(model, tm, clean, corrupt, mf, cur, agg):
            if n not in src and n not in singles: singles.append(n); src[n] = "edge"
    for _, n in ranked:
        if len(singles) >= pool: break
        if n not in src and n not in singles: singles.append(n); src[n] = "importance"
    groups = {}                                                                   # whole components
    for (kind, l), idx in ev.full.items():
        if kind in ("neuron", "unit"): groups[f"{kind}_bank_{l}"] = [(kind, l, i) for i in idx]
        elif kind == "head": groups[f"heads_layer_{l}"] = [(kind, l, i) for i in idx]
    for gname, gn in (extra_groups or {}).items(): groups[gname] = [tuple(n) for n in gn]             # e.g. co-active neighbours from baseline_map.coactive_groups
    suff = ev.sufficiency(cur); curve = [{"n_nodes": len(cur), "sufficiency": round(suff, 3), "added": None, "source": "seed"}]
    while suff < target and len(cur) < max_nodes + 512:
        opts = [([n], src[n]) for n in singles if n not in cur] + [([n for n in g if n not in cur], ("coactive" if gname.startswith("coactive") else "bank")) for gname, g in groups.items() if any(n not in cur for n in g)]
        if not opts: break
        scored = [(ev.sufficiency(cur + o), o, k) for o, k in opts]; best_s, best, kind = max(scored, key=lambda r: r[0])
        if best_s <= suff + 1e-3: break
        for n in best: src[n] = kind
        cur += best; suff = best_s; curve.append({"n_nodes": len(cur), "sufficiency": round(suff, 3), "added": f"{len(best)} node(s): {best[0] if len(best) == 1 else best[0][0] + ' group layer ' + str(best[0][1])}", "source": kind})
        singles = [n for n in singles if n not in cur]
    n_before_prune = len(cur); seed_set = {(n["kind"], n["layer"], n["index"]) for n in seed}; grown = list(cur)

    def shrink(nodes, order):
        for chunk in (16, 4, 1):                                                   # coarse-to-fine: drop members while sufficiency stays >= target
            for c0 in range(0, len(order), chunk):
                drop = set(order[c0:c0 + chunk]) & set(nodes); trial = [n for n in nodes if n not in drop]
                if drop and ev.sufficiency(trial) >= target: nodes = trial
        return nodes
    rng = random.Random(seed_rng); rand_pruned = None
    if prune and suff >= target:
        removable = [n for n in cur if n not in seed_set]; base_s = ev.sufficiency(cur)
        loo = {n: base_s - ev.sufficiency([m_ for m_ in cur if m_ != n]) for n in removable}                  # leave-one-out cost: sufficiency LOST when this node alone is removed from the grown set
        cur = shrink(cur, sorted(removable, key=lambda n: loo[n])); suff = ev.sufficiency(cur)
        curve.append({"n_nodes": len(cur), "sufficiency": round(suff, 3), "added": f"pruned {n_before_prune - len(cur)} nodes (cheapest to remove first, by leave-one-out sufficiency cost)", "source": "prune"})
        ro = list(removable); rng.shuffle(ro); rand_pruned = len(shrink(list(grown), ro))             # CONTROL: same procedure, random removal order
    allnodes = [(k[0], k[1], i) for k, idx in ev.full.items() if k[0] in ("head", "neuron", "unit") for i in idx]; n_added = len(cur) - len(seed)
    rnd = [(n["kind"], n["layer"], n["index"]) for n in seed] + (rng.sample([n for n in allnodes if n not in {(q['kind'], q['layer'], q['index']) for q in seed}], n_added) if n_added > 0 else [])
    return {"nodes": [{"kind": k, "layer": l, "index": i, "source": src.get((k, l, i), "bank")} for k, l, i in cur], "curve": curve, "sufficiency": round(suff, 3), "ablate_drop": round(ev.ablate_drop(cur), 3),
            "seed_sufficiency": curve[0]["sufficiency"], "random_growth_control_sufficiency": round(ev.sufficiency(rnd), 3), "n_nodes_if_pruned_in_random_order": rand_pruned, "n_nodes": len(cur), "total_nodes_in_model": len(allnodes), "reached_target": suff >= target, "target": target}


def union_core(circuits, min_frac=0.6):
    """circuits {name: [{kind,layer,index}]} -> nodes present in >= min_frac of them (macro circuit) + per-node counts."""
    cnt = {}
    for c in circuits.values():
        for n in {(x["kind"], x["layer"], x["index"]) for x in c}: cnt[n] = cnt.get(n, 0) + 1
    need = min_frac * len(circuits); return {"core": sorted(n for n, c in cnt.items() if c >= need), "counts": {f"{k}.{l}.{i}": c for (k, l, i), c in sorted(cnt.items(), key=lambda r: -r[1])}}
