"""READ/ANALYSIS tools for the LLM-style (ModelMap) mode of the harness.  Reached through `HARNESS_MODE=llm python agent_tools/run.py tools RUN CMD ...` (or agent_tools/dispatch.py).
Every call is logged to RUN/tools.jsonl.  None reads CANARY/HOLD, none returns closed-loop success.  The model is the run's target (loop.target_model).

  inventory                                     every editable component (blocks, heads, MLP banks, matrices with guessed roles) + the op that edits it
  pairs_check --file pairs.json                 quality-check clean/corrupt prompt pairs (equal length, model prefers `correct` on clean, corruption really hurts)
  circuit --file pairs.json --name N [--top_frac F] [--edges]
                                                exact node patching over all heads + MLP units (the reliable ranking), t-statistic + sign-agreement selection, faithfulness
                                                (ablate_drop, keep_only_recovered); --edges adds first-order edge attribution checked against exact direct-path edge patching
  grow --file pairs.json --name N [--target 0.8]  grow a circuit until it is SUFFICIENT (whole banks + single nodes, then prune); reports the sufficiency curve and a random-order-prune control
  baseline --name N [--n 512 --where last --min_corr 0.6]   what lights up on NORMAL in-distribution prompts: per-layer dead/always-on units, co-activation clusters (saved under RUN/baseline)
  neighbors --name BASELINE --circuit C [--file pairs.json --min_corr 0.5 --k 32]   units co-active with a saved circuit = candidate connective tissue; with --file they are VERIFIED against equal-size random units
  union --names A,B,... [--min_frac 0.6]        nodes present in >= min_frac of the saved circuits (macro circuit)
  overlap --names A,B,...                       Jaccard / shared core / behaviour-specific nodes between saved circuits
  proposals --name N [--gamma G] [--only specific_to] edit proposals (JSON op configs) for a saved circuit's nodes
  weights --matrix NAME [--rows a:b --cols c:d] raw weights (<= 4000 values; HARNESS_RAW_VIEW=0 gives the summary-only arm)
  note --kind hypothesis|result|decision --text '...'   free, hash-chained research note (write one before and after every charged proposal)
  diff --a CKPT --b CKPT                        per-matrix relative weight change between two checkpoints (ints or 'final')
pairs.json = [{"name","clean":[token ids],"corrupt":[token ids],"correct":id,"wrong":id}, ...]   (strings also accepted when HARNESS_TOKENIZER names an HF tokenizer)"""
import os, sys, json, argparse, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import caveats as CV, circuit_grow as CG, baseline_map as BM, loop as LP, ops as OPS, adapter, structure as S, patching as P, circuit as C, prompt_pairs as PP, generic_model as GM, read_tools as RT
torch.set_num_threads(1)


def _log(run, rec):
    with open(os.path.join(run, "tools.jsonl"), "a") as f: f.write(json.dumps(rec, default=str) + "\n")


def _tok():
    n = os.environ.get("HARNESS_TOKENIZER")
    if not n: return None
    from transformers import AutoTokenizer; return AutoTokenizer.from_pretrained(n)


def _model(run, which=None):
    if which is None: return LP.target_model(LP.Run(run).config())
    return adapter.load_ckpt(LP.Run(run).config()["target_seed"], int(which) if str(which).isdigit() else which)


def _usable(m, file):
    enc = PP.encode(json.load(open(file)), _tok()); keep, rep = PP.check(m, enc); return keep, rep


def _group(keep):
    from collections import Counter
    L = Counter(int(p["clean"].shape[0]) for p in keep).most_common(1)[0][0]; return [p for p in keep if int(p["clean"].shape[0]) == L], L


def inventory(m):
    tm = S.discover(m); rows = []
    if tm.kind == "transformer":
        for l in range(tm.n_layers):
            rows.append({"component": f"block{l}.attention", "role": "attention heads", "units": tm.n_heads, "edit_with": f"scale_head/head_temperature layer={l} head=0..{tm.n_heads - 1}; nudge_weights/scale_slice on its q/k/v/o matrices"})
            rows.append({"component": f"block{l}.mlp", "role": "feedforward bank", "units": tm.d_ff, "edit_with": f"scale_neurons/prune_neurons layer={l} neurons=0..{tm.d_ff - 1}"})
    else:
        for l in range(tm.n_hidden): rows.append({"component": f"hidden{l}", "role": "MLP hidden layer", "units": tm.widths[l], "edit_with": f"scale_neurons/prune_neurons layer={l}"})
    return {"family": tm.family if tm.kind == "transformer" else "mlp", "n_params": int(sum(p.numel() for p in m.parameters())), "components": rows,
            "matrices": {n: {"role": r, "shape": list(s)} for n, (r, s) in GM.guess_roles(m).items()}, "ops_anywhere": "composite (<=4 sub-edits), revert_delta, blend_matrix, blend_slice, nudge_weights, scale_slice, rank1_update, bias_shift, rank_truncate, shrink_subspace, project_out_top_svd"}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("cmd", choices=["inventory", "pairs_check", "circuit", "overlap", "proposals", "weights", "diff", "note", "grow", "union", "baseline", "neighbors"])
    for k in ("file", "name", "names", "matrix", "rows", "cols", "a", "b", "kind", "text"): ap.add_argument(f"--{k}")
    ap.add_argument("--ckpt", default=None, help="analyse this checkpoint (0..9 or final) instead of the target"); ap.add_argument("--top_frac", type=float, default=0.8); ap.add_argument("--target", type=float, default=0.8); ap.add_argument("--n", type=int, default=512); ap.add_argument("--where", default="last"); ap.add_argument("--min_corr", type=float, default=None); ap.add_argument("--k", type=int, default=32); ap.add_argument("--circuit"); ap.add_argument("--baseline"); ap.add_argument("--min_frac", type=float, default=0.6); ap.add_argument("--gamma", type=float, default=0.0); ap.add_argument("--edges", action="store_true"); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv); run = a.run; m = _model(run, a.ckpt); out = None
    if a.cmd == "inventory": out = inventory(m)
    elif a.cmd == "pairs_check":
        keep, rep = _usable(m, a.file); out = {"n_usable": len(keep), "n_total": len(rep), "rows": rep}
    elif a.cmd == "circuit":
        keep, rep = _usable(m, a.file)
        if len(keep) < 3: out = {"error": f"only {len(keep)} usable pairs (need >= 3; ideally 5-10)", "rows": rep}
        else:
            grp, L = _group(keep); clean, corrupt, mf = PP.batch(grp); tm = S.discover(m)
            ns = P.exact_node_scores(m, tm, clean, corrupt, mf); agg = C.aggregate([ns]); circ = C.select(agg, top_frac=a.top_frac); f = C.faithfulness(m, tm, clean, corrupt, mf, circ)
            out = {"name": a.name, "n_pairs_used": len(grp), "n_pairs_dropped": len(rep) - len(grp), "token_len": L, "clean_metric": float(ns["clean_metric"].mean()), "corrupt_metric": float(ns["corrupt_metric"].mean()), "circuit": circ, "faithfulness": f}
            if a.edges and tm.kind == "transformer":
                es = P.edge_scores(m, tm, clean, corrupt, mf); rows = []
                for (wk, rk), v in es.items():
                    v = v.detach(); v = v.reshape(v.shape[0], -1) if v.ndim > 1 else v[:, None]
                    for h in range(v.shape[1]): rows.append((float(v[:, h].mean()), wk, rk, h))
                rows.sort(key=lambda r: -abs(r[0])); ed = []
                for fo, wk, rk, h in rows[:5]:
                    ex = float(P.exact_edge_effect(m, tm, clean, corrupt, wk, rk, mf, head=h if wk[0] == "head" else None).mean()); ed.append({"writer": list(wk), "reader": list(rk), "head": h if wk[0] == "head" else None, "first_order": round(fo, 4), "exact_direct_path": round(ex, 4)})
                out["top_edges"] = ed
            os.makedirs(os.path.join(run, "circuits"), exist_ok=True); json.dump(out, open(os.path.join(run, "circuits", f"{a.name}.json"), "w"), indent=1)
    elif a.cmd == "baseline":
        tm = S.discover(m); tok = adapter.probe_obs("ID", a.n)      # normal in-distribution prompts from the research side; never CANARY/HOLD
        tab = BM.record(m, tm, tok, a.where); st = BM.stats(tab); co = BM.coactivation(tab, st, a.min_corr or 0.6); os.makedirs(os.path.join(run, "baseline"), exist_ok=True)
        np.savez(os.path.join(run, "baseline", f"{a.name}.npz"), corr=co["corr"], cluster=co["cluster"], nodes=np.array([f"{k}.{l}.{i}" for k, l, i in co["nodes"]]))
        out = {"name": a.name, "n_prompts": int(len(tok)), "position": a.where, "per_layer": BM.summary(st), "largest_clusters": BM.clusters_summary(co)}
        json.dump(out, open(os.path.join(run, "baseline", f"{a.name}.json"), "w"), indent=1)
    elif a.cmd == "neighbors":
        z = np.load(os.path.join(run, "baseline", f"{a.name}.npz")); co = {"nodes": [(k, int(l), int(i)) for k, l, i in (x.split(".") for x in z["nodes"])], "corr": z["corr"], "cluster": z["cluster"]}
        circ = json.load(open(os.path.join(run, "circuits", f"{a.circuit}.json")))["circuit"]; nodes = [(n["kind"], n["layer"], n["index"]) for n in circ]; nb = BM.neighbors(co, nodes, a.min_corr or 0.5, a.k)
        out = {"circuit": a.circuit, "n_candidates": len(nb), "candidates": nb[:a.top]}
        if a.file and nb:
            keep, rep = _usable(m, a.file); grp, L = _group(keep); clean, corrupt, mf = PP.batch(grp); tm = S.discover(m)
            out["verification"] = BM.verify_neighbors(CG._Eval(m, tm, clean, corrupt, mf), nodes, [tuple(r["node"]) for r in nb], n_random=8)
    elif a.cmd == "grow":
        keep, rep = _usable(m, a.file)
        if len(keep) < 3: out = {"error": f"only {len(keep)} usable pairs (need >= 3)", "rows": rep}
        else:
            grp, L = _group(keep); clean, corrupt, mf = PP.batch(grp); tm = S.discover(m); ns = P.exact_node_scores(m, tm, clean, corrupt, mf); agg = C.aggregate([ns]); seed = C.select(agg, top_frac=a.top_frac)
            extra = None
            if a.baseline:
                z = np.load(os.path.join(run, "baseline", f"{a.baseline}.npz")); co = {"nodes": [(k, int(l), int(i)) for k, l, i in (x.split(".") for x in z["nodes"])], "corr": z["corr"], "cluster": z["cluster"]}; extra = BM.coactive_groups(co, [(n["kind"], n["layer"], n["index"]) for n in seed], a.min_corr or 0.4)
            out = CG.grow(m, tm, clean, corrupt, mf, agg, seed, target=a.target, extra_groups=extra); out["name"] = a.name
            os.makedirs(os.path.join(run, "circuits"), exist_ok=True); json.dump({**out, "circuit": out["nodes"]}, open(os.path.join(run, "circuits", f"{a.name}.json"), "w"), indent=1)
    elif a.cmd == "union":
        cs = {n: json.load(open(os.path.join(run, "circuits", f"{n}.json")))["circuit"] for n in a.names.split(",")}; u = CG.union_core(cs, a.min_frac); out = {"core": [list(x) for x in u["core"]], "counts": u["counts"]}
    elif a.cmd == "overlap":
        cs = {n: json.load(open(os.path.join(run, "circuits", f"{n}.json")))["circuit"] for n in a.names.split(",")}; out = C.overlap(cs)
    elif a.cmd == "proposals":
        circ = json.load(open(os.path.join(run, "circuits", f"{a.name}.json")))["circuit"]; nodes = [(n["kind"], n["layer"], n["index"]) for n in circ]
        out = {"proposals": C.edit_proposals(nodes, a.gamma)}
    elif a.cmd == "weights": out = RT.read_weights(m, a.matrix, a.rows, a.cols)
    elif a.cmd == "diff":
        m1, m2 = _model(run, a.a), _model(run, a.b); t1, t2 = GM.generic_matrix_table(m1), GM.generic_matrix_table(m2)
        rows = sorted(({"matrix": n, "rel_change": round(float((t2[n] - t1[n]).norm() / (t1[n].norm() + 1e-12)), 4)} for n in t1), key=lambda r: -r["rel_change"]); out = {"a": a.a, "b": a.b, "matrices": rows[:a.top]}
    elif a.cmd == "note":
        if a.kind not in ("hypothesis", "result", "decision"): raise SystemExit("--kind must be hypothesis|result|decision")
        r = LP.Run(run); r.log = os.path.join(run, "research.jsonl"); r.append(a.kind, {"text": a.text}, "analyst"); out = {"noted": True}
    _log(run, {"cmd": a.cmd, "args": {k: v for k, v in vars(a).items() if v not in (None, False)}})
    out = CV.attach(a.cmd, out)
    print(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__": main()
