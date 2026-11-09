"""Does the cheap teacher-forced probe predict the CHARGED closed-loop gain? Uses every planted-0.24M edit already evaluated: the Opus/scripted ledgers (plA, plB, pl_scripted) and nav_planted/results/scan_small.json."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "core"))
import research as R, ops as OPS, adapter
from scipy.stats import spearmanr, pearsonr
base = adapter.load_ckpt(7, "final"); refs = {k: adapter.load_ckpt(7, k) for k in range(10)}; b_r, b_t = R.teacher_forced(base, "research"), R.teacher_forced(base, "train"); rows = []
for run in ("plA", "plB", "pl_scripted"):
    for l in open(os.path.join(ROOT, "runs_navopus", run, "log.jsonl")):
        e = json.loads(l)
        if e["kind"] == "iterate": p = e["payload"]; rows.append((f"{run}:{p['op']}", p["op"], p["cfg"], p["metrics"]["d_success"]))
sj = json.load(open(os.path.join(ROOT, "nav_planted", "results", "scan_small.json"))); b0 = sj["base"]["VAL_SEEN"]
def comp(order):
    ops = []
    for b, (k, l) in enumerate([("enc", 0), ("enc", 1), ("dec", 0), ("dec", 1)]):
        ns = sorted(int(g % 256) for g in order if g // 256 == b)
        if ns: ops.append({"op": "scale_neurons", "cfg": {"kind": k, "layer": l, "neurons": ns, "gamma": 0.0}})
    return ops
for nm, a in list(sj["arms"].items()) + [(f"rand{j}", r) for j, r in enumerate(sj["random"])]:
    for k, val in a["all"]:
        if k <= len(a["order"]): rows.append((f"scan:{nm}:k{k}", "composite", {"ops": comp(a["order"][:k])}, val - b0))
res = []
for name, op, cfg, d in rows:
    e = OPS.apply(op, cfg, base, ctx={"ref_models": refs}); res.append((name, d, R.teacher_forced(e, "research") - b_r, R.teacher_forced(e, "train") - b_t))
d, r_, t_ = (np.array([x[i] for x in res]) for i in (1, 2, 3))
out = {"n_edits": len(res), "spearman_dsucc_vs_dresearchL1": float(spearmanr(d, r_)[0]), "pearson": float(pearsonr(d, r_)[0]), "spearman_dsucc_vs_dtrainL1": float(spearmanr(d, t_)[0]),
       "base_research_l1": b_r, "base_train_l1": b_t}
hi = d >= 0.02; out["mean_dresearchL1_when_dsucc>=+2pts"] = float(r_[hi].mean()) if hi.any() else None; out["mean_dresearchL1_when_dsucc<+2pts"] = float(r_[~hi].mean())
out["rows"] = [(n, round(x, 4), round(y, 5), round(z, 5)) for n, x, y, z in res]
json.dump(out, open(os.path.join(HERE, "proxy_check.json"), "w"), indent=1); print({k: v for k, v in out.items() if k != "rows"})
