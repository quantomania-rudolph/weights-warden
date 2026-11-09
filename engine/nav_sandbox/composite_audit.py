"""Stacking-hack audit: composites add a selection surface (best-of-N over many 4-block random stacks). Evaluate N random-neuron composites (4 sub-edits x K neurons, gamma in {0,0.3}) through the FULL
reward against the unedited base, report false-accept rate and the best-of-N one. Also N random WEIGHT-noise composites of 4 matched-size perturbations aren't needed (ops are neuron scalings only).
usage: composite_audit.py TARGET_SEED N [K]"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, "core"))
import loop as LP, ops as OPS, rewards as RW
seed, N = int(sys.argv[1]), int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 24
cfg = {"target_seed": seed}; base = LP.target_model(cfg); ev = RW.Evaluator(base, RW.N_VAL, RW.N_ID, RW.N_CAN); rng = np.random.default_rng(1000 + seed)
BL = [("enc", 0), ("enc", 1), ("dec", 0), ("dec", 1)]; rows = []
for i in range(N):
    g = float(rng.choice([0.0, 0.3])); ops = [{"op": "scale_neurons", "cfg": {"kind": k, "layer": l, "neurons": sorted(int(x) for x in rng.choice(256, K, replace=False)), "gamma": g}} for k, l in BL]
    rep = ev.evaluate(OPS.apply("composite", {"ops": ops}, base, ctx={})); m = RW.public(rep); rows.append({"i": i, "gamma": g, "d_success": m["d_success"], "id_drop": m["id_drop"], "accepted": bool(rep["accept"])}); print(rows[-1], flush=True)
best = max(rows, key=lambda r: r["d_success"]); print(json.dumps({"seed": seed, "n": N, "false_accepts": sum(r["accepted"] for r in rows), "best_of_n_d_success": best["d_success"], "best_accepted": best["accepted"], "mean_d_success": float(np.mean([r["d_success"] for r in rows]))}))
json.dump(rows, open(os.path.join(HERE, f"composite_audit_s{seed}.json"), "w"))
