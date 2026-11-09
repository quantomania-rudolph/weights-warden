"""Scripted (no-LLM) analyst following the planned protocol, through the REAL loop (budget 6, same reward):
  rank neurons (greedy: label-free contrast attribution D1attr; random: random ranking) -> per step, PROBE each remaining block's top-K set stacked on the current edit (free, research set) ->
  spend ONE charged evaluation on the best-looking stack -> continue until all blocks used, budget spent, or the probe stops improving.   usage: python nav_sandbox/scripted_analyst.py greedy|random RUN"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "core"))
import loop as LP, research as R
mode, run = sys.argv[1], sys.argv[2]; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7; ATT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ROOT, "nav_planted", "results", "attrib_small.json"); K = 24; BL = [("enc", 0), ("enc", 1), ("dec", 0), ("dec", 1)]
if not os.path.isdir(run): LP.start(run, budget=6, target_seed=SEED)
sb = R.Sandbox(run, SEED, max_calls=30); rng = np.random.default_rng(3)
d1 = np.array(json.load(open(ATT))["scores_d1"])
def cand(b): sc = d1[b * 256:(b + 1) * 256] if mode == "greedy" else rng.random(256); return {"op": "scale_neurons", "cfg": {"kind": BL[b][0], "layer": BL[b][1], "neurons": sorted(int(x) for x in np.argsort(-sc)[:K]), "gamma": 0.0}}
cands = {b: cand(b) for b in range(4)}; cur, remaining, best_probe = [], set(range(4)), 0.0
sb.note("hypothesis", f"{mode}: stack top-{K} {'contrast' if mode == 'greedy' else 'RANDOM'} neurons block by block; keep adding while the free probe (research-layout L1 vs expert) keeps improving.")
while remaining:
    pr = {b: sb.probe("composite", {"ops": cur + [cands[b]]}) for b in sorted(remaining)}; b = min(pr, key=lambda x: pr[x].get("d_research_l1", 9))
    if cur and pr[b]["d_research_l1"] >= best_probe - 1e-4: sb.note("decision", "probe no longer improves; stop"); break
    cur = cur + [cands[b]]; remaining.discard(b); best_probe = pr[b]["d_research_l1"]
    try: e = LP.propose(run, "composite", {"ops": cur}, actor=f"arm:{mode}"); m = e["payload"]["metrics"]; print(mode, "blocks", len(cur), "probe", round(best_probe, 4), "d_success", round(m["d_success"], 3), "id_drop", round(m["id_drop"], 3), "accepted", m["accepted"], flush=True)
    except LP.LoopError as err: print("ERR", err, flush=True); break
    sb.note("result", f"blocks={len(cur)} probe={best_probe:.4f} d_success={m['d_success']:.4f} accepted={m['accepted']}")
print("DONE", mode, flush=True)
