import sys, os, json, numpy as np
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R + "/core"); import loop as LP
d = np.array(json.load(open(_R + "/nav_planted/results/attrib_small.json"))["scores_d1"]); run = _R + "/runs_navopus/pl_scripted"
if not os.path.isdir(run): LP.start(run, budget=6, target_seed=7)
for b, (k, l) in enumerate([("enc", 0), ("enc", 1), ("dec", 0), ("dec", 1)]):
    top = sorted(int(x) for x in np.argsort(-d[b * 256:(b + 1) * 256])[:24])
    try: e = LP.propose(run, "scale_neurons", {"kind": k, "layer": l, "neurons": top, "gamma": 0.0}, actor="arm:scripted_D1attr"); m = e["payload"]["metrics"]; print(k, l, {x: (round(m[x], 3) if isinstance(m[x], float) else m[x]) for x in ("d_success", "id_drop", "accepted") if x in m}, flush=True)
    except LP.LoopError as err: print(k, l, "ERR", err, flush=True)
