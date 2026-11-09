"""Label-free D1 contrast attribution for the UNPLANTED policy (negative control): train labels = the CLEAN expert labels the policy was trained on. Same attr() as nav_planted. usage: unplanted_attrib.py RUNDIR OUT.json"""
import os, sys, json, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "nav_planted")); sys.path.insert(0, os.path.join(ROOT, "core"))
import attrib as A
S, P, PM = A.S, A.P, A.PM
run, out = sys.argv[1], sys.argv[2]; m = P.load_policy(os.path.join(run, "final.pt")); rng = np.random.default_rng(0)
obs, clean, planted, mask = PM.build(); idx = rng.choice(len(obs), 4000, replace=False)
o, y = torch.from_numpy(obs[idx]), torch.from_numpy(clean[idx])
import adapter; vo = adapter.probe_obs("VAL", 2000); vy = torch.as_tensor(np.asarray(adapter.backend().expert_chunk(vo.numpy())), dtype=torch.float32)
d1 = A.attr(m, o, y, np.arange(len(o))) - A.attr(m, vo, vy, np.arange(len(vo))); json.dump({"scores_d1": [float(x) for x in d1]}, open(out, "w")); print("saved", out)
