import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "core"))
import ridge_op; ridge_op.enable(); import loop as LP
run = sys.argv[1]; seed = int(sys.argv[2])
if not os.path.isdir(run): LP.start(run, budget=6, target_seed=seed)
e = LP.propose(run, "ridge_head", {"lam": 0.1, "max_ratio": 1.5}, actor="scripted:ridge"); m = e["payload"]["metrics"]; print("ridge_head", {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
import ops as OPS
ed = OPS.apply("ridge_head", {"lam": 0.1, "max_ratio": 1.5}, LP.target_model({"target_seed": seed}), ctx={"ref_models": LP.ref_models({"target_seed": seed})}); print(getattr(ed, "_ridge_report", None))
