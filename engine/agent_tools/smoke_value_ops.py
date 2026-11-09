"""End-to-end smoke: opt-in value ops through the REAL loop (validation, charging, evaluation) + the analyst-facing op listing."""
import os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "core"))
import more_ops; more_ops.enable(); import loop as LP, ops as OPS
txt = OPS.describe_ops(LP.target_model({"target_seed": 0})); print("\n".join(l for l in txt.splitlines() if "nudge_weights" in l or "scale_slice" in l or "head_temperature" in l or l.startswith("    deltas") or l.startswith("    positions"))[:1500])
run = "/tmp/smoke_value_ops"; shutil.rmtree(run, ignore_errors=True); LP.start(run, budget=12, target_seed=0)
for op, cfg in (("nudge_weights", {"matrix": "dec1.fc2", "axis": "col", "index": 5, "positions": [0, 1, 2], "deltas": [0.01, -0.01, 0.02]}), ("scale_slice", {"matrix": "dec1.fc1", "axis": "row", "indices": [3, 4], "gamma": 0.8}),
                ("head_temperature", {"kind": "dec_cross", "layer": 1, "head": 0, "gamma": 1.5}), ("nudge_weights", {"matrix": "dec1.fc2", "axis": "col", "index": 5, "positions": [0], "deltas": [5.0]})):
    try: m = LP.propose(run, op, cfg, actor="smoke")["payload"]["metrics"]; print(op, "d_success", round(m["d_success"], 3), "accepted", m["accepted"])
    except LP.LoopError as e: print(op, "REJECTED+charged:", str(e)[:120])
for op, cfg in (("bias_shift", {"bias": "dec_layers.1.mlp.fc2.bias", "positions": [0, 1], "deltas": [0.01, -0.01]}), ("rank1_update", {"matrix": "dec1.fc2", "rows": [0, 1], "cols": [2, 3], "u": [0.1, 0.1], "v": [0.1, -0.1], "alpha": 0.5}), ("blend_matrix", {"matrix": "dec1.fc2", "ref_ckpt": 9, "frac": 0.3})):
    try: m = LP.propose(run, op, cfg, actor="smoke")["payload"]["metrics"]; print(op, "d_success", round(m["d_success"], 3), "accepted", m["accepted"])
    except LP.LoopError as e: print(op, "REJECTED+charged:", str(e)[:140])
