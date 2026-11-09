"""Haiku-launched data-room collection. Usage: python core/collect.py RUN_DIR true|wrong [--n-ablate 200]
true  -> data room of the TARGET model (seed 0 final) with all 10 earlier checkpoints as references
wrong -> CONTROL: identical-format data room of the seed-1 model (the peek tool then also reads that model)
Writes RUN_DIR/room/* (digests + MANIFEST.json), RUN_DIR/../_private/<run>_kind.txt (overseer eyes only), RUN_DIR/peek.sh and prints the manifest sha256.
The ablation is resumable (RUN_DIR/room_ablation_cache.json, kept OUTSIDE the room); re-run the same command if it stops early."""
import os, sys, json, hashlib, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch; torch.set_num_threads(1)
import adapter, dataroom as DR
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("kind", choices=["true", "wrong"]); ap.add_argument("--n-ablate", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); run, kind = a.run, a.kind
room = os.path.join(run, "room"); os.makedirs(room, exist_ok=True)
cache = os.path.join(run, f"_{kind}_ablation_cache.json")
if kind == "true":
    man = DR.build_dataroom(adapter.load_ckpt(a.seed, "final"), room, ref_models={k: adapter.load_ckpt(a.seed, k) for k in range(10)}, n_ablate=a.n_ablate, ablation_cache=cache); seed = a.seed
else:
    man = DR.build_wrong_room(room, wrong_seed=1, n_ablate=a.n_ablate, ablation_cache=cache); seed = 1
priv = os.path.join(os.path.dirname(os.path.abspath(run)), "_private"); os.makedirs(priv, exist_ok=True)
open(os.path.join(priv, os.path.basename(os.path.abspath(run)) + "_kind.txt"), "w").write(kind + "\n")
open(os.path.join(run, "peek.sh"), "w").write("#!/bin/bash\nOMP_NUM_THREADS=1 ${PYTHON:-python} %s/peek.py %s --seed %d \"$@\"\n" % (HERE, os.path.abspath(room), seed))
os.chmod(os.path.join(run, "peek.sh"), 0o755)
print(json.dumps({"room": room, "n_files": len(man["files"]), "complete": man["complete"], "manifest_sha256": hashlib.sha256(open(os.path.join(room, "MANIFEST.json"), "rb").read()).hexdigest()}))
