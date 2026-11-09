"""Scripted (no-LLM) arms for the Nav2D arena, same loop/reward/budget as the Opus runs.
  null      : N=40 random schema-valid edits (all op families) evaluated zero-shot with the SAME evaluator (no budget): the null distribution of d_success
              and the false-accept rate; best-of-6 of this null is the fair comparison for an analyst that gets 6 tries.
  heuristic : 6 fixed generic edits (no reading of any data room), through the real loop (budget 6) + one HOLD score.
  random6   : the first 6 null edits through the real loop (budget 6) + one HOLD score.
Usage: python core/run_scripted_arms.py null|heuristic|random6   (sequential; one process at a time)"""
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import loop as LP, ops as OPS

ROOT = os.path.dirname(HERE)
MATS = ["tok_self", "tok_goal", "tok_obs", "type_emb", "head", "enc0.qkv", "enc0.out", "enc0.fc1", "enc0.fc2", "enc1.qkv", "enc1.out", "enc1.fc1", "enc1.fc2",
        "dec0.sqkv", "dec0.sout", "dec0.cq", "dec0.ckv", "dec0.cout", "dec0.fc1", "dec0.fc2", "dec1.sqkv", "dec1.sout", "dec1.cq", "dec1.ckv", "dec1.cout", "dec1.fc1", "dec1.fc2"]
HEUR = [("revert_delta", {"frac": 0.3, "ref_ckpt": 5}), ("revert_delta", {"frac": 0.3, "ref_ckpt": 8}),
        ("rank_truncate", {"matrix": "enc0.fc1", "rank": 32}), ("shrink_subspace", {"matrix": "tok_obs", "k": 1, "gamma": 0.5}),
        ("prune_neurons", {"kind": "enc", "layer": 1, "frac": 0.3, "gamma": 0.3}), ("project_out_top_svd", {"matrix": "head", "k": 1})]


def gen(rng):
    op = rng.choice(["project_out_top_svd", "rank_truncate", "shrink_subspace", "scale_head", "scale_neurons", "prune_neurons", "revert_delta"])
    if op == "project_out_top_svd": return op, {"matrix": rng.choice(MATS), "k": rng.randint(1, 2)}
    if op == "rank_truncate": return op, {"matrix": rng.choice(MATS), "rank": rng.choice([4, 8, 16, 32])}
    if op == "shrink_subspace": return op, {"matrix": rng.choice(MATS), "k": rng.randint(1, 3), "gamma": round(rng.uniform(0, 1), 2)}
    if op == "scale_head": return op, {"kind": rng.choice(["enc_self", "dec_self", "dec_cross"]), "layer": rng.randint(0, 1), "head": rng.randint(0, 3), "gamma": round(rng.uniform(0, 2), 2)}
    if op == "scale_neurons":
        return op, {"kind": rng.choice(["enc", "dec"]), "layer": rng.randint(0, 1), "neurons": sorted(rng.sample(range(256), 8)), "gamma": round(rng.uniform(0, 2), 2)}
    if op == "prune_neurons": return op, {"kind": rng.choice(["enc", "dec"]), "layer": rng.randint(0, 1), "frac": round(rng.uniform(0.1, 0.5), 2), "gamma": round(rng.uniform(0, 1), 2), "which": rng.choice(["lowest", "highest"])}
    return op, {"frac": round(rng.uniform(-0.3, 0.5), 2), "ref_ckpt": rng.randint(0, 9)}


def random_edits(n, seed=0):
    rng = random.Random(seed); base = LP.target_model(); out = []
    while len(out) < n:
        op, cfg = gen(rng)
        try: OPS.validate_cfg(op, cfg, base)
        except (ValueError, KeyError, TypeError): continue
        if (op, cfg) not in out: out.append((op, cfg))
    return out


if __name__ == "__main__":
    arm = sys.argv[1]
    if arm == "null":
        d = os.path.join(ROOT, "runs_navopus", "null")
        if not os.path.isdir(d): LP.start(d, budget=6)
        ev = LP.evaluator(d); base = LP.target_model(); refs = LP.ref_models(); rows = []
        for i, (op, cfg) in enumerate(random_edits(40, seed=0)):
            try: e = OPS.apply(op, cfg, base, ctx={"ref_models": refs}); rep = ev.evaluate(e)
            except Exception as err: print(i, op, "ERR", err, flush=True); continue
            m = LP.RW.public(rep); rows.append({"i": i, "op": op, "cfg": cfg, "d_success": m["d_success"], "id_drop": m["id_drop"], "accepted": bool(rep["accept"])})
            print(i, op, round(m["d_success"], 3), rows[-1]["accepted"], flush=True)
            json.dump(rows, open(os.path.join(ROOT, "results", "nav_null_edits.json"), "w"), indent=1)
    else:
        d = os.path.join(ROOT, "runs_navopus", arm)
        if not os.path.isdir(d): LP.start(d, budget=6)
        props = HEUR if arm == "heuristic" else random_edits(6, seed=0)
        done = len(LP.Run(d).of("iterate")) + len(LP.Run(d).strikes())
        for op, cfg in props[done:]:
            try: e = LP.propose(d, op, cfg, actor=f"arm:{arm}"); m = e["payload"]["metrics"]; print(arm, op, cfg, round(m["d_success"], 3), m["accepted"], flush=True)
            except LP.LoopError as err: print(arm, op, cfg, "ERR", err, flush=True)
        r = LP.finish(d); print(json.dumps(r, default=str)[:1500])
