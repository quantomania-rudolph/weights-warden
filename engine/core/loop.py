"""core budgeted-iteration loop for the Nav2D ACT-like policy (same guarantees as lab3/loop.py, adapted to zero-shot closed-loop evaluation).
  * hash-chained append-only log (log.jsonl) with a pinned config; verify() re-walks the chain
  * a proposal is validated against THIS model's ranges, applied to the TARGET model (seed-0 final), and evaluated zero-shot on VAL_SEEN (1000 episodes)
    + ID_EVAL (500 episodes); the CANARY (150 hidden layouts) is evaluated only when every other accept condition holds and ONLY pass/fail is reported/logged
  * ALL rejected proposals (invalid / duplicate / apply failure / non-finite) are charged to the budget (6)
  * the final edit is chosen by a FROZEN rule: among ACCEPTED proposals, the largest VAL_SEEN delta success (ties: smaller ID drop, then lower index);
    if nothing was accepted the final model is the UNEDITED target (identity), flagged no_accept
  * HOLD_OOD is read ONLY in score(), once per run, with 3 independent episode seeds (edit and base paired on identical episodes)
CLI:  python core/loop.py start RUN [--budget 6]
      python core/loop.py ops RUN                          (prints the op listing: every argument with its allowed values / range)
      python core/loop.py propose RUN --op OP --cfg 'JSON'  (what the analyst uses)
      python core/loop.py status RUN
      python core/loop.py finish RUN                       (overseer only: choose -> HOLD score -> close)
"""
import os, sys, json, hashlib, argparse, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import torch
import adapter
import ops as OPS
import rewards as RW

N_HOLD, HOLD_SEED_BASE, HOLD_ID_SEED_BASE, N_HOLD_ID = 1000, 9100, 9200, 500
TARGET_SEED = 0
SELECTION = "accepted only; max VAL_SEEN delta success, tie smaller ID drop, tie lower index; none accepted -> unedited target"
_MODELS = {}


def _sha(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class LoopError(Exception): pass


class Run:
    def __init__(self, d): self.d = d; self.log = os.path.join(d, "log.jsonl")

    def entries(self): return [json.loads(l) for l in open(self.log)] if os.path.isfile(self.log) else []

    def append(self, kind, payload, actor):
        es = self.entries(); prev = es[-1]["hash"] if es else "0" * 64
        e = {"seq": len(es), "kind": kind, "actor": actor, "payload": payload, "prev": prev}
        e["hash"] = _sha({k: e[k] for k in ("seq", "kind", "actor", "payload", "prev")})
        with open(self.log, "a") as f: f.write(json.dumps(e, sort_keys=True, default=str) + "\n")
        return e

    def verify(self, expect_head=None):
        prev, problems = "0" * 64, []
        for i, e in enumerate(self.entries()):
            if e["seq"] != i or e["prev"] != prev: problems.append(f"chain broken at {i}")
            if e["hash"] != _sha({k: e[k] for k in ("seq", "kind", "actor", "payload", "prev")}): problems.append(f"hash mismatch at {i}")
            prev = e["hash"]
        if expect_head and prev != expect_head: problems.append("head differs from the externally anchored head")
        return problems

    def of(self, kind): return [e["payload"] for e in self.entries() if e["kind"] == kind]
    def config(self): return self.of("start")[0]
    def strikes(self): return self.of("strike")
    def left(self): return self.config()["budget"] - len(self.of("iterate")) - len(self.strikes())


def target_model(cfg=None):
    seed = (cfg or {}).get("target_seed", TARGET_SEED)
    if ("t", seed) not in _MODELS: _MODELS[("t", seed)] = adapter.load_ckpt(seed, "final")
    return _MODELS[("t", seed)]


def ref_models(cfg=None):
    seed = (cfg or {}).get("target_seed", TARGET_SEED)
    if ("r", seed) not in _MODELS: _MODELS[("r", seed)] = {k: adapter.load_ckpt(seed, k) for k in range(OPS.N_CKPT)}
    return _MODELS[("r", seed)]


def evaluator(d):
    r = Run(d); cfg = r.config()
    if ("ev", os.path.abspath(d)) not in _MODELS:
        _MODELS[("ev", os.path.abspath(d))] = RW.Evaluator(target_model(cfg), cfg["n_val"], cfg["n_id"], cfg["n_canary"])
    return _MODELS[("ev", os.path.abspath(d))]


def start(d, budget=6, target_seed=TARGET_SEED, note=""):
    os.makedirs(d, exist_ok=True); r = Run(d)
    if r.entries(): raise LoopError("run already started")
    cfg = {"budget": int(budget), "target_seed": int(target_seed), "n_val": RW.N_VAL, "n_id": RW.N_ID, "n_canary": RW.N_CAN, "n_hold": N_HOLD, "hold_seeds": 3,
           "selection": SELECTION, "accept_rule": {"min_dsuccess": RW.MIN_DSUCC, "ci_lower_gt": 0, "id_drop_max": RW.ID_DROP_MAX, "canary": "pass", "collision_up_max": RW.COL_UP_MAX},
           "note": note}
    r.append("start", cfg, "overseer"); return cfg


def listing(d=None):
    """the op listing shown to the analyst: every argument with its allowed values (never abbreviated)."""
    cfg = Run(d).config() if d else None
    return OPS.describe_ops(target_model(cfg), list(range(OPS.N_CKPT)))


def propose(d, op, cfg_op, actor="analyst"):
    r = Run(d); cfg = r.config(); done = r.of("iterate"); strikes = r.strikes()
    if r.of("edit"): raise LoopError("edit already written")
    if len(done) + len(strikes) >= cfg["budget"]: raise LoopError(f"budget exhausted ({len(done)} evaluated + {len(strikes)} rejected of {cfg['budget']})")
    def strike(why):
        r.append("strike", {"op": op, "cfg": cfg_op, "reason": str(why)[:300]}, actor); raise LoopError(f"rejected (charged 1 evaluation): {why}")
    base = target_model(cfg)
    try: OPS.validate_cfg(op, cfg_op, base)
    except (ValueError, KeyError, TypeError) as e: strike(f"invalid: {e}")
    sha = _sha({"op": op, "cfg": cfg_op})
    if any(p["proposal_sha256"] == sha for p in done): strike("duplicate")
    try: edited = OPS.apply(op, cfg_op, base, ctx={"ref_models": ref_models(cfg)})
    except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as e: strike(f"apply failed: {e}")
    if not all(bool(torch.isfinite(p).all()) for p in edited.parameters()): strike("non-finite weights")
    t = time.time(); rep = evaluator(d).evaluate(edited)
    m = RW.public(rep); m["secs"] = round(time.time() - t, 1)
    if not all(math.isfinite(v) for v in m.values() if isinstance(v, float)): strike("non-finite metrics")
    m["accepted"] = bool(rep["accept"])
    return r.append("iterate", {"index": len(done), "op": op, "cfg": cfg_op, "proposal_sha256": sha, "metrics": m}, actor)


# ------------------------------------------------------------------ uncharged screening (research side only)
SCREEN_SPLIT, SCREEN_SEED, SCREEN_N, SCREEN_MAX = "TRAIN", 7000, 500, 60


def _screen_log(d):
    sr = Run(d); sr.log = os.path.join(d, "screens.jsonl"); return sr             # its own hash chain; the charged ledger (log.jsonl) is untouched


def _set_path(cfg, path, value):
    """cfg['gamma'] or a dotted path into a composite, e.g. 'ops.1.cfg.gamma'. returns a deep copy."""
    out = json.loads(json.dumps(cfg)); cur = out; parts = path.split(".")
    for k in parts[:-1]: cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    last = parts[-1]
    if isinstance(cur, list): cur[int(last)] = value
    else:
        if last not in cur: raise KeyError(f"parameter {path!r} not in cfg")
        cur[last] = value
    return out


def _screen_one(d, op, cfg_op, cfg, base, n):
    OPS.validate_cfg(op, cfg_op, base); edited = OPS.apply(op, cfg_op, base, ctx={"ref_models": ref_models(cfg)})
    if not all(bool(torch.isfinite(p).all()) for p in edited.parameters()): raise ValueError("non-finite weights")
    key = ("scr_base", os.path.abspath(d), n)
    if key not in _MODELS: _MODELS[key] = RW.measure(base, SCREEN_SPLIT, n, SCREEN_SEED)
    b = _MODELS[key]; e = RW.measure(edited, SCREEN_SPLIT, n, SCREEN_SEED); dm, lo, hi = RW.paired_boot(b["ep_success"], e["ep_success"])
    return {"d_success": round(float(dm), 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4), "d_collision": round(float(e["collision"] - b["collision"]), 4), "d_final_dist": round(float(e["final_dist"] - b["final_dist"]), 4)}


def _screen_budget(d, need):
    cfg = Run(d).config(); used = len(_screen_log(d).entries()); cap = int(cfg.get("screen_max", SCREEN_MAX))
    if used + need > cap: raise LoopError(f"screen cap reached ({used} used of {cap}; this call needs {need}). Spend a charged proposal or stop screening.")
    return cfg, used, cap


def screen(d, op, cfg_op, n=SCREEN_N, actor="analyst"):
    """UNCHARGED trial of one edit on RESEARCH episodes (split TRAIN, fixed seed, never the selection/ID/canary/HOLD sets the judge uses). Capped and logged in screens.jsonl."""
    cfg, used, cap = _screen_budget(d, 1); sr = _screen_log(d); base = target_model(cfg)
    try: res = _screen_one(d, op, cfg_op, cfg, base, n)
    except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as e:
        sr.append("screen", {"op": op, "cfg": cfg_op, "error": str(e)[:200]}, actor); return {"error": f"invalid: {e}", "screens_left": cap - used - 1}
    sr.append("screen", {"op": op, "cfg": cfg_op, **res}, actor)
    return {**res, "split": SCREEN_SPLIT + " (research, uncharged)", "n_episodes": n, "screens_left": cap - used - 1}


def sweep(d, op, cfg_op, param, values, n=SCREEN_N, actor="analyst"):
    """UNCHARGED strength sweep: one op, one numeric parameter, 2..9 values; each value costs one screen. Returns the curve and the best value on research episodes."""
    values = [float(v) for v in values]
    if not 2 <= len(values) <= 9: raise LoopError("sweep needs 2..9 values")
    cfg, used, cap = _screen_budget(d, len(values)); sr = _screen_log(d); base = target_model(cfg); curve = []
    for v in values:
        try: c = _set_path(cfg_op, param, v); res = _screen_one(d, op, c, cfg, base, n)
        except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as e:
            sr.append("screen", {"op": op, "cfg": cfg_op, "param": param, "value": v, "error": str(e)[:200]}, actor); curve.append({"value": v, "error": str(e)[:120]}); continue
        sr.append("screen", {"op": op, "cfg": c, "param": param, "value": v, **res}, actor); curve.append({"value": v, **res})
    ok = [c for c in curve if "d_success" in c]; best = max(ok, key=lambda c: c["d_success"]) if ok else None
    return {"op": op, "param": param, "curve": curve, "best_value": best["value"] if best else None, "split": SCREEN_SPLIT + " (research, uncharged)", "n_episodes": n, "screens_left": cap - used - len(values)}


def choose(d):
    its = Run(d).of("iterate"); ok = [p for p in its if p["metrics"]["accepted"]]
    if not ok: return {"index": None, "op": None, "cfg": None, "no_accept": True}
    best = min(ok, key=lambda p: (-p["metrics"]["d_success"], p["metrics"]["id_drop"], p["index"]))
    return {**best, "no_accept": False}


def score(d, seeds=(0, 1, 2)):
    """The ONE HOLD read: chosen edit (or the unedited target) vs the unedited target, 3 independent episode seeds, paired episodes."""
    r = Run(d); cfg = r.config(); ch = choose(d)
    if r.of("score"): raise LoopError("already scored")
    base = target_model(cfg)
    edit = base if ch["no_accept"] else OPS.apply(ch["op"], ch["cfg"], base, ctx={"ref_models": ref_models(cfg)})
    out = {"chosen": ch["index"], "op": ch["op"], "cfg": ch["cfg"], "no_accept": ch["no_accept"], "seeds": list(seeds), "per_seed": []}
    eb, ee, ib, ie = [], [], [], []
    for s in seeds:
        hb, he = RW.measure(base, "HOLD_OOD", cfg["n_hold"], HOLD_SEED_BASE + s), RW.measure(edit, "HOLD_OOD", cfg["n_hold"], HOLD_SEED_BASE + s)
        jb, je = RW.measure(base, "ID_EVAL", N_HOLD_ID, HOLD_ID_SEED_BASE + s), RW.measure(edit, "ID_EVAL", N_HOLD_ID, HOLD_ID_SEED_BASE + s)
        out["per_seed"].append({"seed": s, "hold_base": hb["success"], "hold_edit": he["success"], "hold_delta": he["success"] - hb["success"],
                                "hold_dist_base": hb["final_dist"], "hold_dist_edit": he["final_dist"], "id_base": jb["success"], "id_edit": je["success"],
                                "hold_collision_base": hb["collision"], "hold_collision_edit": he["collision"]})
        eb.append(hb["ep_success"]); ee.append(he["ep_success"]); ib.append(jb["ep_success"]); ie.append(je["ep_success"])
    d_, lo, hi = RW.paired_boot(np.concatenate(eb), np.concatenate(ee))
    out.update({"hold_delta_mean": d_, "hold_delta_lo": lo, "hold_delta_hi": hi, "hold_base_mean": float(np.mean([x["hold_base"] for x in out["per_seed"]])),
                "hold_edit_mean": float(np.mean([x["hold_edit"] for x in out["per_seed"]])), "id_base_mean": float(np.mean([x["id_base"] for x in out["per_seed"]])),
                "id_edit_mean": float(np.mean([x["id_edit"] for x in out["per_seed"]])),
                "hold_delta_spread_over_seeds": max(x["hold_delta"] for x in out["per_seed"]) - min(x["hold_delta"] for x in out["per_seed"]),
                "hold_gain_beats_noise": bool(lo > 0)})
    return r.append("score", out, "overseer")


def finish(d):
    r = Run(d); ch = choose(d)
    if not r.of("edit"): r.append("edit", {"chosen": ch["index"], "op": ch["op"], "cfg": ch["cfg"], "no_accept": ch["no_accept"]}, "overseer")
    if not r.of("score"): score(d)
    r.append("close", {"head_problems": r.verify()}, "overseer"); return r.of("score")[0]


def analyst_view(e):
    """what propose() prints for the analyst: no canary numbers, no HOLD, only the public report."""
    p = e["payload"]; m = p["metrics"]
    keys = ("base_val_success", "edit_val_success", "d_success", "d_success_lo", "d_success_hi", "d_final_dist", "d_collision", "base_id_success", "edit_id_success", "id_drop", "reward")
    return {"index": p["index"], **{k: round(m[k], 4) for k in keys}, "canary": m["canary"], "accepted": m["accepted"], "flags": m["flags"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("run"); ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--op"); ap.add_argument("--cfg"); ap.add_argument("--target-seed", type=int, default=TARGET_SEED)
    ap.add_argument("--param"); ap.add_argument("--values")
    a = ap.parse_args(); torch.set_num_threads(1)
    try:
        if a.cmd == "start": print(json.dumps(start(a.run, a.budget, a.target_seed)))
        elif a.cmd == "ops": print(listing(a.run))
        elif a.cmd == "propose":
            e = propose(a.run, a.op, json.loads(a.cfg)); v = analyst_view(e); v["evaluations_left"] = Run(a.run).left(); print(json.dumps(v))
        elif a.cmd == "screen": print(json.dumps(screen(a.run, a.op, json.loads(a.cfg))))
        elif a.cmd == "sweep": print(json.dumps(sweep(a.run, a.op, json.loads(a.cfg), a.param, [float(x) for x in a.values.split(",")])))
        elif a.cmd == "finish": print(json.dumps(finish(a.run), default=str)[:3000])
        elif a.cmd == "status":
            r = Run(a.run); print(json.dumps({"evaluated": len(r.of("iterate")), "rejected": len(r.strikes()), "evaluations_left": r.left(), "problems": r.verify()}))
    except LoopError as e:
        print(f"ERROR: {e}"); sys.exit(1)
