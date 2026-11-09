"""Red-team audit of rewards.py against the hacks.py library + NULL control.
audit(base_model, ctx, ...) -> {hack_pass_rate, honest_pass_rate, naive_hack_pass_rate, null_false_accept_rate, ...}
  * every hack and honest edit goes through the full Evaluator (VAL_SEEN + ID_EVAL + lazy CANARY) against the unedited base
  * NULL control: n_null random weight perturbations matched in relative size to the honest edits (plus two tiny sizes = pure chaos), all with the
    canary evaluated, so we can report the false-accept rate of (a) the full rule, (b) the rule WITHOUT canary, (c) the VAL-only 'naive' rule
  * selection hack: the best-by-VAL null edit (best of n_null) is run through the rule: does selecting on VAL_SEEN fool the accept rule?
Resumable: per-edit results are pickled to cache_path after each edit, and `max_seconds` stops cleanly (re-run to continue) so one call never runs for
more than a few minutes.  Run: OMP_NUM_THREADS=1 python core/redteam.py [cache.pkl] [max_seconds]
"""
import os, sys, time, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import numpy as np
import adapter, ops, rewards as RW, hacks as HK


def _load(p):
    if p and os.path.exists(p):
        with open(p, "rb") as f: return pickle.load(f)
    return {}


def _save(p, c):
    if p:
        with open(p + ".tmp", "wb") as f: pickle.dump(c, f)
        os.replace(p + ".tmp", p)


def _slim(rep):
    return {k: v for k, v in rep.items() if k != "_canary_private"} | ({"canary_d": rep["_canary_private"]["d_success"]} if "_canary_private" in rep else {})


def null_plan(ctx, base, n_null, extra=(0.001, 0.005)):
    """sizes matched to the honest edits (measured on the actual model) + two tiny sizes; cycled to n_null edits."""
    sizes = [HK.rel_delta(base, e["fn"](base, ctx)) for e in HK.HONEST.values()] + list(extra)
    return [(i, sizes[i % len(sizes)]) for i in range(n_null)]


def audit(base_model, ctx, cache_path=None, only=None, n_null=12, max_seconds=None, verbose=False, evaluator=None, n_val=RW.N_VAL, n_id=RW.N_ID, n_can=RW.N_CAN):
    t0 = time.time(); cache = _load(cache_path); HK.register_backend_hacks()
    ev = evaluator or RW.Evaluator(base_model, n_val, n_id, n_can)
    ev_null = RW.Evaluator(base_model, n_val, n_id, n_can, always_canary=True); ev_null.base = ev.base          # share the (lazily computed) base measurements
    out = {"hacks": {}, "honest": {}, "null": {}, "complete": True}

    def run(kind, name, build, evaluator):
        key = f"{kind}/{name}"
        if key not in cache:
            if max_seconds and time.time() - t0 > max_seconds:
                out["complete"] = False; return
            t = time.time(); rep = RW.public(evaluator.evaluate(build())); rep["_canary_d"] = rep.get("canary_d")
            cache[key] = _slim(rep); _save(cache_path, cache)
            if verbose:
                r = cache[key]; print(f"{kind:7s} {name:30s} accept={r['accept']!s:5s} dS={r['d_success']:+.3f} lo={r['d_success_lo']:+.3f} idDrop={r['id_drop']:+.3f} "
                                      f"dCol={r['d_collision']:+.3f} R={r['reward']:+.3f} canary={r['canary']} flags={r['flags']} ({time.time() - t:.0f}s)", flush=True)
        out[{"hack": "hacks", "honest": "honest", "null": "null"}[kind]][name] = cache[key]

    for name, e in HK.HACKS.items():
        if not only or name in only: run("hack", name, lambda e=e: e["fn"](base_model, ctx), ev)
    for name, e in HK.HONEST.items():
        if not only or name in only: run("honest", name, lambda e=e: e["fn"](base_model, ctx), ev)
    for i, size in null_plan(ctx, base_model, n_null):
        nm = f"null{i}_rel{size:.4f}"
        if not only or nm in only or "null" in only: run("null", nm, lambda i=i, size=size: HK.random_matched_edit(base_model, size, 9000 + i), ev_null)
    h, o, nl = out["hacks"], out["honest"], out["null"]
    rate = lambda d, f: sum(bool(f(r)) for r in d.values()) / max(len(d), 1)
    out["hack_pass_rate"] = rate(h, lambda r: r["accept"]); out["honest_pass_rate"] = rate(o, lambda r: r["accept"])
    out["naive_hack_pass_rate"] = rate(h, lambda r: r["d_success"] >= RW.MIN_DSUCC)        # VAL-delta-only rule
    out["null_false_accept_rate"] = rate(nl, lambda r: r["accept"])
    out["null_false_accept_rate_no_canary"] = rate(nl, lambda r: r["accept_before_canary"])
    out["null_naive_false_accept_rate"] = rate(nl, lambda r: r["d_success"] >= RW.MIN_DSUCC)
    out["n_hacks"], out["n_honest"], out["n_null"] = len(h), len(o), len(nl)
    if nl:
        best = max(nl.items(), key=lambda kv: kv[1]["d_success"])
        out["selection_best_null"] = {"name": best[0], "d_success": best[1]["d_success"], "accept": best[1]["accept"], "accept_before_canary": best[1]["accept_before_canary"],
                                      "canary": best[1]["canary"]}
    out["secs"] = time.time() - t0
    return out


if __name__ == "__main__":
    import torch; torch.set_num_threads(1)
    cp = sys.argv[1] if len(sys.argv) > 1 else None; ms = float(sys.argv[2]) if len(sys.argv) > 2 else 240
    base = adapter.load_ckpt(0, "final"); ctx = {"ref_models": {k: adapter.load_ckpt(0, k) for k in range(10)}}
    r = audit(base, ctx, cache_path=cp, max_seconds=ms, verbose=True)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict) or k == "selection_best_null"}, default=str))
