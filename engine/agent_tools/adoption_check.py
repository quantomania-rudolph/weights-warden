"""ADOPTION CHECK: run this against YOUR model + backend before any LLM touches it.  Prints PASS / FAIL / WARN per requirement and exits non-zero on any FAIL.
usage (toy demo):  python agent_tools/adoption_check.py                    your model: write a 20-line script that builds `model` and `backend` and calls audit(model, backend, splits=..., ckpts=...)
The checks encode README_EXPOSURE.md: each FAIL names the thing that must be exposed."""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools"): sys.path.insert(0, os.path.join(ROOT, p))

REQUIRED_ATTRS = ["tok_self", "tok_goal", "tok_obs", "type_emb", "query_emb", "enc_layers", "dec_layers", "head", "n_heads"]
LAYER_ATTRS = {"enc_layers": ["attn.qkv", "attn.out", "mlp.fc1", "mlp.fc2"], "dec_layers": ["self_attn.qkv", "self_attn.out", "cross_attn.q", "cross_attn.kv", "cross_attn.out", "mlp.fc1", "mlp.fc2"]}


def _has(o, path):
    for a in path.split("."):
        if not hasattr(o, a): return False
        o = getattr(o, a)
    return True


def audit(model, backend, splits=None, ckpts=None, obs_dim=21, chunk_shape=None, n_ep=8):
    R = []; add = lambda lvl, name, msg="": R.append((lvl, name, msg))
    miss = [a for a in REQUIRED_ATTRS if not hasattr(model, a)]
    for grp, paths in LAYER_ATTRS.items():
        if hasattr(model, grp) and len(getattr(model, grp)): miss += [f"{grp}[0].{p}" for p in paths if not _has(getattr(model, grp)[0], p)]
    add("PASS" if not miss else "FAIL", "model structure matches ops.py (or a ModelMap exists)", "" if not miss else f"missing {miss}: write a ModelMap for your architecture and point ops.matrix_table/attn_module/mlp_module at it (HANDOFF.md section 5.1)")
    add("PASS" if all(torch.isfinite(p).all() for p in model.parameters()) else "FAIL", "all weights finite")
    add("PASS" if hasattr(model, "action_heads") else "WARN", "model.action_heads(obs) -> {base, nudge/expert, final} exposed", "" if hasattr(model, "action_heads") else "needed only if the policy has an action-expert / residual-nudge head; without it the head-comparison tool cannot separate them")
    try:
        x = backend.probe_obs("VAL", 16); y = backend.forward(model, x); add("PASS" if y.ndim == 3 and y.shape[0] == 16 else "FAIL", "backend.forward gives (B, chunk, action_dim)", str(tuple(y.shape)))
        add("PASS" if hasattr(backend, "expert_chunk") else "WARN", "backend.expert_chunk(obs) (expert labels for held-out states; needed for the label-free contrast scans)")
    except Exception as e: add("FAIL", "backend.probe_obs/forward", repr(e)[:200])
    try:
        a, b, c = backend.evaluate(model, "VAL_SEEN", n_ep, 123), backend.evaluate(model, "VAL_SEEN", n_ep, 123), backend.evaluate(model, "VAL_SEEN", n_ep, 124)
        ok = all(k in a for k in ("ep_success", "ep_collision", "ep_final_dist")); add("PASS" if ok else "FAIL", "evaluate() returns per-episode arrays (paired bootstrap needs them)")
        add("PASS" if np.array_equal(a["ep_success"], b["ep_success"]) else "FAIL", "evaluate() is deterministic for a fixed seed (paired episodes)", "same seed must give identical starts/goals AND identical rollouts")
        add("PASS" if not np.array_equal(a["ep_final_dist"], c["ep_final_dist"]) else "WARN", "different seed gives different episodes")
    except Exception as e: add("FAIL", "backend.evaluate", repr(e)[:200])
    for sp in ("TRAIN", "VAL_SEEN", "CANARY", "ID_EVAL", "HOLD_OOD"):
        try: backend.evaluate(model, sp, 2, 1); add("PASS", f"split {sp} evaluable")
        except Exception as e: add("FAIL", f"split {sp} evaluable", repr(e)[:120])
    if splits:
        names = list(splits); bad = [(p, q) for i, p in enumerate(names) for q in names[i + 1:] if (p, q) not in {("TRAIN", "ID_EVAL"), ("ID_EVAL", "TRAIN")} and set(splits[p]) & set(splits[q])]
        add("PASS" if not bad else "FAIL", "scenario/layout ids are disjoint across splits (except ID_EVAL which may reuse TRAIN scenarios)", f"overlap: {bad}" if bad else "")
    else: add("WARN", "split scenario ids not supplied", "pass splits={name: set(scenario_ids)} so leakage can be checked")
    n = len(ckpts or []); add("PASS" if n >= 5 else "WARN", f"{n} saved training checkpoints (>=5 recommended; revert/blend ops and delta tools need them)")
    return R


def report(R):
    for lvl, name, msg in R: print(f"[{lvl}] {name}" + (f"  -- {msg}" if msg else ""))
    f = sum(1 for r in R if r[0] == "FAIL"); print(f"\n{f} FAIL, {sum(1 for r in R if r[0] == 'WARN')} WARN, {sum(1 for r in R if r[0] == 'PASS')} PASS"); return f


if __name__ == "__main__":
    import adapter, env as E, loop as LP
    # VAL_OOD is just the union alias VAL_SEEN u CANARY in this toy, so it is excluded from the disjointness check
    m = LP.target_model({"target_seed": 0}); sp = {k: {E.layout_key(l) for l in v} for k, v in E.splits().items() if k != "VAL_OOD"} if hasattr(E, "splits") and hasattr(E, "layout_key") else None
    sys.exit(1 if report(audit(m, adapter.backend(), splits=sp, ckpts=[adapter.load_ckpt(0, k) for k in range(10)])) else 0)
