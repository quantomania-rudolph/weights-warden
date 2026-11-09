"""BRIDGE: opt-in registration of the closed-form `ridge_head` op into core's op registry WITHOUT touching core (so old runs/prompts/tests stay byte-identical).
Enable by `import ridge_op; ridge_op.enable()` (or env LAB4_ENABLE_RIDGE=1 before importing core.ops users that call nav_sandbox code). Default: NOT registered.
Why this op: the AutoResearch blog's trick = build a TEACHER by changing *inference* (not expressible as one weight edit), then ridge-fit chosen weights to reproduce it.
Our teacher = mean output of the saved reference checkpoints (a checkpoint ensemble, SWA-like).  Student = final linear head (dec_ln features -> 2), refit in closed form on
RESEARCH-layout states only (never VAL/HOLD); half the research states are held out and the fit is REFUSED (weights unchanged) if held-out error is worse than the guard allows."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import ops as OPS, fit_teacher as FT, research as R

_CACHE = {}


def _features(m, obs):
    """dec_ln output = the exact input of m.head (hook it so we fit the real thing)."""
    cap = []; h = m.head.register_forward_pre_hook(lambda mod, inp: cap.append(inp[0].detach()))
    with torch.no_grad(): y = m(obs)
    h.remove(); return cap[0], y


def _ridge_head(model, cfg, ctx):
    import copy
    refs = ctx.get("ref_models") or {}; ks = sorted(refs)
    if not ks: raise ValueError("ridge_head needs reference checkpoints")
    obs = R.research_obs(1600)
    with torch.no_grad(): Y = torch.stack([refs[k](obs) for k in ks]).mean(0)                    # teacher = checkpoint ensemble, (N,H,2)
    X, _ = _features(model, obs); N, H, d = X.shape
    Xf, Yf = X.reshape(-1, d), Y.reshape(-1, 2); half = (N // 2) * H
    res = FT.fit_with_guard(model.head.weight.data, Xf[:half], Yf[:half], Xf[half:], Yf[half:], bias=model.head.bias.data, lam=float(cfg["lam"]), max_ratio=float(cfg["max_ratio"]))
    out = copy.deepcopy(model)
    if res["applied"]:
        W = FT.ridge_to_teacher(model.head.weight.data, Xf, Yf, model.head.bias.data, float(cfg["lam"]))   # refit on ALL research states once the guard passed
        with torch.no_grad(): out.head.weight.copy_(W)
    out._ridge_report = {k: v for k, v in res.items() if k != "W"}
    return out


def enable():
    if "ridge_head" in OPS.REGISTRY: return
    OPS.register("ridge_head", {"lam": {"type": "float", "min": 1e-6, "max": 10.0}, "max_ratio": {"type": "float", "min": 1.0, "max": 3.0}},
                 "Closed-form refit of the final linear head toward the mean output of the saved reference checkpoints (checkpoint-ensemble teacher), on research-layout states, "
                 "refused (no change) if the held-out fit is worse than max_ratio x the fit-set error.")(_ridge_head)


if os.environ.get("LAB4_ENABLE_RIDGE") == "1": enable()
