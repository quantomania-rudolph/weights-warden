"""ZERO-SHOT closed-loop reward for an EDIT of the core Nav2D policy (no continued training: apply the edit, then evaluate in the simulator).

measure(model, split, n, seed)  -> eval dict with per-episode arrays (seed fixed per split => edit and base see IDENTICAL starts/goals => paired)
Evaluator(base_model).evaluate(edit_model) -> report; base measurements are cached.
  VAL_SEEN  (N_VAL=1000 episodes)    primary: paired delta success + paired bootstrap CI, delta mean final distance, delta collision rate
  ID_EVAL   (N_ID=500 episodes)      collateral damage: ID success drop
  CANARY    (N_CAN=1000 episodes)    disjoint hidden layouts; the analyst only ever sees pass/fail; only evaluated when everything else passes
  HOLD_OOD is NEVER read here (see loop.score).
Reward   R = d_succ + W_DIST * d_dist - W_ID * max(0, id_drop - ID_FREE) - W_COL * max(0, d_col - COL_TOL)      (d_succ, id_drop, d_col in fractions, d_dist in arena units)
ACCEPT   d_succ >= MIN_DSUCC (+3 points)  AND  paired-bootstrap (2.5th pct) lower bound of d_succ > 0  AND  id_drop <= ID_DROP_MAX (3 points)
         AND canary pass  AND R > 0  AND no hard flag (collision_up: collision rate up by more than COL_UP_MAX).
Canary pass: paired delta success on CANARY >= CANARY_MIN (+1 point) and its bootstrap lower bound > CANARY_LB (-3 points).
All thresholds are module constants (see REWARD_NOTES.md for how they were chosen).
"""
import numpy as np
import adapter

N_VAL, N_ID, N_CAN = 1000, 500, 1000
SEED = {"VAL_SEEN": 4101, "ID_EVAL": 4202, "CANARY": 4303}      # fixed per split: paired episodes for base and edit
BOOT_B, BOOT_SEED = 2000, 12345
MIN_DSUCC = 0.03
ID_DROP_MAX = 0.03
CANARY_MIN, CANARY_LB = 0.01, -0.03
W_DIST, W_ID, W_COL = 0.25, 2.0, 0.5
ID_FREE, COL_TOL = 0.01, 0.02
COL_UP_MAX = 0.05
_EPS = 1e-9


def measure(model, split, n, seed=None):
    res = adapter.evaluate(model, split, n, SEED[split] if seed is None else seed)
    out = {k: float(res[k]) for k in ("success", "collision", "timeout", "final_dist", "path_ratio") if k in res}
    out["ep_success"] = adapter.per_episode(res, "success"); out["ep_collision"] = adapter.per_episode(res, "collision")
    out["ep_final_dist"] = adapter.per_episode(res, "final_dist"); out["n"] = len(out["ep_success"])
    return out


def paired_boot(base, edit, B=None, seed=BOOT_SEED):
    B = BOOT_B if B is None else B
    """paired bootstrap over episodes of mean(edit - base). Returns (mean, lo2.5, hi97.5)."""
    d = np.asarray(edit, float) - np.asarray(base, float)
    if len(d) == 0:
        raise ValueError("no episodes")
    idx = np.random.RandomState(seed).randint(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def canary_decision(base_c, edit_c):
    d, lo, hi = paired_boot(base_c["ep_success"], edit_c["ep_success"])
    return {"pass": bool(d >= CANARY_MIN - _EPS and lo > CANARY_LB), "d_success": d, "lo": lo, "hi": hi}


def reward_report(base, edit, canary=None):
    """Pure function of measures. base/edit = {"val": measure, "id": measure}; canary = None (not run) | bool | canary_decision dict.
    Returns the full verdict; keys starting with '_' are PRIVATE (canary numbers) and must never reach the analyst."""
    d, lo, hi = paired_boot(base["val"]["ep_success"], edit["val"]["ep_success"])
    d_dist = base["val"]["final_dist"] - edit["val"]["final_dist"]
    d_col = edit["val"]["collision"] - base["val"]["collision"]
    id_drop = base["id"]["success"] - edit["id"]["success"]
    R = d + W_DIST * d_dist - W_ID * max(0.0, id_drop - ID_FREE) - W_COL * max(0.0, d_col - COL_TOL)
    flags = []
    if d_col > COL_UP_MAX: flags.append("collision_up")
    if id_drop > ID_DROP_MAX + _EPS: flags.append("id_damage")
    if d < MIN_DSUCC - _EPS: flags.append("gain_below_threshold")
    if lo <= 0: flags.append("ci_includes_zero")
    if R <= 0: flags.append("nonpositive_reward")
    canary_pass = canary["pass"] if isinstance(canary, dict) else canary
    rep = {"reward": R, "d_success": d, "d_success_lo": lo, "d_success_hi": hi, "d_final_dist": d_dist, "d_collision": d_col, "id_drop": id_drop,
           "base_val_success": base["val"]["success"], "edit_val_success": edit["val"]["success"], "base_id_success": base["id"]["success"],
           "edit_id_success": edit["id"]["success"], "flags": flags,
           "canary": "not_run" if canary_pass is None else ("pass" if canary_pass else "fail")}
    if isinstance(canary, dict): rep["_canary_private"] = canary
    pre = ("collision_up", "id_damage", "gain_below_threshold", "ci_includes_zero", "nonpositive_reward")
    rep["accept_before_canary"] = not any(f in flags for f in pre)
    if canary_pass is False: flags.append("canary_fail")
    rep["accept"] = bool(rep["accept_before_canary"] and canary_pass is True)
    return rep


class Evaluator:
    """Caches the base measurements; evaluate(edit_model) -> report.  Canary is evaluated lazily: only if every other accept condition holds
    (saves time and keeps the number of canary reads small)."""

    def __init__(self, base_model, n_val=N_VAL, n_id=N_ID, n_can=N_CAN, always_canary=False):
        self.base_model, self.n_val, self.n_id, self.n_can, self.always_canary = base_model, n_val, n_id, n_can, always_canary
        self._base = None; self._base_can = None; self.canary_reads = 0

    def base(self):
        if self._base is None:
            self._base = {"val": measure(self.base_model, "VAL_SEEN", self.n_val), "id": measure(self.base_model, "ID_EVAL", self.n_id)}
        return self._base

    def base_canary(self):
        if self._base_can is None:
            self._base_can = measure(self.base_model, "CANARY", self.n_can)
        return self._base_can

    def evaluate(self, edit_model):
        base = self.base(); edit = {"val": measure(edit_model, "VAL_SEEN", self.n_val), "id": measure(edit_model, "ID_EVAL", self.n_id)}
        rep = reward_report(base, edit)
        if rep["accept_before_canary"] or self.always_canary:
            self.canary_reads += 1
            rep = reward_report(base, edit, canary_decision(self.base_canary(), measure(edit_model, "CANARY", self.n_can)))
        return rep


def public(rep):
    """strip private keys (canary numbers) before showing a report to the analyst / writing it to the log."""
    return {k: v for k, v in rep.items() if not k.startswith("_")}
