"""Nav2D: vectorised point-mass navigation among 5 circular obstacles (see SPEC.md).

Observation (21 floats): self(x,y,vx,vy) | goal(x,y) | 5 x obstacle(cx,cy,r).
"""
import hashlib
import numpy as np

T = 60
STEP = 0.04
N_OBS = 5
R_MIN, R_MAX = 0.06, 0.12
C_MIN, C_MAX = 0.1, 0.9
SEP_GAP = 0.10          # min free gap between two obstacles (centre dist >= r1+r2+SEP_GAP)
START_MARGIN = 0.05
GOAL_TOL = 0.05
MIN_SG_DIST = 0.5
OBS_DIM = 21
H = 8                   # action chunk length (labels)
TAN_SOFT = 0.15
EXEC_K = 4              # actions executed before re-planning

# split sizes / seeds (fixed, never regenerated)
SPLIT_SEED = 20240611
N_TRAIN, N_VAL, N_HOLD = 40, 300, 600
N_TRAINED = 12          # the policies are trained on the FIRST 12 TRAIN layouts only (recipe k=12); ID_EVAL must be exactly those (fix 2026-11-04: it used all 40, 28 of them unseen)
N_VAL_SEEN = 150


class LayoutError(ValueError):
    pass


# --------------------------------------------------------------------------- layouts
def validate_layout(layout):
    """Raise LayoutError for a malformed layout (shape 5x3)."""
    a = np.asarray(layout, dtype=np.float64)
    if a.shape != (N_OBS, 3):
        raise LayoutError(f"layout shape {a.shape} != {(N_OBS, 3)}")
    if not np.all(np.isfinite(a)):
        raise LayoutError("non-finite layout")
    c, r = a[:, :2], a[:, 2]
    eps = 1e-9
    if r.min() < R_MIN - eps or r.max() > R_MAX + eps:
        raise LayoutError("radius out of range")
    if c.min() < C_MIN - eps or c.max() > C_MAX + eps:
        raise LayoutError("centre out of range")
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    need = r[:, None] + r[None] + SEP_GAP - 1e-9
    off = ~np.eye(N_OBS, dtype=bool)
    if np.any(d[off] < need[off]):
        raise LayoutError("obstacles overlap / too close")


def validate_layouts(layouts):
    a = np.asarray(layouts)
    if a.ndim != 3:
        raise LayoutError(f"layouts must be (L,5,3), got shape {a.shape}")
    if a.shape[0] == 0:
        raise LayoutError("empty layout set")
    for l in a:
        validate_layout(l)
    return a.astype(np.float64)


def _sample_layout(rng):
    while True:
        r = rng.uniform(R_MIN, R_MAX, N_OBS)
        c = rng.uniform(C_MIN, C_MAX, (N_OBS, 2))
        d = np.linalg.norm(c[:, None] - c[None], axis=-1)
        need = r[:, None] + r[None] + SEP_GAP
        off = ~np.eye(N_OBS, dtype=bool)
        if np.all(d[off] >= need[off]):
            return np.concatenate([c, r[:, None]], 1)


def layout_key(layout):
    """Hashable identity of a layout (rounded to 1e-6)."""
    a = np.round(np.asarray(layout, dtype=np.float64), 6)
    return hashlib.sha1(a.tobytes()).hexdigest()


_SPLITS = None


def splits():
    """dict of float64 layout arrays: TRAIN, VAL_OOD, VAL_SEEN, CANARY, HOLD_OOD, ID_EVAL.
    Generated from ONE fixed-seed stream: TRAIN first, then VAL, then HOLD."""
    global _SPLITS
    if _SPLITS is None:
        rng = np.random.default_rng(SPLIT_SEED)
        allL, seen = [], set()
        while len(allL) < N_TRAIN + N_VAL + N_HOLD:
            l = _sample_layout(rng)
            k = layout_key(l)
            if k in seen:
                continue
            seen.add(k)
            allL.append(l)
        allL = np.stack(allL)
        tr = allL[:N_TRAIN]
        va = allL[N_TRAIN:N_TRAIN + N_VAL]
        ho = allL[N_TRAIN + N_VAL:]
        _SPLITS = dict(TRAIN=tr, VAL_OOD=va, VAL_SEEN=va[:N_VAL_SEEN], CANARY=va[N_VAL_SEEN:],
                       HOLD_OOD=ho, ID_EVAL=tr[:N_TRAINED])
        for v in _SPLITS.values():
            v.setflags(write=False)
    return _SPLITS


def split_keys():
    return {k: {layout_key(l) for l in v} for k, v in splits().items()}


# --------------------------------------------------------------------------- episodes
def _free_points(rng, lay, n_try=64):
    """lay (B,5,3) -> one point per row, in [0.05,0.95]^2, outside every obstacle by START_MARGIN."""
    B = lay.shape[0]
    out = np.zeros((B, 2))
    todo = np.ones(B, bool)
    while todo.any():
        idx = np.nonzero(todo)[0]
        p = rng.uniform(0.05, 0.95, (len(idx), 2))
        d = np.linalg.norm(p[:, None] - lay[idx, :, :2], axis=-1) - lay[idx, :, 2]
        ok = (d > START_MARGIN).all(1)
        out[idx[ok]] = p[ok]
        todo[idx[ok]] = False
    return out


def sample_episodes(layouts, n, seed):
    """Deterministic function of (layouts, n, seed): paired seeds give identical starts/goals."""
    layouts = validate_layouts(layouts)
    rng = np.random.default_rng([int(seed), 7919])
    li = rng.integers(0, len(layouts), n)
    lay = layouts[li]
    start = _free_points(rng, lay)
    goal = _free_points(rng, lay)
    bad = np.linalg.norm(start - goal, axis=1) <= MIN_SG_DIST
    while bad.any():
        idx = np.nonzero(bad)[0]
        goal[idx] = _free_points(rng, lay[idx])
        bad = np.linalg.norm(start - goal, axis=1) <= MIN_SG_DIST
    return dict(layout_idx=li, layouts=lay, start=start, goal=goal)


def make_obs(pos, vel, goal, lay):
    B = pos.shape[0]
    return np.concatenate([pos, vel, goal, lay.reshape(B, -1)], 1).astype(np.float32)


# --------------------------------------------------------------------------- expert
def expert_action(obs, margin=0.14, k_rep=1.6, k_tan=1.6):
    """Deterministic scripted planner; function of (pos, goal, obstacles) only. obs (B,21) -> (B,2)."""
    obs = np.asarray(obs, dtype=np.float64)
    pos, goal = obs[:, 0:2], obs[:, 4:6]
    lay = obs[:, 6:].reshape(-1, N_OBS, 3)
    g = goal - pos
    d = np.linalg.norm(g, axis=1, keepdims=True)
    gdir = g / np.maximum(d, 1e-9)
    att = gdir * np.minimum(1.0, d / STEP)
    v = pos[:, None, :] - lay[:, :, :2]                       # (B,5,2) away from centre
    dist = np.linalg.norm(v, axis=-1)
    clear = np.maximum(dist - lay[:, :, 2], 1e-3)
    vhat = v / np.maximum(dist, 1e-9)[..., None]
    w = np.clip((margin - clear) / margin, 0.0, None) ** 2   # 0 outside margin
    rep = (w[..., None] * vhat * k_rep).sum(1)
    # tangential escape: perpendicular to vhat, sign picks the side that agrees with goal direction
    perp = np.stack([-vhat[..., 1], vhat[..., 0]], -1)
    cr = (perp * gdir[:, None, :]).sum(-1)
    s = np.tanh(cr / TAN_SOFT)                                # smooth side-selection (sign flip softened)
    s = np.where(np.abs(cr) < 1e-6, 1.0, s)
    tan = (w[..., None] * perp * s[..., None] * k_tan).sum(1)
    a = att + rep + tan
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(1.0, n)


def expert_chunk(obs, h=H):
    """Label: expert's closed-loop h-step action sequence simulated from this state (no collision stop)."""
    obs = np.asarray(obs, dtype=np.float64)
    pos = obs[:, 0:2].copy()
    cur = obs.copy()
    out = np.zeros((obs.shape[0], h, 2), np.float32)
    for i in range(h):
        a = expert_action(cur)
        out[:, i] = a
        pos = np.clip(pos + STEP * a, 0.0, 1.0)
        cur[:, 0:2] = pos
        cur[:, 2:4] = a
    return out


# --------------------------------------------------------------------------- simulation core
def _seg_hits(p0, p1, lay):
    """closest approach of segment p0->p1 to each obstacle centre < r  -> (B,) bool."""
    dvec = p1 - p0
    f = p0[:, None, :] - lay[:, :, :2]
    dd = (dvec ** 2).sum(-1, keepdims=True)
    t = np.clip(-(f * dvec[:, None, :]).sum(-1) / np.maximum(dd, 1e-12), 0, 1)
    q = f + t[..., None] * dvec[:, None, :]
    return (np.linalg.norm(q, axis=-1) < lay[:, :, 2]).any(1)


def run_episodes(ep, policy_fn, exec_k=EXEC_K, beta=0.0, noise=0.0, rng=None, record=False):
    """Receding-horizon rollout of all episodes in `ep` simultaneously.
    policy_fn(obs (B,21)) -> chunk (B,h,2) (first min(exec_k,h) actions executed, then re-plan) or (B,2).
    beta: per-step prob. of replacing the action by the expert's; noise: gaussian sigma added to executed action.
    Returns dict(status (0 running/timeout,1 success,2 collision), final_dist, path_len, [obs list]).
    """
    lay, goal = ep["layouts"], ep["goal"]
    B = len(goal)
    pos, vel = ep["start"].copy(), np.zeros((B, 2))
    status = np.zeros(B, np.int8)
    path = np.zeros(B)
    rec_obs, rec_idx = [], []
    chunk, rep = None, 1
    for t in range(T):
        alive = status == 0
        if not alive.any():
            break
        obs = make_obs(pos, vel, goal, lay)
        if policy_fn is not None and (t % rep == 0 or chunk is None):
            chunk = np.asarray(policy_fn(obs))
            if chunk.ndim == 2:
                chunk = chunk[:, None]
            rep = max(1, min(exec_k, chunk.shape[1]))
            t0 = t
        a = chunk[:, (t - t0) % rep] if policy_fn is not None else expert_action(obs)
        if beta > 0:
            m = rng.random(B) < beta
            if m.any():
                a = np.where(m[:, None], expert_action(obs), a)
        if noise > 0:
            a = a + noise * rng.standard_normal(a.shape)
        a = np.clip(np.nan_to_num(a), -1, 1)
        if record:
            rec_obs.append(obs[alive]); rec_idx.append(np.nonzero(alive)[0])
        new = np.clip(pos + STEP * a, 0.0, 1.0)
        disp = new - pos
        hit = _seg_hits(pos, new, lay) & alive
        path = path + np.where(alive, np.linalg.norm(disp, axis=1), 0.0)
        mv = alive & ~hit
        vel = np.where(mv[:, None], disp / STEP, vel)
        pos = np.where(mv[:, None], new, pos)
        status[hit] = 2
        succ = mv & (np.linalg.norm(pos - goal, axis=1) < GOAL_TOL)
        status[succ] = 1
    res = dict(status=status, final_dist=np.linalg.norm(pos - goal, axis=1), path_len=path)
    if record:
        res["obs"] = np.concatenate(rec_obs) if rec_obs else np.zeros((0, OBS_DIM), np.float32)
        res["ep_idx"] = np.concatenate(rec_idx) if rec_idx else np.zeros(0, int)
    return res


# --------------------------------------------------------------------------- evaluation
def bootstrap_ci(x, n_boot=1000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot=2000, alpha=0.05, seed=0):
    """CI of mean(b - a) over paired episodes."""
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    lo, hi = bootstrap_ci(d, n_boot, alpha, seed)
    return dict(mean=float(d.mean()), lo=lo, hi=hi)


def evaluate(policy_fn, layouts, n_episodes, seed, exec_k=EXEC_K, ci_boot=1000):
    """Closed-loop evaluation. policy_fn: batch obs -> action chunk (B,h,2) or action (B,2); None = expert.
    Paired seeds: same (layouts, n_episodes, seed) => identical starts/goals regardless of policy."""
    ep = sample_episodes(layouts, n_episodes, seed)
    r = run_episodes(ep, policy_fn, exec_k=exec_k)
    st = r["status"]
    sg = np.linalg.norm(ep["goal"] - ep["start"], axis=1)
    per = dict(success=(st == 1).astype(float), collision=(st == 2).astype(float),
               timeout=(st == 0).astype(float), final_dist=r["final_dist"], path_ratio=r["path_len"] / sg)
    out = {"n": int(n_episodes), "seed": int(seed), "per_episode": per, "layout_idx": ep["layout_idx"]}
    for k, v in per.items():
        out[k] = float(v.mean())
        out[k + "_ci"] = bootstrap_ci(v, ci_boot, seed=seed)
    return out


def expert_fn(obs):
    return expert_action(obs)
