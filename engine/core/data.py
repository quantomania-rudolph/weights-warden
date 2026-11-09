"""Datasets: HAPPY, DART, DAGGER -- TRAIN layouts only, with a leak guard.

Dataset = obs (N,21) float32, act (N,H=8,2) float32 (clean expert chunk), layout_id (N) into TRAIN, source (N) {0 happy,1 dart,2 dagger}.
Sizes with defaults (eps_per_layout=30, K=40): HAPPY ~1200 eps (~55k samples), DART 3 sigmas x 1200 eps (~165k), DAGGER ~600 eps/round.
"""
import numpy as np
import env as E

HAPPY, DART, DAGGER = 0, 1, 2
DART_SIGMAS = (0.1, 0.2, 0.3)


class LeakError(ValueError):
    pass


class Dataset:
    def __init__(self, obs, act, layout_id, source):
        self.obs = np.asarray(obs, np.float32)
        self.act = np.asarray(act, np.float32)
        self.layout_id = np.asarray(layout_id, np.int32)
        self.source = np.asarray(source, np.int8)

    def __len__(self):
        return len(self.obs)

    @staticmethod
    def empty():
        return Dataset(np.zeros((0, E.OBS_DIM)), np.zeros((0, E.H, 2)), np.zeros(0), np.zeros(0))

    @staticmethod
    def concat(parts):
        parts = [p for p in parts if len(p)]
        if not parts:
            return Dataset.empty()
        return Dataset(*(np.concatenate([getattr(p, k) for p in parts]) for k in ("obs", "act", "layout_id", "source")))

    def sizes(self):
        return {n: int((self.source == s).sum()) for n, s in (("happy", HAPPY), ("dart", DART), ("dagger", DAGGER))}


def check_layouts(layouts, allowed=None):
    """Leak guard on a layout array: every layout must be a TRAIN layout (and never VAL/HOLD)."""
    sp = E.splits()
    keys = E.split_keys()
    for l in np.asarray(layouts):
        k = E.layout_key(l)
        for name in ("VAL_OOD", "HOLD_OOD"):
            if k in keys[name]:
                raise LeakError(f"layout from {name} found in training data")
        if k not in keys["TRAIN"]:
            raise LeakError("layout not in TRAIN")


def check_dataset(ds):
    """Raise LeakError unless every sample's obstacle block is exactly its TRAIN layout; also reject NaNs."""
    if not (np.all(np.isfinite(ds.obs)) and np.all(np.isfinite(ds.act))):
        raise LeakError("non-finite values in dataset")
    if len(ds) == 0:
        return
    tr = E.splits()["TRAIN"]
    if ds.layout_id.min() < 0 or ds.layout_id.max() >= len(tr):
        raise LeakError("layout_id outside TRAIN")
    keys = E.split_keys()
    lay = ds.obs[:, 6:].reshape(-1, E.N_OBS, 3)
    # unique layouts present in obs
    uniq = np.unique(np.round(lay.astype(np.float64), 6).reshape(len(lay), -1), axis=0)
    for u in uniq:
        # float32 round-trip: compare by nearest TRAIN layout instead of exact hash
        d = np.abs(tr.reshape(len(tr), -1) - u[None]).max(1)
        if d.min() > 1e-5:
            raise LeakError("sample obstacle block matches no TRAIN layout (VAL/HOLD leak or corruption)")
    mism = np.abs(lay - tr[ds.layout_id].astype(np.float32)).max(axis=(1, 2))
    if mism.max() > 1e-5:
        raise LeakError("layout_id does not match sample's obstacle block")


def _episodes(layouts_train, k, eps_per_layout, seed):
    """(K layouts x eps_per_layout) episodes on the first k TRAIN layouts."""
    lay = layouts_train[:k]
    check_layouts(lay)
    ep = E.sample_episodes(lay, k * eps_per_layout, seed)
    # force an exactly balanced layout assignment, re-sampling start/goal around it
    li = np.repeat(np.arange(k), eps_per_layout)
    ep["layout_idx"] = li
    ep["layouts"] = lay[li]
    rng = np.random.default_rng([int(seed), 31337])
    ep["start"] = E._free_points(rng, ep["layouts"])
    goal = E._free_points(rng, ep["layouts"])
    bad = np.linalg.norm(ep["start"] - goal, axis=1) <= E.MIN_SG_DIST
    while bad.any():
        idx = np.nonzero(bad)[0]
        goal[idx] = E._free_points(rng, ep["layouts"][idx])
        bad = np.linalg.norm(ep["start"] - goal, axis=1) <= E.MIN_SG_DIST
    ep["goal"] = goal
    return ep


def _to_ds(res, ep, source):
    obs = res["obs"]
    li = ep["layout_idx"][res["ep_idx"]]
    ds = Dataset(obs, E.expert_chunk(obs), li, np.full(len(obs), source))
    return ds


def collect_happy(k=40, eps_per_layout=30, seed=0):
    ep = _episodes(E.splits()["TRAIN"], k, eps_per_layout, seed * 1000 + 1)
    ds = _to_ds(E.run_episodes(ep, None, record=True), ep, HAPPY)
    check_dataset(ds)
    return ds


def collect_dart(k=40, eps_per_layout=30, seed=0, sigmas=DART_SIGMAS):
    parts = []
    for j, s in enumerate(sigmas):
        ep = _episodes(E.splits()["TRAIN"], k, eps_per_layout, seed * 1000 + 10 + j)
        rng = np.random.default_rng([seed, 55, j])
        parts.append(_to_ds(E.run_episodes(ep, None, noise=s, rng=rng, record=True), ep, DART))
    ds = Dataset.concat(parts)
    check_dataset(ds)
    return ds


def collect_dagger(policy_fn, beta, k=40, eps_per_layout=15, seed=0, round_idx=0, noise=0.0):
    """Roll out the current learner (expert action substituted w.p. beta per step), relabel visited states."""
    ep = _episodes(E.splits()["TRAIN"], k, eps_per_layout, seed * 1000 + 100 + round_idx)
    rng = np.random.default_rng([seed, 77, round_idx])
    ds = _to_ds(E.run_episodes(ep, policy_fn, beta=beta, noise=noise, rng=rng, record=True), ep, DAGGER)
    check_dataset(ds)
    return ds
