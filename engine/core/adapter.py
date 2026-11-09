"""Single seam between the core edit/analysis side and the arena (env.py / policy.py / data.py / train.py written by another builder).
Everything the analysis side needs from the arena goes through the functions below, so that tests can swap in a stand-in
(`set_backend(obj)`) and the real integration is one class (RealBackend) to adjust.

Backend protocol (all methods):
  evaluate(model, split, n_episodes, seed) -> dict with scalar keys success, collision, timeout, final_dist, path_ratio AND per-episode
        numpy arrays ep_success, ep_collision, ep_final_dist (length n_episodes; SAME seed => identical starts/goals => paired).
        split in {"VAL_SEEN","CANARY","ID_EVAL","HOLD_OOD","TRAIN"}.
  forward(model, obs)  -> (B, H, 2) action chunk for a float tensor of flat observations (B, 21) = self(4) goal(2) obstacles(5x3).
  probe_obs(kind, n)   -> fixed float tensor (n, 21); kind in {"ID","VAL"} (TRAIN-layout vs VAL_SEEN-layout observations; no HOLD, no CANARY).
  load_ckpt(seed, which) -> model (which = "final" or int 0..9)
"""
import os
import numpy as np
import torch

_STATE = {"backend": None}
SPLITS = ("TRAIN", "VAL_SEEN", "CANARY", "ID_EVAL", "HOLD_OOD")


def set_backend(b):
    _STATE["backend"] = b


def backend():
    if _STATE["backend"] is None:
        _STATE["backend"] = RealBackend()
    return _STATE["backend"]


def evaluate(model, split, n_episodes, seed):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return backend().evaluate(model, split, int(n_episodes), int(seed))


def forward(model, obs):
    return backend().forward(model, obs)


def probe_obs(kind="ID", n=256):
    return backend().probe_obs(kind, n)


def load_ckpt(seed, which="final"):
    return backend().load_ckpt(seed, which)


def per_episode(res, name):
    """Return the per-episode array for 'success' | 'collision' | 'final_dist' from an evaluate() result."""
    for k in (f"ep_{name}", f"{name}_ep", f"{name}_per_episode", f"per_episode_{name}"):
        if k in res:
            return np.asarray(res[k], dtype=float)
    raise KeyError(f"evaluate() result has no per-episode array for {name!r} (keys: {sorted(res)})")


class RealBackend:
    """Binding to core/env.py, policy.py and the checkpoints written by core/train.py (models/nav/policy_s{seed}/{ckpt_000..009,final}.pt).
    Override the checkpoint root with the env var LAB4_RUNS."""
    _probe = {}

    def __init__(self):
        import env, policy  # lazy: stand-in tests never need them
        self.env, self.policy = env, policy
        self.runs = os.environ.get("LAB4_RUNS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "nav"))

    def evaluate(self, model, split, n_episodes, seed):
        res = self.policy.evaluate_model(model, self.env.splits()[split], n_episodes, seed)
        return normalise_eval(res, n_episodes)

    def forward(self, model, obs):
        return model(obs)

    def probe_obs(self, kind, n):
        """Fixed batch of states visited by the EXPERT on TRAIN layouts (kind 'ID') or VAL_SEEN layouts ('VAL'); never CANARY/HOLD."""
        key = (kind, n)
        if key not in self._probe:
            E = self.env
            lay = E.splits()["ID_EVAL" if kind == "ID" else "VAL_SEEN"]
            ep = E.sample_episodes(lay, 64, 777 if kind == "ID" else 778)
            obs = E.run_episodes(ep, None, record=True)["obs"]
            sel = np.random.default_rng(5).permutation(len(obs))[:n]
            self._probe[key] = torch.as_tensor(obs[sel], dtype=torch.float32)
        return self._probe[key].clone()

    def expert_chunk(self, obs):
        return self.env.expert_chunk(obs)

    def finetune_on_split(self, model, split, steps=400, n_layouts=30, seed=0):
        """ADVERSARY tool (hacks.overfit_val_layouts): behaviour-clone the expert on `n_layouts` layouts of `split`; returns the fine-tuned model."""
        E = self.env
        lay = E.splits()[split][:n_layouts]
        ep = E.sample_episodes(lay, 20 * len(lay), 31337 + seed)
        obs = E.run_episodes(ep, None, record=True)["obs"]
        X = torch.as_tensor(obs, dtype=torch.float32); Y = torch.as_tensor(E.expert_chunk(obs), dtype=torch.float32)
        g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0); model.train()
        for _ in range(steps):
            i = torch.randint(0, len(X), (256,), generator=g)
            loss = (model(X[i]) - Y[i]).abs().mean(); opt.zero_grad(); loss.backward(); opt.step()
        return model.eval()

    def ckpt_path(self, seed, which):
        name = "final.pt" if which == "final" else f"ckpt_{int(which):03d}.pt"
        return os.path.join(self.runs, f"policy_s{int(seed)}", name)

    def load_ckpt(self, seed, which):
        return self.policy.load_policy(self.ckpt_path(seed, which))


def normalise_eval(res, n):
    """Map the env's result dict to the protocol above: scalar keys + ep_success/ep_collision/ep_final_dist arrays."""
    out = dict(res)
    per = res.get("per_episode") or {}
    for k in ("success", "collision", "timeout", "final_dist", "path_ratio"):
        if k in per:
            out[f"ep_{k}"] = np.asarray(per[k], float)
            out[k] = float(np.mean(per[k]))
    return out
