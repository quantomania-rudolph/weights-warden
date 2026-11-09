"""Stand-in arena for the core tests: a tiny policy with the SAME module names as docs/SPEC.md and a cheap `evaluate` shim.
Success of episode i = (the model's first 4 actions on observation i agree with a fixed TEACHER model within tau).  Splits are fixed observation pools
(ID = N(0,1), VAL_SEEN/CANARY/HOLD = shifted distributions), so edits that move the model toward the teacher improve every split (honest), edits that
only fit one pool do not (hack-like).  NOT a model of Nav2D; it exercises plumbing, statistics and the accept rule only."""
import copy
import numpy as np
import torch
import torch.nn as nn

D, H_ACT, NTOK = 32, 8, 7


class MHA(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.n_heads = nh; s.qkv = nn.Linear(d, 3 * d); s.out = nn.Linear(d, d)

    def forward(s, x):
        B, N, d = x.shape; q, k, v = s.qkv(x).view(B, N, 3, s.n_heads, d // s.n_heads).unbind(2)
        a = torch.einsum("bqhd,bkhd->bhqk", q, k).div(np.sqrt(d // s.n_heads)).softmax(-1)
        return s.out(torch.einsum("bhqk,bkhd->bqhd", a, v).reshape(B, N, d))


class Cross(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.n_heads = nh; s.q = nn.Linear(d, d); s.kv = nn.Linear(d, 2 * d); s.out = nn.Linear(d, d)

    def forward(s, x, mem):
        B, N, d = x.shape; dh = d // s.n_heads
        q = s.q(x).view(B, N, s.n_heads, dh); k, v = s.kv(mem).view(B, mem.shape[1], 2, s.n_heads, dh).unbind(2)
        a = torch.einsum("bqhd,bkhd->bhqk", q, k).div(np.sqrt(dh)).softmax(-1)
        return s.out(torch.einsum("bhqk,bkhd->bqhd", a, v).reshape(B, N, d))


class MLP(nn.Module):
    def __init__(s, d, ff):
        super().__init__(); s.fc1 = nn.Linear(d, ff); s.fc2 = nn.Linear(ff, d); s.act = nn.GELU()

    def forward(s, x): return s.fc2(s.act(s.fc1(x)))


class Enc(nn.Module):
    def __init__(s, d, nh, ff):
        super().__init__(); s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d); s.attn = MHA(d, nh); s.mlp = MLP(d, ff)

    def forward(s, x):
        x = x + s.attn(s.ln1(x)); return x + s.mlp(s.ln2(x))


class Dec(nn.Module):
    def __init__(s, d, nh, ff):
        super().__init__(); s.ln1, s.ln2, s.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        s.self_attn = MHA(d, nh); s.cross_attn = Cross(d, nh); s.mlp = MLP(d, ff)

    def forward(s, x, mem):
        x = x + s.self_attn(s.ln1(x)); x = x + s.cross_attn(s.ln2(x), mem); return x + s.mlp(s.ln3(x))


class TinyPolicy(nn.Module):
    def __init__(s, d=D, nh=4, ff=64, le=2, ld=2, seed=0):
        super().__init__(); g = torch.get_rng_state(); torch.manual_seed(seed)
        s.n_heads = nh; s.tok_self = nn.Linear(4, d); s.tok_goal = nn.Linear(2, d); s.tok_obs = nn.Linear(3, d); s.type_emb = nn.Embedding(3, d)
        s.enc_layers = nn.ModuleList([Enc(d, nh, ff) for _ in range(le)]); s.dec_layers = nn.ModuleList([Dec(d, nh, ff) for _ in range(ld)])
        s.query_emb = nn.Parameter(torch.randn(H_ACT, d) * 0.1); s.head = nn.Linear(d, 2); torch.set_rng_state(g)

    def forward(s, obs):                     # obs (B,21) = self4 goal2 obstacles 5x3
        B = obs.shape[0]; te = s.type_emb.weight
        toks = torch.cat([(s.tok_self(obs[:, :4]) + te[0])[:, None], (s.tok_goal(obs[:, 4:6]) + te[1])[:, None],
                          s.tok_obs(obs[:, 6:].reshape(B, 5, 3)) + te[2]], 1)
        for l in s.enc_layers: toks = l(toks)
        x = s.query_emb[None].expand(B, -1, -1)
        for l in s.dec_layers: x = l(x, toks)
        return torch.tanh(s.head(x))


def make_family(sigma=0.25, n_ckpt=10, seed=0):
    """teacher; target = teacher + noise (the 'overfit' model); ckpt k interpolates noise 1.0 -> 0.1 toward the teacher (ckpt 0 farthest)."""
    teacher = TinyPolicy(seed=seed).eval(); g = torch.Generator().manual_seed(123 + seed)
    noise = {n: torch.randn(p.shape, generator=g) * (p.abs().mean() + 0.05) * sigma for n, p in teacher.named_parameters()}

    def mk(scale):
        m = copy.deepcopy(teacher)
        with torch.no_grad():
            for n, p in m.named_parameters(): p.add_(noise[n] * scale)
        return m.eval()
    ckpts = {k: mk(1.0 - 0.9 * k / (n_ckpt - 1)) for k in range(n_ckpt)}
    return teacher, mk(1.0), ckpts


class StandinBackend:
    POOL = 700

    def __init__(s, sigma=0.125, tau=0.1, seed=0):
        s.teacher, s.target, s.ckpts = make_family(sigma, seed=seed); s.tau = tau; s.calls = []
        g = torch.Generator().manual_seed(7)
        s.pools = {"TRAIN": torch.randn(s.POOL, 21, generator=g), "ID_EVAL": torch.randn(s.POOL, 21, generator=g) * 0.9,
                   "VAL_SEEN": torch.randn(s.POOL, 21, generator=g) * 1.4 + 0.2, "CANARY": torch.randn(s.POOL, 21, generator=g) * 1.4 + 0.2,
                   "HOLD_OOD": torch.randn(s.POOL, 21, generator=g) * 1.4 + 0.2}
        s.ref = {k: s.teacher(v).detach() for k, v in s.pools.items()}

    def evaluate(s, model, split, n, seed):
        s.calls.append((split, n, seed))
        idx = torch.from_numpy(np.random.RandomState(seed).randint(0, s.POOL, n)).long()
        was = model.training; model.eval()
        with torch.no_grad():
            a = model(s.pools[split][idx])[:, :4]
        model.train(was)
        err = (a - s.ref[split][idx][:, :4]).abs().mean((1, 2)).numpy()
        succ = (err < s.tau).astype(float); col = (err > 3 * s.tau).astype(float)
        return {"success": succ.mean(), "collision": col.mean(), "timeout": 1 - succ.mean() - col.mean(), "final_dist": float(err.mean()), "path_ratio": 1.0,
                "ep_success": succ, "ep_collision": col, "ep_final_dist": err}

    def forward(s, model, obs): return model(obs)

    def probe_obs(s, kind, n): return s.pools["TRAIN" if kind == "ID" else "VAL_SEEN"][:n].clone()

    def load_ckpt(s, seed, which):
        if which == "final": return copy.deepcopy(s.target if seed == 0 else make_family(0.125, seed=seed)[1])
        return copy.deepcopy(s.ckpts[which])
