"""Small ACT-like transformer policy (no CVAE; L1 loss). Pure function of weights (LayerNorm only)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import env as E

DEFAULT_CFG = dict(d=128, n_heads=4, n_enc=3, n_dec=3, mlp_mult=4, H=E.H)


class MLP(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def _heads(x, h):
    B, L, D = x.shape
    return x.view(B, L, h, D // h).transpose(1, 2)   # B,h,L,dh


class SelfAttn(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.store_attn = False
        self.last_attn = None

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (_heads(t, self.n_heads) for t in (q, k, v))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
        if self.store_attn:
            self.last_attn = att.detach()
        y = (att @ v).transpose(1, 2).reshape(x.shape)
        return self.out(y)


class CrossAttn(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.store_attn = False
        self.last_attn = None

    def forward(self, x, mem):
        q = _heads(self.q(x), self.n_heads)
        k, v = (_heads(t, self.n_heads) for t in self.kv(mem).chunk(2, dim=-1))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
        if self.store_attn:
            self.last_attn = att.detach()
        y = (att @ v).transpose(1, 2).reshape(x.shape)
        return self.out(y)


class EncLayer(nn.Module):
    def __init__(self, d, n_heads, mult):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttn(d, n_heads)
        self.mlp = MLP(d, mult)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class DecLayer(nn.Module):
    def __init__(self, d, n_heads, mult):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = SelfAttn(d, n_heads)
        self.cross_attn = CrossAttn(d, n_heads)
        self.mlp = MLP(d, mult)

    def forward(self, x, mem):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), mem)
        return x + self.mlp(self.ln3(x))


class NavPolicy(nn.Module):
    """obs (B,21) -> action chunk (B,H,2). Token types: 0 self, 1 goal, 2 obstacle."""

    def __init__(self, cfg=None):
        super().__init__()
        c = dict(DEFAULT_CFG); c.update(cfg or {})
        self.cfg = c
        d, nh, m = c["d"], c["n_heads"], c["mlp_mult"]
        self.n_heads, self.H = nh, c["H"]
        self.tok_self = nn.Linear(39, d)   # x,y,vx,vy + goal-relative (dx,dy,dist): disclosed deviation
        self.tok_goal = nn.Linear(18, d)
        self.tok_obs = nn.Linear(19, d)    # cx,cy (Fourier) + r; absolute coords only
        self.type_emb = nn.Embedding(3, d)
        self.enc_layers = nn.ModuleList([EncLayer(d, nh, m) for _ in range(c["n_enc"])])
        self.enc_ln = nn.LayerNorm(d)
        self.query_emb = nn.Parameter(torch.randn(c["H"], d) * 0.02)
        self.dec_layers = nn.ModuleList([DecLayer(d, nh, m) for _ in range(c["n_dec"])])
        self.dec_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        self.register_buffer("types", torch.tensor([0, 1] + [2] * E.N_OBS), persistent=False)

    @staticmethod
    def fourier(z):
        """z (...,n) -> [z, sin(z*f), cos(z*f)] for f in 1,2,4,8 (fixed, not learned)."""
        f = torch.tensor([1.0, 2.0, 4.0, 8.0], device=z.device)
        zf = z[..., None] * f
        return torch.cat([z, torch.sin(zf).flatten(-2), torch.cos(zf).flatten(-2)], -1)

    def tokens(self, obs):
        B = obs.shape[0]
        # fixed (non-learned) input featurisation: coordinates centred/scaled + Fourier features; self token also gets
        # goal-relative offset, obstacle tokens are absolute (disclosed deviations from raw tokens).
        rel = (obs[:, 4:6] - obs[:, 0:2]) * 2
        ff = self.fourier
        s = self.tok_self(torch.cat([ff(torch.cat([(obs[:, 0:2] - 0.5) * 4, rel], 1)), obs[:, 2:4], rel.norm(dim=1, keepdim=True)], 1))[:, None]
        g = self.tok_goal(ff((obs[:, 4:6] - 0.5) * 4))[:, None]
        ob = obs[:, 6:].reshape(B, E.N_OBS, 3)
        o = self.tok_obs(torch.cat([ff((ob[..., :2] - 0.5) * 4), ob[..., 2:] * 10], -1))
        return torch.cat([s, g, o], 1) + self.type_emb(self.types)[None]

    def forward(self, obs):
        x = self.tokens(obs)
        for l in self.enc_layers:
            x = l(x)
        mem = self.enc_ln(x)
        q = self.query_emb[None].expand(obs.shape[0], -1, -1)
        for l in self.dec_layers:
            q = l(q, mem)
        return self.head(self.dec_ln(q))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def make_policy_fn(model, exec_k=E.EXEC_K):
    """Batch obs (numpy) -> action chunk (B,H,2) numpy. env.run_episodes executes the first exec_k then re-plans."""
    model.eval()

    @torch.no_grad()
    def fn(obs):
        out = []
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        for i in range(0, len(o), 2048):
            out.append(model(o[i:i + 2048]))
        return torch.cat(out).clamp(-1, 1).numpy()
    return fn


def evaluate_model(model, layouts, n_episodes, seed, exec_k=E.EXEC_K):
    """Chunked receding-horizon closed-loop evaluation (execute first `exec_k` of each chunk, then re-plan)."""
    return E.evaluate(make_policy_fn(model, exec_k), layouts, n_episodes, seed, exec_k=exec_k)


def load_policy(path):
    ck = torch.load(path, map_location="cpu")
    m = NavPolicy(ck["cfg"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()
