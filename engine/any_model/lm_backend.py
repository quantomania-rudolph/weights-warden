"""Out-of-the-box LANGUAGE-MODEL backend for the closed-loop harness (the adapter protocol of core/adapter.py), used by the green run.

WHY THIS EXISTS: the core loop (budget, hash-chained ledger, paired-bootstrap accept rule, canary, single HOLD read) only talks to the model through `adapter`
(evaluate / forward / probe_obs / load_ckpt).  Swapping the backend is therefore all it takes to run the SAME loop on a different model family.  Here the model is
a stock HuggingFace GPT-2 (2 layers, d=64, 4 heads, ~0.1M params) trained from scratch on a toy induction task ("... A B ... A -> B").

TASK/SPLITS: an 'episode' is one sequence; success = the last-position argmax is the token that followed the earlier occurrence of the query token.
  collision = predicted a DIFFERENT token that appeared in the context (an attractor error);  final_dist = 1 - p(correct) (continuous, so paired deltas are smooth).
  TRAIN/VAL_SEEN/ID_EVAL/CANARY draw N=5 distinct tokens per sequence from disjoint seed streams; HOLD_OOD uses N=6 (distribution shift) and is read only by loop.py finish.
CHECKPOINTS: ckpt_000..ckpt_009 are 10 equal stages of one training run; `final` = ckpt_009 with its two most important heads scaled down ('planted damage'), so there
  is a real, known improvement for repair edits to find.  meta.json records which heads (used ONLY by the green-run verifier, never shown to an agent).
CLI: python any_model/lm_backend.py train [OUT_DIR]"""
import os, sys, json, copy, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import adapter

V, N_TRAIN, N_HOLD = 8, 5, 6
CFG = dict(n_layer=2, n_embd=64, n_head=4, vocab_size=V, n_positions=16, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
BASE = {"TRAIN": 11, "VAL_SEEN": 22, "ID_EVAL": 33, "CANARY": 44, "HOLD_OOD": 55}
STAGES, STEPS_PER_STAGE = 10, 200


def make_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config(**CFG))


def logits_of(model, x):
    out = model(x); return out.logits if hasattr(out, "logits") else out


def make_batch(n, rng, n_tok=N_TRAIN):
    """returns tokens (n,n_tok+1), target (n,)  where the last token repeats an earlier one and the target is the token that followed it."""
    seqs, tgt = [], []
    for _ in range(n):
        t = rng.permutation(V); seen = t[:n_tok]; j = int(rng.integers(0, n_tok - 1)); seqs.append(list(seen) + [seen[j]]); tgt.append(int(seen[j + 1]))
    return torch.tensor(seqs), torch.tensor(tgt)


def default_root(): return os.environ.get("LAB_LM_RUNS", os.path.join(os.path.dirname(ROOT), "models", "lm_toy"))


class LMBackend:
    def __init__(self, root=None): self.root = root or default_root(); self._probe = {}

    def evaluate(self, model, split, n_episodes, seed):
        rng = np.random.default_rng(BASE[split] * 100003 + int(seed))            # fixed (split, seed) -> identical episodes for base and edit => paired
        x, y = make_batch(n_episodes, rng, N_HOLD if split == "HOLD_OOD" else N_TRAIN)
        with torch.no_grad(): lg = logits_of(model, x)[:, -1]
        p = torch.softmax(lg, -1); pred = lg.argmax(-1)
        succ = (pred == y).float().numpy()
        seen = torch.zeros(len(x), V, dtype=torch.bool).scatter_(1, x, True)             # tokens present in the context
        col = ((pred != y) & seen[torch.arange(len(x)), pred]).float().numpy()
        fd = (1.0 - p[torch.arange(len(x)), y]).numpy()
        return {"success": float(succ.mean()), "collision": float(col.mean()), "timeout": 0.0, "final_dist": float(fd.mean()), "path_ratio": 1.0,
                "ep_success": succ, "ep_collision": col, "ep_final_dist": fd, "ep_timeout": np.zeros(len(succ)), "ep_path_ratio": np.ones(len(succ))}

    def forward(self, model, obs): return logits_of(model, obs)

    def probe_obs(self, kind, n):
        key = (kind, n)
        if key not in self._probe: self._probe[key] = make_batch(n, np.random.default_rng(777 if kind == "ID" else 778))[0]
        return self._probe[key].clone()

    def ckpt_path(self, seed, which):
        return os.path.join(self.root, f"policy_s{int(seed)}", "final.pt" if which == "final" else f"ckpt_{int(which):03d}.pt")

    def load_ckpt(self, seed, which):
        m = make_model(); m.load_state_dict(torch.load(self.ckpt_path(seed, which))); return m.eval()


def accuracy(model, split="VAL_SEEN", n=1000, seed=0): return LMBackend().evaluate(model, split, n, seed)["success"]


def head_ablation_drops(model, n=1000):
    base = accuracy(model, n=n); H = CFG["n_head"]; d = CFG["n_embd"] // H; out = {}
    for l in range(CFG["n_layer"]):
        for h in range(H):
            m = copy.deepcopy(model)
            with torch.no_grad(): m.transformer.h[l].attn.c_proj.weight[h * d:(h + 1) * d, :] = 0       # Conv1D weight is (in, out): rows = the head's slice of the concatenated heads
            out[(l, h)] = base - accuracy(m, n=n)
    return out


def damage(model, heads, g):
    m = copy.deepcopy(model); d = CFG["n_embd"] // CFG["n_head"]
    with torch.no_grad():
        for l, h in heads: m.transformer.h[l].attn.c_proj.weight[h * d:(h + 1) * d, :] *= g
    return m


def train_and_save(out_dir=None, seed=0, quiet=False):
    """train the toy model in STAGES stages, save the 10 checkpoints and a damaged `final`; returns meta (also written to meta.json)."""
    torch.manual_seed(seed); root = out_dir or default_root(); d = os.path.join(root, f"policy_s{seed}"); os.makedirs(d, exist_ok=True)
    m = make_model(); rng = np.random.default_rng(seed); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for s in range(STAGES):
        for _ in range(STEPS_PER_STAGE):
            x, y = make_batch(128, rng); loss = nn.functional.cross_entropy(logits_of(m, x)[:, -1], y); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); torch.save(m.state_dict(), os.path.join(d, f"ckpt_{s:03d}.pt")); m.train()
        if not quiet: print(f"stage {s} loss {float(loss):.3f}", flush=True)
    m.eval(); acc9 = accuracy(m)
    if acc9 < 0.9: raise RuntimeError(f"toy model did not learn induction (acc {acc9:.3f}); change seed")
    drops = head_ablation_drops(m); top = sorted(drops, key=drops.get, reverse=True)[:2]
    best = None
    for g in (0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2):                                      # pick the damage that leaves ~60% of the intact accuracy
        a = accuracy(damage(m, top, g))
        if best is None or abs(a - 0.6 * acc9) < abs(best[1] - 0.6 * acc9): best = (g, a)
    g, a_dam = best; fin = damage(m, top, g); torch.save(fin.state_dict(), os.path.join(d, "final.pt"))
    meta = {"params": int(sum(p.numel() for p in m.parameters())), "acc_ckpt009": acc9, "acc_final_damaged": a_dam, "planted_heads": [list(h) for h in top], "damage_gamma": g,
            "ablation_drops": {f"{l}.{h}": v for (l, h), v in drops.items()}, "NOTE": "verifier-only: never show planted_heads to an agent"}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    if not quiet: print(json.dumps({k: meta[k] for k in ("params", "acc_ckpt009", "acc_final_damaged", "planted_heads", "damage_gamma")}))
    return meta


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train": train_and_save(sys.argv[2] if len(sys.argv) > 2 else None)
    else: sys.exit(__doc__)
