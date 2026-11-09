"""nav_sandbox: research-phase tools for the Nav2D edit harness (inspired by AutoResearch-style weight editing: many cheap looks, few charged evaluations).
THREE-WAY DATA DISCIPLINE: RESEARCH set (free, many looks: 100 brand-new layouts that appear in NO other split) -> VAL_SEEN (charged: the budgeted loop) -> HOLD_OOD (read once).
The sandbox answers cheap TEACHER-FORCED questions (L1 error of the policy vs the expert's action chunk on expert-visited states) for any edit, logs every call in a hash-chained
research ledger (so a reader can audit exactly what the analyst looked at), and has a call budget. It never returns closed-loop success: that stays charged."""
import os, sys, json, hashlib, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, "core"))
import env as E, ops as OPS, adapter
torch.set_num_threads(1)
N_RESEARCH_LAYOUTS, RESEARCH_SEED = 100, 88001


def research_layouts():
    rng = np.random.default_rng(RESEARCH_SEED); used = set().union(*E.split_keys().values()); out, seen = [], set()
    while len(out) < N_RESEARCH_LAYOUTS:
        l = E._sample_layout(rng); k = E.layout_key(l)
        if k in used or k in seen: continue
        seen.add(k); out.append(l)
    L = np.stack(out); assert not ({E.layout_key(l) for l in L} & used); return L


_cache = {}
def research_obs(n=2000):
    if n not in _cache:
        ep = E.sample_episodes(research_layouts(), 64, 991); obs = E.run_episodes(ep, None, record=True)["obs"]; sel = np.random.default_rng(6).permutation(len(obs))[:n]
        _cache[n] = torch.as_tensor(obs[sel], dtype=torch.float32)
    return _cache[n].clone()


@torch.no_grad()
def teacher_forced(model, which="research", n=2000):
    obs = research_obs(n) if which == "research" else adapter.probe_obs("ID", n); y = torch.as_tensor(np.asarray(E.expert_chunk(obs.numpy())), dtype=torch.float32)
    model.eval(); return float((model(obs) - y).abs().mean())


class Sandbox:
    """Budgeted, logged research sandbox. ledger: RUN_DIR/research.jsonl (hash-chained: each line carries sha256 of (prev, payload))."""
    def __init__(self, run_dir, target_seed, max_calls=60):
        self.p = os.path.join(run_dir, "research.jsonl"); self.max_calls = max_calls; self.base = adapter.load_ckpt(target_seed, "final"); self.refs = {k: adapter.load_ckpt(target_seed, k) for k in range(10)}
        self.base_tf = {w: teacher_forced(self.base, w) for w in ("research", "train")}

    def _entries(self): return [json.loads(l) for l in open(self.p)] if os.path.isfile(self.p) else []

    def _log(self, kind, payload):
        es = self._entries(); prev = es[-1]["hash"] if es else "0" * 64; e = {"seq": len(es), "kind": kind, "payload": payload, "prev": prev}
        e["hash"] = hashlib.sha256(json.dumps({k: e[k] for k in ("seq", "kind", "payload", "prev")}, sort_keys=True, default=str).encode()).hexdigest()
        open(self.p, "a").write(json.dumps(e, sort_keys=True, default=str) + "\n"); return e

    def note(self, kind, text):
        """kind in hypothesis|result|decision: free-text research log entries (free, not counted against the call budget)."""
        assert kind in ("hypothesis", "result", "decision"); self._log(kind, {"text": str(text)[:2000]})

    def probe(self, op, cfg):
        """teacher-forced change in L1 vs expert (negative = closer to the expert) on the RESEARCH layouts and on TRAIN-layout states, for ONE edit (op may be 'composite')."""
        used = len([e for e in self._entries() if e["kind"] == "probe"])
        if used >= self.max_calls: raise RuntimeError(f"research budget exhausted ({used}/{self.max_calls} probes)")
        out = {"op": op, "cfg": cfg}
        try:
            e = OPS.apply(op, cfg, self.base, ctx={"ref_models": self.refs})
            out["d_research_l1"] = teacher_forced(e, "research") - self.base_tf["research"]; out["d_train_l1"] = teacher_forced(e, "train") - self.base_tf["train"]
        except (ValueError, KeyError, TypeError) as err: out["error"] = str(err)[:200]
        self._log("probe", out); return out
