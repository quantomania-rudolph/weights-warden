"""Drop-in hook for ANY PyTorch training loop (behavioural cloning, RL, ...): when a trigger fires, freeze a snapshot, hand it to an AGENT (Fable/Opus + the agent_tools tools, or a script), gate the agent's weight edit
through an EVALUATOR you control, apply it in place, keep training, and AUTOMATICALLY ROLL BACK if the watched metric gets worse afterwards.

    hook = HarnessHook("runs/x/harness", total_steps=100_000, triggers=[AtFraction(0.10), Plateau("val_loss", window=2000), GapAbove("train_loss", "val_loss", ratio=1.5)],
                       agent=my_agent, evaluator=my_evaluator, watch_metric="val_loss", watch_steps=1500, max_interventions=2)
    for step in range(total_steps):
        ... loss.backward(); opt.step() ...
        hook.step(step, {"train_loss": ..., "val_loss": ..., "success": ..., "my_reward": ...}, model, opt)      # call every step with whatever metrics you have (missing ones are skipped)

agent(request_dir, model) -> edited state_dict | None.  request_dir holds pre.pt (weights), curves.json (every metric so far), request.json (reason, step, metric names, the exact protocol the agent must follow).
   A scripted agent is just a function; an LLM agent is `cmd_agent(["python","my_agent.py"])`: it is run with the request dir as argv[1] and must write edit.pt (a state_dict) there.
evaluator(pre_model, edited_model) -> {"accept": bool, ...numbers to log}.  YOU define it (held-out closed-loop success, a reward, a loss on held-out data, ...).  With evaluator=None every edit is REFUSED unless allow_unverified=True.
Every decision is appended to harness/ledger.jsonl as a hash chain (tamper-evident).  Nothing here reads your test/hold-out data: the evaluator you pass decides what is measured."""
import os, json, copy, math, time, hashlib, subprocess, torch


class Trigger:
    name = "trigger"
    def check(self, step, total, hist): return None


class AtFraction(Trigger):
    def __init__(self, frac): self.frac = frac; self.name = f"at_{frac:g}_of_training"; self.done = False
    def check(self, step, total, hist):
        if self.done or step < int(self.frac * total): return None
        self.done = True; return f"reached {self.frac:.0%} of training"      # one-shot


class Plateau(Trigger):
    """metric has not improved by min_delta (relative) over the last `window` points vs the best before that window."""
    def __init__(self, metric, window=1000, min_delta=0.01, mode="min"): self.m, self.w, self.d, self.mode = metric, window, min_delta, mode; self.name = f"plateau_{metric}"
    def check(self, step, total, hist):
        s = [v for _, v in hist.get(self.m, [])]
        if len(s) < 2 * self.w: return None
        best_before, recent = (min(s[:-self.w]), min(s[-self.w:])) if self.mode == "min" else (max(s[:-self.w]), max(s[-self.w:]))
        gain = (best_before - recent) / (abs(best_before) + 1e-12) if self.mode == "min" else (recent - best_before) / (abs(best_before) + 1e-12)
        return f"{self.m} plateau: improvement {gain:.4f} < {self.d} over last {self.w} points" if gain < self.d else None


class GapAbove(Trigger):
    def __init__(self, train_metric, val_metric, ratio=1.5): self.a, self.b, self.r = train_metric, val_metric, ratio; self.name = f"gap_{train_metric}_{val_metric}"
    def check(self, step, total, hist):
        if not hist.get(self.a) or not hist.get(self.b): return None
        a, b = hist[self.a][-1][1], hist[self.b][-1][1]; return f"generalisation gap: {self.b}/{self.a} = {b / (a + 1e-12):.2f} > {self.r}" if b > self.r * a else None


class Threshold(Trigger):
    def __init__(self, metric, value, below=True): self.m, self.v, self.below = metric, value, below; self.name = f"threshold_{metric}"
    def check(self, step, total, hist):
        if not hist.get(self.m): return None
        x = hist[self.m][-1][1]; return f"{self.m}={x:.4g} {'<' if self.below else '>'} {self.v}" if (x < self.v if self.below else x > self.v) else None


class Divergence(Trigger):
    def __init__(self, metric, factor=3.0): self.m, self.f = metric, factor; self.name = f"divergence_{metric}"
    def check(self, step, total, hist):
        s = [v for _, v in hist.get(self.m, [])]; return f"{self.m} diverging: {s[-1]:.4g} > {self.f} x best {min(s):.4g}" if len(s) > 10 and (not math.isfinite(s[-1]) or s[-1] > self.f * min(s)) else None


class Custom(Trigger):
    def __init__(self, fn, name="custom"): self.fn, self.name = fn, name
    def check(self, step, total, hist): return self.fn(step, total, hist)


def cmd_agent(cmd, timeout=3600):
    """Wrap an external agent (e.g. a Claude/Fable session driven by agent/PROMPT.md): `cmd + [request_dir]` must write request_dir/edit.pt."""
    def run(request_dir, model):
        subprocess.run(list(cmd) + [request_dir], timeout=timeout, check=False); p = os.path.join(request_dir, "edit.pt")
        return torch.load(p, map_location="cpu") if os.path.exists(p) else None
    return run


class HarnessHook:
    def __init__(self, out_dir, total_steps, triggers, agent, evaluator=None, watch_metric=None, watch_mode="min", watch_steps=1000, watch_tolerance=0.02, max_interventions=2, cooldown_steps=2000,
                 reset_optimizer_state=True, allow_unverified=False, protocol_path=None):
        self.dir, self.total, self.triggers, self.agent, self.ev = out_dir, total_steps, triggers, agent, evaluator
        self.wm, self.wmode, self.ws, self.wtol, self.maxi, self.cool, self.reset, self.unver = watch_metric, watch_mode, watch_steps, watch_tolerance, max_interventions, cooldown_steps, reset_optimizer_state, allow_unverified
        self.hist, self.n, self.last_end, self.pending, self.protocol = {}, 0, -10 ** 9, None, protocol_path
        os.makedirs(out_dir, exist_ok=True); self.ledger = os.path.join(out_dir, "ledger.jsonl")

    def _log(self, kind, payload):
        prev = "0" * 64
        if os.path.exists(self.ledger): prev = json.loads(open(self.ledger).read().strip().splitlines()[-1])["sha"]
        rec = {"t": time.time(), "kind": kind, "payload": payload, "prev": prev}; rec["sha"] = hashlib.sha256(json.dumps({k: rec[k] for k in ("t", "kind", "payload", "prev")}, sort_keys=True, default=str).encode()).hexdigest()
        open(self.ledger, "a").write(json.dumps(rec, default=str) + "\n")

    def revert(self, model, n=None, optimizer=None):
        """undo an applied edit by hand (n = intervention number, default the latest): reload its pre.pt snapshot. Logged. Returns the number reverted."""
        n = n or self.n; p = os.path.join(self.dir, f"intervention_{n:02d}", "pre.pt")
        if not os.path.exists(p): raise FileNotFoundError(p)
        model.load_state_dict(torch.load(p)); self.pending = None; self._log("manual_revert", {"n": n}); return n

    def _best(self, xs): return min(xs) if self.wmode == "min" else max(xs)

    def step(self, step, metrics, model, optimizer=None):
        for k, v in metrics.items():
            if v is not None and (not isinstance(v, float) or True): self.hist.setdefault(k, []).append((step, float(v)))
        changed = False
        if self.pending and step >= self.pending["check_at"] and self.wm in self.hist:                              # ---- post-edit safety: roll back if the watched metric got worse
            recent = [v for s, v in self.hist[self.wm] if s > self.pending["step"]][-max(3, self.ws // 100):]
            if recent:
                now = sum(recent) / len(recent); base = self.pending["pre_value"]; worse = (now > base * (1 + self.wtol)) if self.wmode == "min" else (now < base * (1 - self.wtol))
                if worse: model.load_state_dict(torch.load(self.pending["pre"])); changed = True
                self._log("rollback" if worse else "kept", {"step": step, "watch_metric": self.wm, "pre": base, "after": now}); self.pending = None; self.last_end = step
        if self.pending or self.n >= self.maxi or step - self.last_end < self.cool: return changed
        for t in self.triggers:
            reason = t.check(step, self.total, self.hist)
            if reason: return self._intervene(step, t.name, reason, model, optimizer) or changed
        return changed

    def _intervene(self, step, trig, reason, model, optimizer):
        self.n += 1; d = os.path.join(self.dir, f"intervention_{self.n:02d}"); os.makedirs(d, exist_ok=True); pre = os.path.join(d, "pre.pt"); torch.save(model.state_dict(), pre)
        json.dump({k: v for k, v in self.hist.items()}, open(os.path.join(d, "curves.json"), "w"))
        json.dump({"step": step, "total_steps": self.total, "trigger": trig, "reason": reason, "metrics": sorted(self.hist), "protocol": self.protocol or "docs/PROTOCOL_HARNESS_ADOPTION.md + agent/PROMPT.md"}, open(os.path.join(d, "request.json"), "w"), indent=1)
        self._log("trigger", {"n": self.n, "step": step, "trigger": trig, "reason": reason, "pre_sha": hashlib.sha256(open(pre, "rb").read()).hexdigest()})
        try: sd = self.agent(d, model)
        except Exception as e: self._log("agent_error", {"n": self.n, "error": str(e)[:300]}); self.last_end = step; return False
        if sd is None: self._log("no_edit", {"n": self.n}); self.last_end = step; return False
        edited = copy.deepcopy(model); edited.load_state_dict(sd)
        if not all(bool(torch.isfinite(p).all()) for p in edited.parameters()): self._log("rejected", {"n": self.n, "why": "non-finite weights"}); self.last_end = step; return False
        if self.ev is None and not self.unver: self._log("rejected", {"n": self.n, "why": "no evaluator and allow_unverified=False"}); self.last_end = step; return False
        rep = self.ev(model, edited) if self.ev else {"accept": True, "unverified": True}
        self._log("evaluated", {"n": self.n, **{k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in rep.items()}})
        if not rep.get("accept"): self.last_end = step; return False
        model.load_state_dict(sd)
        if optimizer is not None and self.reset:
            for st in optimizer.state.values():
                for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                    if k in st: st[k].zero_()
        self.pending = {"step": step, "check_at": step + self.ws, "pre": pre, "pre_value": (sum(v for _, v in self.hist[self.wm][-5:]) / len(self.hist[self.wm][-5:])) if self.wm in self.hist else None}
        if self.wm not in self.hist: self.pending = None
        self._log("applied", {"n": self.n, "step": step, "watching": self.wm}); return True
