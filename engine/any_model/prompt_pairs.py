"""Clean/corrupt prompt PAIRS: build, quality-check and batch.  An agent writes 5-10 pairs per behaviour (JSON), this module tells it which are usable.
pair = {"name", "clean": str|list[int], "corrupt": str|list[int], "correct": str|int, "wrong": str|int}  (strings need a tokenizer; the answer tokens are single tokens).
QUALITY RULES (a pair that fails is dropped with a reason): equal token length; answer tokens differ; the MODEL prefers `correct` on the clean prompt by >= min_margin; the corrupt prompt reduces that preference by >= min_drop (otherwise it does not isolate the behaviour)."""
import torch, patching as P


def encode(pairs, tokenizer=None):
    out = []
    for p in pairs:
        f = lambda x: torch.tensor(x if isinstance(x, list) else tokenizer.encode(x), dtype=torch.long)
        g = lambda x: int(x) if isinstance(x, int) else tokenizer.encode(x)[-1] if len(tokenizer.encode(x)) >= 1 else None
        out.append({"name": p.get("name", "pair"), "clean": f(p["clean"]), "corrupt": f(p["corrupt"]), "correct": g(p["correct"]), "wrong": g(p["wrong"])})
    return out


@torch.no_grad()
def check(model, enc_pairs, min_margin=0.5, min_drop=0.5):
    """returns (usable_pairs, report).  report rows: name, ok, reason, clean_diff, corrupt_diff."""
    keep, rep = [], []
    for p in enc_pairs:
        row = {"name": p["name"]}
        if p["clean"].shape != p["corrupt"].shape: row.update(ok=False, reason=f"token length differs {tuple(p['clean'].shape)} vs {tuple(p['corrupt'].shape)}")
        elif p["correct"] == p["wrong"]: row.update(ok=False, reason="correct == wrong")
        else:
            f = P.logit_diff([p["correct"]], [p["wrong"]]); cd, kd = float(f(model(p["clean"][None]))[0]), float(f(model(p["corrupt"][None]))[0]); row.update(clean_diff=round(cd, 3), corrupt_diff=round(kd, 3))
            if cd < min_margin: row.update(ok=False, reason=f"model does not prefer the correct token on the clean prompt (margin {cd:.2f} < {min_margin})")
            elif cd - kd < min_drop: row.update(ok=False, reason=f"corruption barely changes the preference (drop {cd - kd:.2f} < {min_drop})")
            else: row.update(ok=True, reason="")
        rep.append(row)
        if row["ok"]: keep.append(p)
    return keep, rep


def batch(pairs):
    """stack usable pairs of equal length into (clean (B,T), corrupt (B,T), metric_fn)"""
    T = {p["clean"].shape[0] for p in pairs}
    if len(T) != 1: raise ValueError(f"pairs must share one token length to batch; got {sorted(T)} (group by length)")
    return torch.stack([p["clean"] for p in pairs]), torch.stack([p["corrupt"] for p in pairs]), P.logit_diff([p["correct"] for p in pairs], [p["wrong"] for p in pairs])
