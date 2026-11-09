"""Read-only, logged, capped inspection of ONE model's weights (the 'peek' tool for helpers/authors).  Same contract as lab3/peek.py.

  p = Peeker(model, out_dir)           # cap: 15 calls per out_dir (counted from out_dir/peek_log.jsonl), output <= 3000 ASCII chars
  p.tensors(pattern=None)              names/shapes of state_dict tensors (substring filter)
  p.slice(name, spec)                  named tensor slice, spec like "0:4,10:20" (<= 400 elements)
  p.svd(name, k=8)                     top-k singular values of a named 2-D tensor (k <= 32)
  p.head(kind, layer)                  per-head QK / OV norms; kind in enc_self | dec_self | dec_cross; layer an int (0..n-1)
  p.neuron(kind, layer, k=8)           per-neuron stats (sparsity/kurtosis/contribution) on fixed TRAIN-layout states; kind in enc | dec; top-k (k <= 64)
  p.attn(kind, layer)                  attention MASS of every head of that layer on self/goal/obstacle tokens (ID and VAL probe states); kind enc_self | dec_cross
CLI: OMP_NUM_THREADS=1 python core/peek.py OUT_DIR [--seed 0] [--ckpt final] CMD [args...]
Safety: names must be existing state_dict keys (no eval/exec; specs parsed by regex), no tensor is ever written, parameters are hashed before
and after each call and a change raises; refused/failed calls are logged but do not count towards the cap; only TRAIN/VAL_SEEN-layout states are used (no HOLD, no CANARY).
"""
import os, sys, json, re, hashlib, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch
import adapter, ops
import dataroom as D

MAX_CALLS, MAX_CHARS, MAX_ELEMS, MAX_K = 15, 3000, 400, 32
LOG = "peek_log.jsonl"
_SPEC = re.compile(r"^\s*-?\d*\s*:\s*-?\d*\s*(,\s*-?\d*\s*:\s*-?\d*\s*)*$|^\s*-?\d+\s*(,\s*(-?\d+|-?\d*\s*:\s*-?\d*)\s*)*$")


class PeekError(ValueError):
    pass


class PeekCapExceeded(PeekError):
    pass


def _ascii_cap(s, cap=MAX_CHARS):
    s = s.encode("ascii", "replace").decode("ascii")
    return s if len(s) <= cap else s[:cap - 24] + "\n...[output truncated]"


def _fp(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def head_weights(model, kind, l, h):
    """per-head W_Q, W_K, W_V as (dh, d) and W_O as (d, dh)."""
    at = ops.attn_module(model, kind, l); H = ops.n_heads_of(model, at); d = at.out.out_features; dh = at.out.in_features // H
    sl = slice(h * dh, (h + 1) * dh)
    if kind == "dec_cross":
        q = at.q.weight.detach()[sl]; kv = at.kv.weight.detach(); k, v = kv[:at.out.in_features][sl], kv[at.out.in_features:][sl]
    else:
        w = at.qkv.weight.detach(); D_ = at.out.in_features; q, k, v = w[:D_][sl], w[D_:2 * D_][sl], w[2 * D_:][sl]
    return q, k, v, at.out.weight.detach()[:, sl]


class Peeker:
    def __init__(self, model, out_dir, max_calls=MAX_CALLS):
        self._m = model; self._dir = out_dir; self._max = max_calls
        os.makedirs(out_dir, exist_ok=True)
        self._log = os.path.join(out_dir, LOG)

    def n_calls(self):
        if not os.path.exists(self._log):
            return 0
        n = 0
        with open(self._log) as f:
            for ln in f:
                try:
                    n += json.loads(ln).get("status") == "ok"
                except ValueError:
                    pass
        return n

    def _append(self, rec):
        with open(self._log, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def _call(self, tool, args, fn):
        used = self.n_calls()
        if used >= self._max:
            self._append({"tool": tool, "args": args, "status": "refused_cap", "t": time.time()})
            raise PeekCapExceeded(f"peek cap reached ({self._max} calls per room)")
        before = _fp(self._m)
        try:
            out = _ascii_cap(fn())
        except Exception as e:
            self._append({"tool": tool, "args": args, "status": "error", "error": f"{type(e).__name__}: {e}"[:300], "t": time.time()})
            raise
        if _fp(self._m) != before:
            raise RuntimeError("model weights changed during peek (read-only violated)")
        self._append({"tool": tool, "args": args, "status": "ok", "chars": len(out), "call_no": used + 1, "t": time.time()})
        return out

    def _sd(self):
        return dict(self._m.state_dict())

    def _tensor(self, name):
        if not isinstance(name, str):
            raise PeekError("tensor name must be a string")
        sd = self._sd()
        if name not in sd:
            raise PeekError(f"unknown tensor {name!r}; use tensors() to list names")
        return sd[name].detach()

    def _kind_layer(self, kind, layer, kinds):
        if kind not in kinds:
            raise PeekError(f"kind must be one of {kinds}")
        if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < ops.n_layers(self._m, kind):
            raise PeekError(f"layer must be an int in 0..{ops.n_layers(self._m, kind) - 1} for kind {kind}")

    # ---- tools
    def tensors(self, pattern=None):
        def fn():
            if pattern is not None and not isinstance(pattern, str):
                raise PeekError("pattern must be a string")
            rows = [f"{k} {list(v.shape)}" for k, v in self._sd().items() if pattern is None or pattern in k]
            return f"{len(rows)} tensors" + (f" matching {pattern!r}" if pattern else "") + "\n" + "\n".join(rows)
        return self._call("tensors", {"pattern": pattern}, fn)

    def slice(self, name, spec):
        def fn():
            t = self._tensor(name)
            if not isinstance(spec, str) or not _SPEC.match(spec):
                raise PeekError("spec must look like '0:4,10:20' (ints/slices only)")
            idx = []
            for part in spec.split(","):
                part = part.strip()
                if ":" in part:
                    a, b = part.split(":"); idx.append(slice(int(a) if a.strip() else None, int(b) if b.strip() else None))
                else:
                    idx.append(int(part))
            if len(idx) > t.dim():
                raise PeekError(f"too many indices for {name} with shape {list(t.shape)}")
            v = t[tuple(idx)]
            if v.numel() > MAX_ELEMS:
                raise PeekError(f"slice has {v.numel()} elements > cap {MAX_ELEMS}")
            return f"{name}{list(t.shape)}[{spec}] -> shape {list(v.shape)}\n" + json.dumps(D._r(v.double().tolist(), 5))
        return self._call("slice", {"name": name, "spec": spec}, fn)

    def svd(self, name, k=8):
        def fn():
            t = self._tensor(name); kk = int(k)
            if t.dim() != 2:
                raise PeekError(f"{name} is not 2-D (shape {list(t.shape)})")
            if not 1 <= kk <= MAX_K:
                raise PeekError(f"k must be in 1..{MAX_K}")
            s = torch.linalg.svdvals(t.double()); e = s ** 2
            return (f"{name}{list(t.shape)} top-{kk} singular values: " + " ".join(f"{float(v):.4g}" for v in s[:kk]) +
                    f"\nenergy share in top-{kk}: {float(e[:kk].sum() / e.sum()):.4f}; fro={float(e.sum().sqrt()):.4g}")
        return self._call("svd", {"name": name, "k": k}, fn)

    def head(self, kind, layer):
        def fn():
            self._kind_layer(kind, layer, ops.HEAD_KINDS)
            H = ops.n_heads_of(self._m, ops.attn_module(self._m, kind, layer)); rows = []
            for h in range(H):
                q, k, v, o = head_weights(self._m, kind, layer, h)
                qk = q.double().T @ k.double(); ov = o.double() @ v.double()
                rows.append(f"{layer}.{h} {qk.norm():.3f} {ov.norm():.3f} {torch.linalg.svdvals(qk)[0]:.3f} {torch.linalg.svdvals(ov)[0]:.3f}")
            return f"{kind} layer {layer}: per-head norms (id QKfro OVfro QKsv1 OVsv1)\n" + "\n".join(rows)
        return self._call("head", {"kind": kind, "layer": layer}, fn)

    def neuron(self, kind, layer, k=8):
        def fn():
            self._kind_layer(kind, layer, ops.MLP_KINDS)
            kk = int(k)
            if not 1 <= kk <= 64:
                raise PeekError("k must be in 1..64")
            mlp = ops.mlp_module(self._m, kind, layer); acts = []
            hk = mlp.fc2.register_forward_pre_hook(lambda mod, i: acts.append(i[0].detach().double()))
            try:
                with torch.no_grad():
                    adapter.forward(self._m, adapter.probe_obs("ID", D.N_PROBE))
            finally:
                hk.remove()
            a = acts[0].reshape(-1, acts[0].shape[-1]); mu = a.mean(0); sd = a.std(0) + 1e-9
            kurt = (((a - mu) / sd) ** 4).mean(0) - 3.0; spars = (a.abs() < 0.05).double().mean(0)
            contrib = mlp.fc2.weight.detach().double().norm(dim=0) * (a ** 2).mean(0).sqrt(); top = contrib.topk(min(kk, len(contrib))).indices.tolist()
            return (f"{kind}{layer} MLP ({len(a)} token states from TRAIN-layout probes): sparsity_mean={float(spars.mean()):.3f} kurt_mean={float(kurt.mean()):.2f}\n"
                    "neuron contrib kurtosis sparsity mean_act\n" + "\n".join(f"{i} {contrib[i]:.3f} {kurt[i]:.2f} {spars[i]:.3f} {mu[i]:+.3f}" for i in top))
        return self._call("neuron", {"kind": kind, "layer": layer, "k": k}, fn)

    def attn(self, kind, layer):
        def fn():
            self._kind_layer(kind, layer, ["enc_self", "dec_cross"])
            hd = D.build_heads(self._m)[kind]; rows = [f"{kind} layer {layer}: attention mass (ID | VAL): id self goal obs nearest-obstacle-share entropy"]
            for h, d in hd.items():
                if h.split(".")[0] == str(layer):
                    f = lambda x: " ".join(f"{x[c]:.2f}" for c in ("self", "goal", "obs", "nn_share", "entropy"))
                    rows.append(f"{h} {f(d['ID'])} | {f(d['VAL'])}")
            return "\n".join(rows)
        return self._call("attn", {"kind": kind, "layer": layer}, fn)


def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ckpt", default="final"); ap.add_argument("--state", default=None)
    ap.add_argument("cmd", choices=["tensors", "slice", "svd", "head", "neuron", "attn"]); ap.add_argument("args", nargs="*")
    a = ap.parse_args(argv)
    m = adapter.load_ckpt(a.seed, "final" if a.ckpt == "final" else int(a.ckpt))
    if a.state:
        m.load_state_dict(torch.load(a.state, map_location="cpu"))
    m.eval()
    p = Peeker(m, a.out_dir); f = getattr(p, a.cmd); x = a.args
    conv = {"tensors": [str], "slice": [str, str], "svd": [str, int], "head": [str, int], "neuron": [str, int, int], "attn": [str, int]}[a.cmd]
    try:
        print(f(*[c(v) for c, v in zip(conv, x)]))
    except PeekError as e:
        print(f"REFUSED: {e}"); return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
