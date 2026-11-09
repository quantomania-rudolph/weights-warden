"""Clean/corrupt-prompt patching for ANY model discovered by structure.py.
NODES: attention heads (slice of the o_proj input), MLP hidden units (down_proj input); for plain MLPs, hidden units.
 node_scores  : attribution patching (first order): effect = sum (a_corrupt - a_clean) . dmetric/da  = predicted metric change if the corrupt activation is patched into the clean run.  importance = -effect.
 exact_*      : real patching (re-run with the corrupt activation inserted) - the oracle used to validate the first-order numbers and to measure circuit faithfulness.
EDGES (transformer): writer (embed, each head, each MLP) -> reader (each attn-input, mlp-input, final norm).  edge_scores = first-order EAP using the gradient at the READER's input;
 exact_edge_effect = add (w_corrupt - w_clean) at that reader's input only (true direct-path patching).
metric_fn(output) -> (B,) tensor; default logit difference at the last position between answer and distractor token ids.  Clean and corrupt inputs must have the same shape."""
import contextlib, torch, torch.nn as nn
import structure as S


def logit_diff(correct, wrong):
    c, w = torch.as_tensor(correct), torch.as_tensor(wrong)
    return lambda out: (out.logits if hasattr(out, "logits") else out)[torch.arange(len(c)), -1, c] - (out.logits if hasattr(out, "logits") else out)[torch.arange(len(c)), -1, w]


class _Rec:
    """hooks that record (and optionally retain grad of) the input of every node site, and the writer/reader tensors for edges."""
    def __init__(self, model, tm, grad=False, edges=False, bt=None):
        self.tm, self.grad, self.acts, self.read, self.write, self.h, self.bt = tm, grad, {}, {}, {}, [], bt
        if tm.kind == "mlp":
            for i in range(tm.n_hidden): self._pre(tm.lin[i + 1], ("unit", i))
            return
        for l, b in enumerate(tm.blocks):
            self._pre(b.o, ("head", l)); self._pre(b.down, ("neuron", l))
            if edges:
                self._pre(b.attn_norm, ("r", "attn_in", l), read=True); self._pre(b.mlp_norm, ("r", "mlp_in", l), read=True)
                self.h.append(b.down.register_forward_hook(lambda m, a, o, l=l: self.write.__setitem__(("mlp", l), (o[0] if isinstance(o, tuple) else o))))      # MLP output = the down-projection's output (works whether or not the model has an `mlp` submodule)
        if edges:
            self._pre(tm.final_norm, ("r", "final", 0), read=True)
            self.h.append(tm.blocks[0].mod.register_forward_pre_hook(lambda m, a, kw: self.write.__setitem__(("embed", 0), (a[0] if a else kw["hidden_states"])), with_kwargs=True))
    def _pre(self, mod, key, read=False):
        def f(m, args):
            x, new = args[0], None
            if self.grad and x.requires_grad:
                if read: x = x.clone(); new = (x,) + tuple(args[1:])      # each reader gets its OWN copy so its gradient is the direct one (the residual also feeds the skip path and later readers)
                x.retain_grad()
            (self.read if read else self.acts)[key] = x
            return new
        self.h.append(mod.register_forward_pre_hook(f))
    def tok(self, x):
        """restore the (B, T, ...) token axis: some blocks (e.g. OPT) flatten (B,T,d) -> (B*T,d) around the MLP/norms, and their residual add needs the flat shape, so we never change what the model sees - only how we READ it."""
        return x.reshape(*self.bt, x.shape[-1]) if (self.bt is not None and x.ndim == 2 and x.shape[0] == self.bt[0] * self.bt[1]) else x
    def A(self, k): return self.tok(self.acts[k])
    def R(self, k): return self.tok(self.read[k])
    def W(self, k): return self.tok(self.write[k])
    def GA(self, k): return self.tok(self.acts[k].grad)
    def GR(self, k): return self.tok(self.read[k].grad)

    def close(self):
        for h in self.h: h.remove()


@contextlib.contextmanager
def _grad_on(model):
    old = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in old: p.requires_grad_(True)
    try: yield
    finally:
        for p, r in old: p.requires_grad_(r)


def _run(model, tm, inp, grad, edges=False, metric_fn=None):
    rec = _Rec(model, tm, grad, edges, bt=tuple(inp.shape[:2]) if tm.kind == "transformer" else None)
    try:
        with (torch.enable_grad() if grad else torch.no_grad()):
            out = model(inp); m = metric_fn(out) if metric_fn else None
            if grad: m.sum().backward()
    finally: rec.close()
    return rec, m


def _shape(tm, key, a):
    if key[0] == "head": return a.reshape(*a.shape[:-1], tm.n_heads, tm.head_dim)
    if key[0] == "unit": return a.unsqueeze(1) if a.ndim == 2 else a
    return a


def node_scores(model, tm, clean, corrupt, metric_fn):
    """returns {'effect': {key: (B, n_units)}, 'clean_metric': (B,), 'corrupt_metric': (B,)}; keys ('head', layer) / ('neuron', layer) / ('unit', layer)."""
    model.zero_grad(set_to_none=True)
    with _grad_on(model):
        rc, _ = _run(model, tm, corrupt, False); rec, m = _run(model, tm, clean, True, metric_fn=metric_fn)
        with torch.no_grad(): cm = metric_fn(model(corrupt)).detach()
        eff = {}
        for k in rec.acts:
            e = (_shape(tm, k, rc.A(k).detach()) - _shape(tm, k, rec.A(k).detach())) * _shape(tm, k, rec.GA(k))
            eff[k] = (e.sum(dim=(1, 3)) if k[0] == "head" else e.sum(dim=1)).detach()
    return {"effect": eff, "clean_metric": m.detach(), "corrupt_metric": cm}


def _patch_hooks(model, tm, corrupt_acts, nodes, bt=None):
    """pre-hooks inserting the corrupt activation for the listed nodes: nodes = {key: list of unit indices (head index / neuron index)}."""
    hs = []
    for key, units in nodes.items():
        mod = tm.lin[key[1] + 1] if key[0] == "unit" else (tm.blocks[key[1]].o if key[0] == "head" else tm.blocks[key[1]].down)
        def f(m, args, key=key, units=tuple(units)):
            x = args[0].clone(); c = corrupt_acts[key]; shp = x.shape
            if x.ndim == 2 and c.ndim == 3 and x.shape[0] == c.shape[0] * c.shape[1]: x = x.reshape(c.shape[0], c.shape[1], x.shape[-1])      # token axis restored from the stored activation's shape
            if key[0] == "head":
                xv = x.view(*x.shape[:-1], tm.n_heads, tm.head_dim); xv[..., list(units), :] = c.view(*c.shape[:-1], tm.n_heads, tm.head_dim)[..., list(units), :]
            else: x[..., list(units)] = c[..., list(units)]
            return (x.reshape(shp),) + tuple(args[1:])
        hs.append(mod.register_forward_pre_hook(f))
    return hs


def corrupt_acts_of(model, tm, corrupt):
    rc, _ = _run(model, tm, corrupt, False); return {k: rc.A(k).detach() for k in rc.acts}


@torch.no_grad()
def patched_metric(model, tm, clean, corrupt_acts, nodes, metric_fn):
    hs = _patch_hooks(model, tm, corrupt_acts, nodes, bt=tuple(clean.shape[:2]) if tm.kind == "transformer" else None)
    try: return metric_fn(model(clean)).detach()
    finally:
        for h in hs: h.remove()


def exact_node_effect(model, tm, clean, corrupt_acts, key, unit, metric_fn):
    """true metric change (per example) from patching ONE node with its corrupt activation."""
    return patched_metric(model, tm, clean, corrupt_acts, {key: [unit]}, metric_fn) - metric_fn(model(clean)).detach()


# ------------------------------------------------------------------ edges
def _wstack(tm, b):
    W = b.o.weight.detach(); H, dh = tm.n_heads, tm.head_dim
    return (W.reshape(H, dh, -1) if b.o_conv else W.T.reshape(H, dh, -1))                # (H, dh, d_model)


def _writers(tm, rec):
    """{('embed',0): (B,T,d), ('head',l): (B,T,H,d), ('mlp',l): (B,T,d)} from a recording"""
    w = {("embed", 0): rec.W(("embed", 0)).detach(), **{("mlp", l): rec.W(("mlp", l)).detach() for l in range(tm.n_layers)}}
    for l, b in enumerate(tm.blocks):
        x = rec.A(("head", l)).detach(); x = x.reshape(*x.shape[:-1], tm.n_heads, tm.head_dim); w[("head", l)] = torch.einsum("bthk,hkd->bthd", x, _wstack(tm, b))
    return w


def _feeds(wk, rk):
    """does writer wk feed reader rk (layer ordering)?"""
    kind, l = wk; rkind, rl = rk[1], rk[2]
    if kind == "embed": return True
    if rkind == "final": return True
    if kind == "head": return rl > l if rkind == "attn_in" else rl >= l
    return rl > l                                                                       # mlp l feeds attn_in/mlp_in of later layers


def edge_scores(model, tm, clean, corrupt, metric_fn):
    """first-order EAP. returns {(writer_key, reader_key): (B,) or (B,H) tensor of predicted metric change}, writer keys ('embed',0) ('head',l) ('mlp',l)."""
    with _grad_on(model):
        model.zero_grad(set_to_none=True); rc, _ = _run(model, tm, corrupt, False, edges=True); rec, m = _run(model, tm, clean, True, edges=True, metric_fn=metric_fn)
    wc, wk = _writers(tm, rc), _writers(tm, rec); out = {}
    for rk in rec.read:
        g = rec.GR(rk)
        for k in wc:
            if not _feeds(k, rk): continue
            d = wc[k] - wk[k]
            out[(k, rk)] = torch.einsum("bthd,btd->bh", d, g) if k[0] == "head" else torch.einsum("btd,btd->b", d, g)
    return out


@torch.no_grad()
def exact_edge_effect(model, tm, clean, corrupt, wkey, rkey, metric_fn, head=None, scale=1.0):
    """add scale*(w_corrupt - w_clean) at ONE reader's input only (direct-path patch); returns metric change per example."""
    rc, _ = _run(model, tm, corrupt, False, edges=True); rcl, _ = _run(model, tm, clean, False, edges=True); wc, wl = _writers(tm, rc), _writers(tm, rcl); d = wc[wkey] - wl[wkey]
    if wkey[0] == "head": d = d[..., head, :]
    mod = {"attn_in": lambda l: tm.blocks[l].attn_norm, "mlp_in": lambda l: tm.blocks[l].mlp_norm, "final": lambda l: tm.final_norm}[rkey[1]](rkey[2])
    h = mod.register_forward_pre_hook(lambda m, a: (a[0] + scale * d.reshape(a[0].shape),) + tuple(a[1:]))      # add in the reader's OWN shape (never change what the model sees)
    try: pert = metric_fn(model(clean)).detach()
    finally: h.remove()
    return pert - metric_fn(model(clean)).detach()


@torch.no_grad()
def exact_node_scores(model, tm, clean, corrupt, metric_fn, atp=None, k_neurons=256, all_units_below=4096):
    """EXACT patching of every attention head (always) and of MLP units: all of them if there are <= all_units_below per layer-set, else the top-k by |first-order effect| (needs `atp` from node_scores).
    Why: first-order attribution UNDER-estimates saturated components (a head whose softmax is saturated has ~0 gradient at the clean point but a large effect when patched); final circuit membership must come from exact patching.
    returns the same structure as node_scores() but with exact per-example effects (units that were not evaluated have effect 0 and are listed in 'not_evaluated')."""
    ca = corrupt_acts_of(model, tm, corrupt); base = metric_fn(model(clean)).detach(); cm = metric_fn(model(corrupt)).detach(); eff, skipped = {}, 0
    if tm.kind == "transformer":
        for l in range(tm.n_layers): eff[("head", l)] = torch.stack([patched_metric(model, tm, clean, ca, {("head", l): [h]}, metric_fn) - base for h in range(tm.n_heads)], 1)
        unit_keys = [("neuron", l) for l in range(tm.n_layers)]; width = tm.d_ff
    else:
        unit_keys = [("unit", l) for l in range(tm.n_hidden)]; width = None
    for k in unit_keys:
        w = ca[k].shape[-1]; e = torch.zeros(base.shape[0], w); eff[k] = e
    total = sum(ca[k].shape[-1] for k in unit_keys)
    if total <= all_units_below: todo = [(k, i) for k in unit_keys for i in range(ca[k].shape[-1])]
    else:
        if atp is None: raise ValueError("too many units to patch exactly; pass atp=node_scores(...) so the top-k can be shortlisted")
        flat = sorted(((float(atp["effect"][k][:, i].mean().abs()), k, i) for k in unit_keys for i in range(ca[k].shape[-1])), key=lambda r: -r[0])[:k_neurons]; todo = [(k, i) for _, k, i in flat]; skipped = total - len(todo)
    for k, i in todo: eff[k][:, i] = patched_metric(model, tm, clean, ca, {k: [i]}, metric_fn) - base
    return {"effect": eff, "clean_metric": base, "corrupt_metric": cm, "not_evaluated_units": skipped}
