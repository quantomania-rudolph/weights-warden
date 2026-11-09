"""Data room for the core Nav2D ACT-like policy: compact, text-only analysis digests of ONE model, for LLM helpers to read.

build_dataroom(model, out_dir, ref_models=None, seed=0, n_ablate=200) writes
  spectroscopy.json heads.json ablation.json attribution.json delta.json     (full numbers)
  digests/{spectroscopy,heads,ablation,attribution,delta}.txt               (<= 4000 ASCII chars each)
  MANIFEST.json                                                             (sha256 of every other file)
Deterministic given (weights, ref_models, seed, backend).  HOLD- and CANARY-free: only TRAIN-layout (ID_EVAL), VAL_SEEN layouts and the fixed probe states are used.
Contents
  spectroscopy : per-matrix SVD stats (effective rank, energy in top directions)
  heads        : per attention head, attention MASS on self / goal / obstacle tokens and the share of obstacle mass on the obstacle NEAREST to the agent,
                 on ID states (TRAIN layouts) and VAL states (VAL_SEEN layouts); decoder self-attention locality/entropy
  ablation     : per-component zero-ablation (each head, each MLP block, each token-embedding type): paired change of closed-loop success on VAL_SEEN and on ID_EVAL
                 (components whose removal hurts ID but not VAL, or helps VAL, are listed first-class)
  attribution  : Jacobian sensitivity of the first action to self / goal / obstacle input features, plus token-shuffle sensitivity (ID vs VAL), expert L1 loss gap
  delta        : relative weight change vs each earlier checkpoint, and each checkpoint's VAL_SEEN / ID_EVAL success (n_ablate episodes)
build_wrong_room(...) builds the identical format for a DIFFERENT model (the seed-1 control).
ASSUMED token order (policy.py): [self, goal, obstacle x5]; attention-pattern capture re-computes softmax(QK^T/sqrt(dh)) from forward hooks on the qkv / q / kv
Linear modules (layout chunk(3)/(2) over the feature dim, then heads), so it works for any model with the SPEC module names.
"""
import os, sys, json, copy, hashlib, math
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np
import torch
import adapter, ops, rewards as RW

torch.set_num_threads(1)
MAX_CHARS = 4000
FILES = ("spectroscopy", "heads", "ablation", "attribution", "delta")
N_PROBE = 256
ABL_SEEDS = {"VAL_SEEN": 7101, "ID_EVAL": 7202}
N_OBS = 5
PFX = {"enc_self": "E", "dec_self": "S", "dec_cross": "X"}


# ------------------------------------------------------------------ small utils
def _r(o, nd=6):
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            return None
        return round(o, nd) + 0.0
    if isinstance(o, dict):
        return {str(k): _r(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_r(v, nd) for v in o]
    if isinstance(o, (np.floating,)):
        return _r(float(o), nd)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def _ascii(s):
    return s.encode("ascii", "replace").decode("ascii")


def fit_text(lines, limit=MAX_CHARS):
    out, n = [], 0
    for i, ln in enumerate(lines):
        ln = _ascii(ln)
        if n + len(ln) + 1 > limit - 40:
            out.append(f"...[{len(lines) - i} lines truncated]")
            break
        out.append(ln); n += len(ln) + 1
    return "\n".join(out)[:limit]


def _work_copy(model):
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ------------------------------------------------------------------ spectroscopy
def spectro_stats(w, k=(1, 4, 16)):
    s = torch.linalg.svdvals(w.detach().double())
    e = s ** 2; tot = float(e.sum()) + 1e-30
    p = s / (float(s.sum()) + 1e-30)
    erank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
    pr = float(tot ** 2 / ((e ** 2).sum() + 1e-30))
    out = {"shape": list(w.shape), "fro": math.sqrt(tot), "top3_sv": [float(v) for v in s[:3]], "eff_rank": erank, "pr": pr}
    for kk in k:
        out[f"energy_top{kk}"] = float(e[:kk].sum() / tot)
    return out


def build_spectroscopy(model):
    return {n: spectro_stats(w) for n, w in ops.matrix_table(model).items()}


def digest_spectroscopy(sp):
    L = ["spectroscopy: per-matrix SVD (names as in the op listing; shape=[out,in]; fro=Frobenius norm; sv1..3=top singular values;",
         "  erank=exp-entropy effective rank; pr=participation ratio of sv^2; E1/E4/E16=share of squared-sv energy in top-1/4/16 directions)",
         "name        shape       fro    sv1    sv2    sv3  erank    pr   E1   E4  E16"]
    for n, s in sp.items():
        t = s["top3_sv"] + [0] * 3
        L.append(f"{n:11s} {str(tuple(s['shape'])):11s} {s['fro']:5.2f} {t[0]:6.2f} {t[1]:6.2f} {t[2]:6.2f} {s['eff_rank']:5.1f} {s['pr']:5.1f} "
                 f"{s['energy_top1']:.2f} {s['energy_top4']:.2f} {s['energy_top16']:.2f}")
    lo = sorted(sp, key=lambda n: sp[n]["eff_rank"] / min(sp[n]["shape"]))[:4]
    L.append("lowest erank/min(dim): " + " ".join(f"{n}:{sp[n]['eff_rank'] / min(sp[n]['shape']):.2f}" for n in lo))
    return L


# ------------------------------------------------------------------ attention capture
def _heads(x, h):
    B, L, D = x.shape
    return x.view(B, L, h, D // h).transpose(1, 2)


@torch.no_grad()
def attention_patterns(model, obs):
    """{('enc_self',l): (B,H,7,7), ('dec_self',l): (B,H,8,8), ('dec_cross',l): (B,H,8,7)} recomputed from hooked qkv/q/kv outputs."""
    m = _work_copy(model); cap = {}; hooks = []
    def grab(key):
        return lambda mod, i, o: cap.__setitem__(key, o.detach())
    for l, lay in enumerate(m.enc_layers):
        hooks.append(lay.attn.qkv.register_forward_hook(grab(("enc_qkv", l))))
    for l, lay in enumerate(m.dec_layers):
        hooks += [lay.self_attn.qkv.register_forward_hook(grab(("dec_qkv", l))), lay.cross_attn.q.register_forward_hook(grab(("dec_q", l))),
                  lay.cross_attn.kv.register_forward_hook(grab(("dec_kv", l)))]
    try:
        adapter.forward(m, obs)
    finally:
        for h in hooks: h.remove()
    out = {}
    def sm(q, k):
        return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
    for l, lay in enumerate(m.enc_layers):
        H = ops.n_heads_of(m, lay.attn); q, k, _ = (_heads(t, H) for t in cap[("enc_qkv", l)].chunk(3, -1)); out[("enc_self", l)] = sm(q, k)
    for l, lay in enumerate(m.dec_layers):
        H = ops.n_heads_of(m, lay.self_attn); q, k, _ = (_heads(t, H) for t in cap[("dec_qkv", l)].chunk(3, -1)); out[("dec_self", l)] = sm(q, k)
        Hc = ops.n_heads_of(m, lay.cross_attn); q = _heads(cap[("dec_q", l)], Hc); k, _ = (_heads(t, Hc) for t in cap[("dec_kv", l)].chunk(2, -1)); out[("dec_cross", l)] = sm(q, k)
    return out


def nearest_obstacle(obs):
    """index (0..4) of the obstacle with the smallest surface distance to the agent, per sample."""
    o = obs.double(); pos = o[:, 0:2]; lay = o[:, 6:].reshape(-1, N_OBS, 3)
    return ((pos[:, None, :] - lay[:, :, :2]).norm(dim=-1) - lay[:, :, 2]).argmin(1)


def _mass(a, nn_idx):
    """a (B,H,Q,K) with K=7 tokens [self,goal,obs0..4] -> per-head dict of averages."""
    B, H, Q, K = a.shape
    ms, mg, mo = a[..., 0].mean((0, 2)), a[..., 1].mean((0, 2)), a[..., 2:].sum(-1).mean((0, 2))
    obs_a = a[..., 2:]                                                           # (B,H,Q,5)
    nn_share = (obs_a.gather(-1, nn_idx[:, None, None, None].expand(B, H, Q, 1))[..., 0] / obs_a.sum(-1).clamp_min(1e-9)).mean((0, 2))
    sq = a[:, :, 0]                                                              # the SELF token's attention row (or query 0 of the decoder)
    sq_o = sq[..., 2:].sum(-1).mean(0)
    ent = -(a * a.clamp_min(1e-12).log()).sum(-1).mean((0, 2)) / math.log(K)
    return {"self": ms, "goal": mg, "obs": mo, "nn_share": nn_share, "q0_obs": sq_o, "entropy": ent}


@torch.no_grad()
def build_heads(model, n=N_PROBE):
    out = {"enc_self": {}, "dec_cross": {}, "dec_self": {}}
    for kind in ("ID", "VAL"):
        obs = adapter.probe_obs(kind, n); nn_idx = nearest_obstacle(obs)
        pats = attention_patterns(model, obs)
        for (k, l), a in pats.items():
            if k == "dec_self":
                Q = a.shape[-1]
                d = {"entropy": (-(a * a.clamp_min(1e-12).log()).sum(-1).mean((0, 2)) / math.log(Q)), "diag": torch.diagonal(a, dim1=-2, dim2=-1).mean((0, 2)),
                     "first": a[..., 0].mean((0, 2))}
            else:
                d = _mass(a, nn_idx)
            for h in range(a.shape[1]):
                out[k].setdefault(f"{l}.{h}", {})[kind] = {n_: float(v[h]) for n_, v in d.items()}
    return out


def digest_heads(hd):
    L = ["per-head attention MASS (probe batch of expert-visited states: ID = TRAIN layouts, VAL = VAL_SEEN layouts). token order: self, goal, 5 obstacles.",
         "  s/g/o = mass on self/goal/obstacle tokens averaged over all queries (sum 1); nn = share of the obstacle mass on the obstacle NEAREST to the agent",
         "  (uniform = 0.20); q0o = obstacle mass of query 0 (encoder: the self token's row); H = normalised entropy (1 = uniform). id = layer.head",
         "ENCODER self-attention            ID: s    g    o   nn  q0o  H  |  VAL: s    g    o   nn  q0o  H"]
    def row(k, d):
        return " ".join(f"{d[c]:.2f}" for c in ("self", "goal", "obs", "nn_share", "q0_obs", "entropy"))
    for h, d in hd["enc_self"].items():
        L.append(f"  E{h:5s}  {row('e', d['ID'])}  |  {row('e', d['VAL'])}")
    L.append("DECODER cross-attention (action queries -> scene tokens)")
    for h, d in hd["dec_cross"].items():
        L.append(f"  X{h:5s}  {row('x', d['ID'])}  |  {row('x', d['VAL'])}")
    L.append("DECODER self-attention among the 8 action queries (ID/VAL): entropy, mass on own position, mass on query 0")
    for h, d in hd["dec_self"].items():
        L.append(f"  S{h:5s}  H={d['ID']['entropy']:.2f}/{d['VAL']['entropy']:.2f} diag={d['ID']['diag']:.2f}/{d['VAL']['diag']:.2f} q0={d['ID']['first']:.2f}/{d['VAL']['first']:.2f}")
    return L


# ------------------------------------------------------------------ ablation (closed loop, paired)
def components(model):
    comps = []
    for kind in ops.HEAD_KINDS:
        for l in range(ops.n_layers(model, kind)):
            for h in range(ops.n_heads_of(model, ops.attn_module(model, kind, l))):
                comps.append((f"{PFX[kind]}{l}.{h}", "scale_head", {"kind": kind, "layer": l, "head": h, "gamma": 0.0}))
    for kind in ops.MLP_KINDS:
        for l in range(ops.n_layers(model, kind)):
            nf = ops.mlp_module(model, kind, l).fc2.in_features
            comps.append((f"{'E' if kind == 'enc' else 'D'}{l}.mlp", "scale_neurons", {"kind": kind, "layer": l, "neurons": list(range(nf)), "gamma": 0.0}))
    for t in ops.TOKENS:
        comps.append((f"tok_{t}", "scale_token_embed", {"which": t, "gamma": 0.0}))
    return comps


def build_ablation(model, n=200, cache=None, deadline=None):
    """success on VAL_SEEN and ID_EVAL (n episodes each, fixed seeds => paired) of the model with each component zeroed.  Resumable via `cache` (json path);
    `deadline` (time.time() value) stops early and returns {'complete': False, ...}."""
    import time
    c = {}
    if cache and os.path.exists(cache):
        c = json.load(open(cache))
    def sc(m, split): return adapter.evaluate(m, split, n, ABL_SEEDS[split])
    if "__base__" not in c:
        rv, ri = sc(model, "VAL_SEEN"), sc(model, "ID_EVAL")
        c["__base__"] = {"VAL_SEEN": float(rv["success"]), "ID_EVAL": float(ri["success"]), "VAL_dist": float(rv["final_dist"]), "ID_dist": float(ri["final_dist"])}
    complete = True
    for name, op, cfg in components(model):
        if name in c:
            continue
        if deadline and time.time() > deadline:
            complete = False; break
        e = ops.apply(op, cfg, model); rv, ri = sc(e, "VAL_SEEN"), sc(e, "ID_EVAL")
        c[name] = {"VAL_SEEN": float(rv["success"]), "ID_EVAL": float(ri["success"]), "VAL_dist": float(rv["final_dist"]), "ID_dist": float(ri["final_dist"])}
        if cache: json.dump(c, open(cache, "w"))
    b = c["__base__"]
    rows = {k: {"d_val": v["VAL_SEEN"] - b["VAL_SEEN"], "d_id": v["ID_EVAL"] - b["ID_EVAL"], "d_val_dist": v["VAL_dist"] - b["VAL_dist"], "d_id_dist": v["ID_dist"] - b["ID_dist"]}
            for k, v in c.items() if k != "__base__"}
    return {"n_episodes": n, "base": b, "components": rows, "complete": complete}


def digest_ablation(ab):
    b, rows, n = ab["base"], ab["components"], ab["n_episodes"]
    L = [f"zero-ablation of each component (head: its out-proj columns x0; mlp: fc2 x0; tok_*: that token's embedding x0), closed loop, n={n} episodes per split, paired seeds.",
         f"base success: VAL_SEEN={b['VAL_SEEN']:.3f} ID_EVAL={b['ID_EVAL']:.3f}; entries are CHANGE in success points (x100) after removal; dist = change in mean final distance (x100). se ~ +-3 pts.",
         "ids: E=encoder self-attn head, S=decoder self-attn head, X=decoder cross-attn head, Ei.mlp/Di.mlp=MLP block, layer.head"]
    if not ab["complete"]:
        L.append("(INCOMPLETE: not every component was evaluated)")
    items = sorted(rows.items(), key=lambda kv: -(abs(kv[1]["d_val"]) + abs(kv[1]["d_id"])))
    L.append("id        dVAL   dID  dVALdist dIDdist   class")
    for k, r in items[:34]:
        cl = []
        if r["d_val"] >= 0.03: cl.append("removal-HELPS-VAL")
        if r["d_id"] <= -0.05 and r["d_val"] > -0.02: cl.append("ID-only (memorisation candidate)")
        if r["d_val"] <= -0.05 and r["d_id"] > -0.02: cl.append("VAL-only")
        L.append(f"{k:9s} {100 * r['d_val']:+5.1f} {100 * r['d_id']:+5.1f} {100 * r['d_val_dist']:+7.1f} {100 * r['d_id_dist']:+7.1f}   {', '.join(cl)}")
    if len(items) > 34:
        L.append(f"({len(items) - 34} smaller-effect components omitted; see ablation.json)")
    return L


# ------------------------------------------------------------------ attribution
def build_attribution(model, n=N_PROBE, seed=0):
    m = _work_copy(model); out = {}
    groups = {"self": slice(0, 4), "goal": slice(4, 6), "obs": slice(6, 21)}
    gen = torch.Generator().manual_seed(1000 + seed)
    for kind in ("ID", "VAL"):
        obs = adapter.probe_obs(kind, n).clone().float()
        x = obs.clone().requires_grad_(True)
        a0 = adapter.forward(m, x)[:, 0]                           # (B,2) first action
        J = []
        for j in range(2):
            g, = torch.autograd.grad(a0[:, j].sum(), x, retain_graph=True); J.append(g)
        J = torch.stack(J, 1)                                      # (B,2,21)
        sens = {k: float(J[:, :, s].pow(2).sum((1, 2)).sqrt().mean()) for k, s in groups.items()}
        tot = sum(sens.values()) + 1e-12
        nn_idx = nearest_obstacle(obs); Jo = J[:, :, 6:].reshape(len(obs), 2, N_OBS, 3).pow(2).sum((1, 3)).sqrt()      # (B,5) per obstacle
        nn_s = float(Jo.gather(1, nn_idx[:, None]).mean()); far_s = float((Jo.sum(1) - Jo.gather(1, nn_idx[:, None])[:, 0]).mean() / (N_OBS - 1))
        with torch.no_grad():
            base = adapter.forward(m, obs)
            shuf = {}
            for k, s in groups.items():                            # replace this token group by another sample's (same batch) values
                perm = torch.randperm(len(obs), generator=gen); o2 = obs.clone(); o2[:, s] = obs[perm][:, s]
                shuf[k] = float((adapter.forward(m, o2)[:, :4] - base[:, :4]).abs().mean())
            exp = None
            ec = getattr(adapter.backend(), "expert_chunk", None)
            if ec is not None:
                exp = float((base - torch.as_tensor(ec(obs.numpy()), dtype=base.dtype)).abs().mean())
        out[kind] = {"jac_norm": sens, "jac_share": {k: v / tot for k, v in sens.items()}, "jac_obs_nearest": nn_s, "jac_obs_far_mean": far_s,
                     "shuffle_effect_L1": shuf, "expert_L1": exp}
    note = ""
    if getattr(model, "tok_self", None) is not None and model.tok_self.in_features != 4:
        note = ("NOTE: this policy's self token also receives goal-relative features (tok_self takes %d inputs), so the sensitivity to the 'goal' input features "
                "includes the self-token path." % model.tok_self.in_features)
    return {"note": note, "method": "Jacobian of the first action w.r.t. the 21 input features (per-token-group Frobenius norm, mean over probe states) + token-group shuffle "
                      "(replace one token group by another sample's) change of the first 4 actions", "n": n, "stats": out}


def digest_attribution(at):
    L = [f"attribution of the action to input token types, {at['n']} expert-visited states per set (ID=TRAIN layouts, VAL=VAL_SEEN layouts).",
         "jac = mean Frobenius norm of d(first action)/d(features of that token group); share = jac fraction; shuffle = mean |change of first 4 actions| (action units)",
         "  when that token group is replaced by another state's values; obs_near/obs_far = jac norm per obstacle for the nearest obstacle vs the mean of the 4 others."]
    if at.get("note"):
        L.append(at["note"])
    for kind in ("ID", "VAL"):
        s = at["stats"][kind]
        L.append(f"[{kind}] jac self/goal/obs = " + "/".join(f"{s['jac_norm'][k]:.3f}" for k in ("self", "goal", "obs")) + " ; share = " +
                 "/".join(f"{s['jac_share'][k]:.2f}" for k in ("self", "goal", "obs")) + f" ; obs_near={s['jac_obs_nearest']:.3f} obs_far={s['jac_obs_far_mean']:.3f}")
        L.append(f"[{kind}] shuffle self/goal/obs = " + "/".join(f"{s['shuffle_effect_L1'][k]:.3f}" for k in ("self", "goal", "obs")) +
                 (f" ; expert L1 on these states = {s['expert_L1']:.4f}" if s["expert_L1"] is not None else ""))
    i, v = at["stats"]["ID"], at["stats"]["VAL"]
    L.append("VAL/ID ratio of obstacle shuffle effect = %.2f ; of obs jac = %.2f (a memorising model reacts to obstacle input on TRAIN layouts and differently elsewhere)" %
             (v["shuffle_effect_L1"]["obs"] / max(i["shuffle_effect_L1"]["obs"], 1e-9), v["jac_norm"]["obs"] / max(i["jac_norm"]["obs"], 1e-9)))
    return L


# ------------------------------------------------------------------ delta vs earlier checkpoints
@torch.no_grad()
def delta_stats(model, ref_models, n=200):
    out = {"weights": {}, "ckpt_success": {}}
    cur = ops.matrix_table(model)
    for k in sorted(ref_models or {}):
        ref = ops.matrix_table(ref_models[k])
        out["weights"][f"ckpt{int(k):03d}"] = {nm: float((cur[nm].double() - ref[nm].double()).norm() / (ref[nm].double().norm() + 1e-30)) for nm in cur}
        out["ckpt_success"][f"ckpt{int(k):03d}"] = {s: float(adapter.evaluate(ref_models[k], s, n, ABL_SEEDS[s])["success"]) for s in ("VAL_SEEN", "ID_EVAL")}
    out["ckpt_success"]["current"] = {s: float(adapter.evaluate(model, s, n, ABL_SEEDS[s])["success"]) for s in ("VAL_SEEN", "ID_EVAL")}
    out["n_episodes"] = n
    return out


def digest_delta(dd):
    if not dd["weights"]:
        return ["delta vs earlier checkpoints: no reference checkpoints were provided."]
    L = [f"earlier checkpoints of the SAME training run (ckpt000 = earliest; revert_delta ref_ckpt = that index). success = closed loop, n={dd['n_episodes']} episodes (se ~ +-3 pts).",
         "ckpt     VAL_SEEN  ID_EVAL   relchange(mean/max over matrices)  top3 changed matrices"]
    for k, rel in dd["weights"].items():
        v = list(rel.values()); s = dd["ckpt_success"][k]
        top = sorted(rel, key=lambda n: (-rel[n], n))[:3]
        L.append(f"{k}  {s['VAL_SEEN']:.3f}    {s['ID_EVAL']:.3f}     {sum(v) / len(v):.3f}/{max(v):.3f}   " + " ".join(f"{n}:{rel[n]:.2f}" for n in top))
    s = dd["ckpt_success"]["current"]
    L.append(f"current   {s['VAL_SEEN']:.3f}    {s['ID_EVAL']:.3f}")
    last = list(dd["weights"])[-1]
    L.append(f"per-matrix relchange vs {last}: " + " ".join(f"{n}:{v:.2f}" for n, v in dd["weights"][last].items()))
    return L


# ------------------------------------------------------------------ build
def _write(path, data, text=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(data if text else json.dumps(data, sort_keys=True, indent=1) + "\n")


def build_dataroom(model, out_dir, ref_models=None, seed=0, n_ablate=200, ablation_cache=None, deadline=None):
    """Write the data room for `model` into out_dir. Returns the MANIFEST dict. Deterministic given seed; HOLD- and CANARY-free."""
    os.makedirs(out_dir, exist_ok=True)
    was = model.training; model.eval()
    try:
        data = {"spectroscopy": build_spectroscopy(model), "heads": build_heads(model),
                "ablation": build_ablation(model, n_ablate, ablation_cache, deadline), "attribution": build_attribution(model, seed=seed),
                "delta": delta_stats(model, ref_models, n_ablate)}
    finally:
        model.train(was)
    digs = {"spectroscopy": digest_spectroscopy(data["spectroscopy"]), "heads": digest_heads(data["heads"]), "ablation": digest_ablation(data["ablation"]),
            "attribution": digest_attribution(data["attribution"]), "delta": digest_delta(data["delta"])}
    files = []
    for k in FILES:
        _write(os.path.join(out_dir, f"{k}.json"), _r(data[k])); files.append(f"{k}.json")
        _write(os.path.join(out_dir, "digests", f"{k}.txt"), fit_text(digs[k]) + "\n", text=True); files.append(f"digests/{k}.txt")
    man = {"format": "core-dataroom-v1", "seed": seed, "complete": bool(data["ablation"]["complete"]),
           "files": {f: hashlib.sha256(open(os.path.join(out_dir, f), "rb").read()).hexdigest() for f in sorted(files)}}
    _write(os.path.join(out_dir, "MANIFEST.json"), man)
    return man


def build_wrong_room(out_dir, wrong_seed=1, wrong_model=None, ref_models=None, seed=0, n_ablate=200, ablation_cache=None):
    """Control: identical-format data room of a DIFFERENT model (default = the seed-1 model of the identical recipe, with its own earlier checkpoints)."""
    m = wrong_model if wrong_model is not None else adapter.load_ckpt(wrong_seed, "final")
    if ref_models is None and wrong_model is None:
        ref_models = {k: adapter.load_ckpt(wrong_seed, k) for k in range(10)}
    return build_dataroom(m, out_dir, ref_models=ref_models, seed=seed, n_ablate=n_ablate, ablation_cache=ablation_cache)
