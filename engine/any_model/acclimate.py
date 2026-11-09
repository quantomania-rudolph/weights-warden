"""ACCLIMATION: run this ONCE on a newly plugged-in model before any agent touches it.  It does not assume the tools work - it PROVES, per tool, that they do on THIS model, and disables the ones it cannot prove.
Checks: structure discovered; identity (gamma=1 changes nothing); op == manual surgery via hooks (heads, neurons); edits touch ONLY the intended parameter entries; residual-stream decomposition is exact;
first-order attribution vs exact patching (rank agreement); first-order edge score vs exact at a tiny perturbation; range validation rejects bad configs; generic matrix ops run.
Result: rows (PASS/FAIL/WARN) + `verified` = the tools the agent may use.  100% correctness cannot be proven for an arbitrary model - this makes every claim a tested one and fails LOUDLY otherwise."""
import os, sys, json, copy, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import ops as OPS, structure as S, patching as P, llm_ops, more_ops


def _spearman(a, b):
    import numpy as np
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def _changed(a, b): return sum(int((p != q).sum()) for p, q in zip(a.parameters(), b.parameters()))


def acclimate(model, clean, corrupt, metric_fn, out_path=None, min_spearman=0.6, spec=None):
    model.eval(); R = []; ok = lambda c, name, msg="": R.append(("PASS" if c else "FAIL", name, msg)); verified = set()
    llm_ops.enable(spec); more_ops.enable_value_ops(); more_ops.enable_more_value_ops()      # matrix-level ops only: Nav-structure ops (head_transplant ...) must NOT exist in LLM mode
    try:
        try: tm = S.discover(model, spec); ok(True, "structure discovered", f"{tm.family if tm.kind == 'transformer' else 'mlp'}: " + (f"{tm.n_layers} blocks, {tm.n_heads} heads (kv {tm.n_kv_heads}) x {tm.head_dim}, d_ff {tm.d_ff}" if tm.kind == 'transformer' else f"hidden widths {tm.widths}"))
        except S.Unsupported as e: ok(False, "structure discovered", str(e)[:300]); tm = None
        base = model(clean[:2]); base = base.logits if hasattr(base, "logits") else base
        if tm is not None:
            tf = tm.kind == "transformer"; L = 0; Ln = tm.n_layers if tf else tm.n_hidden; H = 1 if tf else None; units = [0, 1, 2]
            outs = lambda m: (lambda o: o.logits if hasattr(o, "logits") else o)(m(clean[:2]))
            def run_edit(op, cfg): return OPS.apply(op, cfg, model)
            # identity + locality + equivalence for neurons
            e1 = run_edit("scale_neurons", {"layer": L, "neurons": units, "gamma": 1.0}); ok(torch.allclose(outs(e1), base, atol=1e-6), "scale_neurons gamma=1 is the identity")
            e0 = run_edit("scale_neurons", {"layer": L, "neurons": units, "gamma": 0.0}); exp = len(units) * (tm.d_model if tf else (tm.lin[L + 1].out_features)); nch = _changed(model, e0)
            hs = P._patch_hooks(model, tm, {("neuron" if tf else "unit", L): torch.zeros_like(P.corrupt_acts_of(model, tm, clean[:2])[("neuron" if tf else "unit", L)])}, {("neuron" if tf else "unit", L): units}); ref = outs(model); [h.remove() for h in hs]
            same = torch.allclose(outs(e0), ref, atol=1e-5); ok(same, "scale_neurons gamma=0 == zeroing those activations by hook", "" if same else f"max diff {(outs(e0) - ref).abs().max():.2e}")
            ok(nch <= exp and nch > 0, "scale_neurons touches only the intended weights", f"changed {nch} entries, expected <= {exp}")
            if same and nch <= exp: verified |= {"scale_neurons", "prune_neurons"}
            if tf:
                e1 = run_edit("scale_head", {"layer": L, "head": H, "gamma": 1.0}); ok(torch.allclose(outs(e1), base, atol=1e-6), "scale_head gamma=1 is the identity")
                e0 = run_edit("scale_head", {"layer": L, "head": H, "gamma": 0.0}); hs = P._patch_hooks(model, tm, {("head", L): torch.zeros_like(P.corrupt_acts_of(model, tm, clean[:2])[("head", L)])}, {("head", L): [H]}); ref = outs(model); [h.remove() for h in hs]
                same = torch.allclose(outs(e0), ref, atol=1e-5); nch = _changed(model, e0); ok(same, "scale_head gamma=0 == zeroing that head's output by hook", "" if same else f"max diff {(outs(e0) - ref).abs().max():.2e}")
                ok(0 < nch <= tm.head_dim * tm.d_model, "scale_head touches only that head's weights", f"changed {nch}, expected <= {tm.head_dim * tm.d_model}")
                if same and 0 < nch <= tm.head_dim * tm.d_model: verified.add("scale_head")
                if tm.q_layout != "contiguous": R.append(("WARN", "head_temperature disabled", "fused qkv layout not declared contiguous (cannot be established from shapes)")); skip_t = True
                else: skip_t = False
                if not skip_t:
                  e1 = run_edit("head_temperature", {"layer": L, "head": H, "gamma": 1.0}); ident = torch.allclose(outs(e1), base, atol=1e-6); e2 = run_edit("head_temperature", {"layer": L, "head": H, "gamma": 2.0}); nch = _changed(model, e2)
                  ok(ident and nch > 0 and nch <= tm.head_dim * (tm.d_model + 1), "head_temperature: identity at 1, local to that head's query rows", f"changed {nch}")
                  if ident and 0 < nch <= tm.head_dim * (tm.d_model + 1): verified.add("head_temperature")
                # residual decomposition exactness
                rec = P._Rec(model, tm, False, True, bt=tuple(clean[:2].shape[:2]))
                with torch.no_grad(): model(clean[:2])
                rec.close(); w = P._writers(tm, rec); tot = w[("embed", 0)] + sum(w[("head", l)].sum(2) + w[("mlp", l)] for l in range(tm.n_layers)); tot = tot + sum((b.o.bias.detach() if b.o.bias is not None else 0) for b in tm.blocks)
                fin = rec.R(("r", "final", 0)).detach(); err = float((tot - fin).abs().max()); ok(err < 1e-4, "residual stream = embed + sum(heads) + sum(mlps) + biases (decomposition exact)", f"max err {err:.2e}")
                seq = max(float(((rec.R(("r", "attn_in", l)) + w[("head", l)].sum(2) + (b.o.bias.detach() if b.o.bias is not None else 0)) - rec.R(("r", "mlp_in", l))).abs().max()) for l, b in enumerate(tm.blocks))
                R.append(("PASS" if seq < 1e-4 else "WARN", "sequential residual (the MLP reads the post-attention residual)", "" if seq < 1e-4 else f"max err {seq:.2e}: parallel-residual / non-standard block -> edge-level circuit disabled (node-level still valid)"))
                dec_ok = err < 1e-4 and seq < 1e-4
            else: dec_ok = False
            # attribution vs exact
            ns = P.node_scores(model, tm, clean, corrupt, metric_fn); eff = ns["effect"]; ca = P.corrupt_acts_of(model, tm, corrupt)
            flat = sorted(((float(v[:, i].mean()), k, i) for k, v in eff.items() for i in range(v.shape[1])), key=lambda r: -abs(r[0]))[:16]
            ex = [float(P.exact_node_effect(model, tm, clean, ca, k, i, metric_fn).mean()) for _, k, i in flat]; rho = _spearman([f[0] for f in flat], ex); sg = sum((a[0] > 0) == (b > 0) for a, b in zip(flat, ex)) / len(flat)
            R.append(("PASS" if rho >= min_spearman else "WARN", "first-order node attribution vs exact patching (top-16 nodes)", f"spearman {rho:.2f}, sign agreement {sg:.2f}" + ("" if rho >= min_spearman else " -> rank nodes with exact patching only")))
            verified.add("node_circuit_exact"); 
            if rho >= min_spearman: verified.add("node_circuit_attribution")
            if tf:
                es = P.edge_scores(model, tm, clean, corrupt, metric_fn); cand = sorted(((float(v.abs().max()), k) for k, v in es.items()), key=lambda r: -r[0])[:3]; errs = []
                for _, (wk, rk) in cand:
                    v = es[(wk, rk)]; h = int(v.abs().mean(0).argmax()) if wk[0] == "head" else None; pred = float((v[:, h] if h is not None else v).mean()); s = 1e-3
                    exv = float(P.exact_edge_effect(model, tm, clean, corrupt, wk, rk, metric_fn, head=h, scale=s).mean()) / s; errs.append(abs(exv - pred) / (abs(pred) + 1e-9))
                good = sorted(errs)[len(errs) // 2] < 0.25; ok(good, "first-order edge score == exact edge patch at a 0.1% perturbation (median rel. error < 25%)", f"rel errors {[round(e, 3) for e in errs]}")
                if good and dec_ok: verified.add("edge_circuit")
        # validation (poison)
        bad = [("scale_neurons", {"layer": 999, "neurons": [0], "gamma": 0.0}), ("scale_neurons", {"layer": 0, "neurons": [10 ** 6], "gamma": 0.0})] + ([("scale_head", {"layer": 0, "head": 999, "gamma": 0.0})] if tm is not None and tm.kind == "transformer" else [])
        rej = 0
        for op, c in bad:
            try: OPS.validate_cfg(op, c, model)
            except ValueError: rej += 1
        ok(rej == len(bad), "out-of-range configs are rejected", f"{rej}/{len(bad)}")
        # generic matrix ops
        name = next(n for n, p in model.named_parameters() if p.ndim == 2)
        try: e = OPS.apply("scale_slice", {"matrix": name, "axis": "row", "indices": [0], "gamma": 0.5}, model); OPS.apply("nudge_weights", {"matrix": name, "axis": "row", "index": 0, "positions": [0], "deltas": [1e-4]}, model); ok(_changed(model, e) > 0, "generic matrix ops (scale_slice, nudge_weights) run", name); verified.add("matrix_ops")
        except Exception as ex_: ok(False, "generic matrix ops run", repr(ex_)[:200])
    finally: llm_ops.disable()
    rep = {"rows": R, "verified_tools": sorted(verified), "n_fail": sum(r[0] == "FAIL" for r in R), "n_warn": sum(r[0] == "WARN" for r in R)}
    if out_path: json.dump(rep, open(out_path, "w"), indent=1)
    return rep


def show(rep):
    for lvl, n, m in rep["rows"]: print(f"[{lvl}] {n}" + (f"  -- {m}" if m else ""))
    print(f"\nverified tools: {rep['verified_tools']}\n{rep['n_fail']} FAIL, {rep['n_warn']} WARN")
