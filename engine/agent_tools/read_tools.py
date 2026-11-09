"""READ tools (analysis only; HOLD/CANARY-free): raw weights, per-head Q/K/V/O views and QK/OV circuit stats, attention patterns, neuron activation stats.
All outputs are capped (<= 4000 numbers per call) and ASCII/JSON so an LLM can read them."""
import os, sys, math, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core",): sys.path.insert(0, os.path.join(ROOT, p))
import ops as OPS, dataroom as DR
CAP = 4000
RAW = os.environ.get("HARNESS_RAW_VIEW", "1") != "0"      # default ON (analyst sees raw values). Set HARNESS_RAW_VIEW=0 to run the SUMMARY-ONLY ablation arm (does raw access help?)


def raw_view_enabled(): return os.environ.get("HARNESS_RAW_VIEW", "1") != "0"


def read_weights(m, matrix, rows=None, cols=None):
    """matrix: any name in matrix_names OR any parameter name (biases, LayerNorm).  rows/cols: 'a:b' slices.  Returns shape, summary stats and the slice values."""
    tab = OPS.matrix_table(m); P = dict(m.named_parameters()); W = tab.get(matrix, P.get(matrix))
    if W is None: raise ValueError(f"unknown {matrix!r}; matrices: {list(tab)}; parameters: {list(P)}")
    W = W.detach(); sl = lambda s, n: slice(*[int(x) if x else None for x in (s or ":").split(":")]) if s else slice(None)
    V = W[sl(rows, W.shape[0])] if W.ndim == 1 else W[sl(rows, W.shape[0]), sl(cols, W.shape[1])]
    if not raw_view_enabled(): return {"matrix": matrix, "shape": list(W.shape), "rms": round(float(W.pow(2).mean().sqrt()), 5), "frob": round(float(W.norm()), 4), "raw_view": "DISABLED for this run (HARNESS_RAW_VIEW=0): summary statistics only", "slice_row_norms": [round(float(x), 4) for x in (V.norm(dim=-1) if V.ndim > 1 else V.abs())[:64]]}
    if V.numel() > CAP: raise ValueError(f"slice has {V.numel()} values (> {CAP}); narrow rows/cols")
    return {"matrix": matrix, "shape": list(W.shape), "rms": round(float(W.pow(2).mean().sqrt()), 5), "frob": round(float(W.norm()), 4), "slice_shape": list(V.shape), "values": np.round(V.numpy(), 5).tolist()}


def _split(m, kind, layer):
    at = OPS.attn_module(m, kind, layer); H = OPS.n_heads_of(m, at)
    if kind in ("enc_self", "dec_self"):
        d = at.qkv.weight.shape[1]; Wq, Wk, Wv = at.qkv.weight.detach().split(d, 0); bq, bk, bv = at.qkv.bias.detach().split(d, 0)
    else:
        d = at.q.weight.shape[1]; Wq, bq = at.q.weight.detach(), at.q.bias.detach(); Wk, Wv = at.kv.weight.detach().split(d, 0); bk, bv = at.kv.bias.detach().split(d, 0)
    return at, H, d, (Wq, Wk, Wv), (bq, bk, bv)


def qkv_view(m, kind, layer, head, dump=False):
    """Per-head Q/K/V/O: norms, QK circuit (Wq_h^T Wk_h) and OV circuit (Wo_h Wv_h) singular values, which INPUT dims (of the layernormed residual) each projection reads most, and (dump=True) the raw d_head x d_model slices."""
    at, H, d, (Wq, Wk, Wv), (bq, bk, bv) = _split(m, kind, layer); dh = d // H; s = slice(head * dh, (head + 1) * dh); Wo = at.out.weight.detach()[:, s]
    q, k, v = Wq[s], Wk[s], Wv[s]; QK = q.T @ k; OV = Wo @ v; sv = lambda M: [round(float(x), 4) for x in torch.linalg.svdvals(M)[:6]]
    top = lambda M: [int(i) for i in M.norm(dim=0).argsort(descending=True)[:8]]
    out = {"kind": kind, "layer": layer, "head": head, "d_model": d, "d_head": dh, "norms": {"Wq": round(float(q.norm()), 4), "Wk": round(float(k.norm()), 4), "Wv": round(float(v.norm()), 4), "Wo": round(float(Wo.norm()), 4)},
           "bias_norms": {"bq": round(float(bq[s].norm()), 4), "bk": round(float(bk[s].norm()), 4), "bv": round(float(bv[s].norm()), 4)},
           "QK_circuit_top_singular_values": sv(QK), "OV_circuit_top_singular_values": sv(OV), "QK_effective_rank": round(float(torch.linalg.svdvals(QK).sum() ** 2 / (torch.linalg.svdvals(QK).pow(2).sum() + 1e-12)), 3),
           "input_dims_read_most": {"q": top(q), "k": top(k), "v": top(v)}, "output_dims_written_most": [int(i) for i in Wo.norm(dim=1).argsort(descending=True)[:8]],
           "edit_handles": {"rows_in_matrix": {"q": f"rows {head * dh}:{(head + 1) * dh} of {'enc%d.qkv' % layer if kind == 'enc_self' else 'dec%d.sqkv' % layer if kind == 'dec_self' else 'dec%d.cq' % layer}", "o": f"cols {head * dh}:{(head + 1) * dh} of {'enc%d.out' % layer if kind == 'enc_self' else 'dec%d.sout' % layer if kind == 'dec_self' else 'dec%d.cout' % layer}"}}}
    if dump and not raw_view_enabled(): out["raw"] = "DISABLED for this run (HARNESS_RAW_VIEW=0)"
    elif dump:
        if 4 * dh * d > CAP * 4: raise ValueError("dump too large")
        out["raw"] = {n: np.round(t.numpy(), 4).tolist() for n, t in (("Wq", q), ("Wk", k), ("Wv", v), ("Wo_T", Wo.T.contiguous()))}
    return out


@torch.no_grad()
def attention_view(m, obs, kind, layer, head, nsamples=2):
    """mean attention (query-token-type x key-token-type) averaged over `obs`, plus entropy, plus the full pattern for the first `nsamples` states."""
    A = DR.attention_patterns(m, obs)[(kind, layer)][:, head]                      # (B, Q, K)
    keys = ["self", "goal", "obs0", "obs1", "obs2", "obs3", "obs4"]; qn = keys if kind == "enc_self" else [f"q{i}" for i in range(A.shape[1])]; kn = keys if kind != "dec_self" else qn
    mean = A.mean(0); ent = -(A * A.clamp_min(1e-12).log()).sum(-1).mean() / math.log(A.shape[-1])
    return {"kind": kind, "layer": layer, "head": head, "query_tokens": qn, "key_tokens": kn, "mean_attention": np.round(mean.numpy(), 3).tolist(), "normalised_entropy": round(float(ent), 4),
            "examples": [np.round(A[i].numpy(), 3).tolist() for i in range(min(nsamples, len(A)))]}


@torch.no_grad()
def neuron_activity(m, obs_a, obs_b, kind, layer, top=12):
    """post-GELU hidden-unit statistics of one MLP block on state set A (train layouts) vs B (held-out layouts): mean activation, fraction active, and the units whose behaviour differs most between A and B."""
    lay = (m.enc_layers if kind == "enc" else m.dec_layers)[layer]; cap = []
    h = lay.mlp.fc2.register_forward_pre_hook(lambda mod, inp: cap.append(inp[0].detach()))
    m(obs_a); m(obs_b); h.remove(); a, b = cap[0].flatten(0, 1), cap[1].flatten(0, 1)
    ma, mb, fa, fb = a.mean(0), b.mean(0), (a > 0.1).float().mean(0), (b > 0.1).float().mean(0); diff = (ma - mb).abs()
    return {"kind": kind, "layer": layer, "n_units": int(a.shape[1]), "mean_activation_A": round(float(a.mean()), 4), "mean_activation_B": round(float(b.mean()), 4), "dead_units_A": int((fa < 0.001).sum()),
            "units_most_different_A_vs_B": [{"unit": int(i), "mean_A": round(float(ma[i]), 4), "mean_B": round(float(mb[i]), 4), "active_frac_A": round(float(fa[i]), 3), "active_frac_B": round(float(fb[i]), 3)} for i in diff.argsort(descending=True)[:top]]}
