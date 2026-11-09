"""Compare the ACTION-PRODUCING parts of a policy.
1) generic protocol: a model may define `action_heads(obs) -> {name: (B,H,2) tensor}` (e.g. {"base":..., "nudge":..., "final": base+nudge} for a residual-nudge / action-expert head).  compare_heads() then reports, on TRAIN states
   (vs the labels trained on) and HELD-OUT states (vs expert labels): each head's error, its size, and for additive heads whether the nudge moves the output TOWARD the label (cos with label-minus-base, and fraction of states it helps).
2) for our NavPolicy (single base head): `decoder_readouts` = logit lens (apply the real head to the decoder stream after each decoder layer = how the action is built up),
   and `writer_decomposition` = EXACT split of the final action into per-writer contributions (LayerNorm statistics frozen at their true values, so the pieces sum to the output).
No closed-loop success is computed here; all are teacher-forced numbers."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "nav_circuits"): sys.path.insert(0, os.path.join(ROOT, p))
import eap_nav as EN


def _l1(a, b): return float((a - b).abs().mean())


@torch.no_grad()
def compare_heads(model, tr_obs, tr_y, ho_obs, ho_y):
    out = {}
    for tag, o, y in (("train_vs_trained_labels", tr_obs, tr_y), ("heldout_vs_expert", ho_obs, ho_y)):
        hs = model.action_heads(o); rows = {n: {"l1_to_label": round(_l1(t, y), 5), "mean_abs_action": round(float(t.abs().mean()), 5)} for n, t in hs.items()}
        if "base" in hs and "final" in hs:
            nudge = hs["final"] - hs["base"]; want = y - hs["base"]; cos = torch.nn.functional.cosine_similarity(nudge.flatten(1), want.flatten(1), dim=1)
            helped = ((hs["final"] - y).abs().mean((1, 2)) < (hs["base"] - y).abs().mean((1, 2))).float().mean()
            rows["_nudge"] = {"mean_abs": round(float(nudge.abs().mean()), 5), "relative_to_base": round(float(nudge.abs().mean() / (hs["base"].abs().mean() + 1e-9)), 4), "mean_cos_with_needed_correction": round(float(cos.mean()), 4), "fraction_of_states_helped": round(float(helped), 4)}
        out[tag] = rows
    return out


@torch.no_grad()
def decoder_readouts(model, obs):
    """logit lens: head(dec_ln(stream)) after each decoder layer (last one == the real output)."""
    x = model.tokens(obs)
    for l in model.enc_layers: x = l(x)
    mem = model.enc_ln(x); q = model.query_emb[None].expand(obs.shape[0], -1, -1); res = {}
    for i, l in enumerate(model.dec_layers):
        q = l(q, mem); res[f"after_D{i}"] = model.head(model.dec_ln(q))
    return res


def writer_decomposition(model, obs):
    """returns ({writer: (B,H,2) contribution}, reconstruction_error).  contribution_w = head.W (gamma * (w - mean_w)/sigma); constants (biases + LN beta + head bias) go to '_const'."""
    with torch.enable_grad(): full, c = EN.run(model, obs, record=True)       # record=True needs autograd (retain_grad); results are detached below
    full = full.detach(); c = {"readers": {k: v.detach() for k, v in c["readers"].items()}, "writers": {k: v.detach() for k, v in c["writers"].items()}}; s = c["readers"]["final"]; mu = s.mean(-1, keepdim=True); sig = (s.var(-1, unbiased=False, keepdim=True) + model.dec_ln.eps).sqrt()
    g, W = model.dec_ln.weight, model.head.weight; dec = {k: v for k, v in c["writers"].items() if k == "query" or k.startswith("D")}; res = {}
    for k, w in dec.items(): res[k] = ((w - w.mean(-1, keepdim=True)) / sig * g) @ W.T
    tot = sum(res.values()); res["_const"] = full - tot                    # everything else, exactly (biases, beta, head bias)
    return res, float((tot + res["_const"] - full).abs().max())
