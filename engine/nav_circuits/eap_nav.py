"""nav_circuits: edge attribution patching (EAP) for the core ACT-like Nav2D policy, with an EXACT edge-ablation oracle and validation.
GRAPH. Residual-stream decomposition (pre-LN transformer; every LayerNorm is part of its reader, standard EAP):
  encoder stream (7 tokens): writers = embed, E{l}.h{0..3} (head contributions through attn.out), E{l}.mlp (fc2 output)
  encoder readers = E{l}.attn, E{l}.mlp, mem (the enc_ln input that every decoder cross-attention reads)
  decoder stream (8 queries): writers = query, D{l}.sh{h} (self-attn heads), D{l}.xh{h} (cross-attn heads), D{l}.mlp
  decoder readers = D{l}.self, D{l}.cq (cross-attn query), D{l}.mlp, final (dec_ln -> head).   Biases of out/fc2 are always-on constants, never ablated.
EDGE (w -> r) removal = r's input sum loses w's contribution (zero-ablation, summed over tokens).
EXACT score(w->r) = metric(drop that edge) - metric(full), computed by re-running the decomposed forward.
EAP score (first order) = -sum_tokens out_w . dMetric/d(reader input), averaged over samples; one backward pass for ALL edges and ALL nodes.
NEURON score (node level, MLP hidden unit j) = -sum_tokens act_j . dMetric/d act_j  (equals ablating that neuron's fc2 column to first order).
CONTRAST (memorisation-vs-generalisation): metric A = L1 to the labels the model was TRAINED on, on training states; metric B = L1 to expert labels on held-out-layout states.
  contrast = score_A - score_B: large => the component is needed to reproduce the training labels but not the expert behaviour on new layouts (memorisation-like).
  Both sets of states are legitimate analyst inputs (training data, a validation demo set); the plant mask / oracle is never used here."""
import math, itertools, numpy as np, torch
import torch.nn.functional as F


def _split_heads(t, h): B, L, D = t.shape; return t.view(B, L, h, D // h).transpose(1, 2)


def run(m, obs, drop=frozenset(), record=False):
    """decomposed forward. drop = set of (writer, reader) edges to remove. returns (out, cache) ; cache has writers/readers/neuron acts when record=True."""
    nh = m.n_heads; cache = {"writers": {}, "readers": {}, "acts": {}, "edges": []}
    def factor(w, r):          # drop may be a set (edge removed) or a dict edge -> retained fraction (1 = untouched, 0 = removed)
        if isinstance(drop, dict): return drop.get((w, r), 1.0)
        return 0.0 if (w, r) in drop else 1.0
    def read(ws, r, const):
        terms = []
        for n, t in ws:
            cache["edges"].append((n, r)) if record else None
            f = factor(n, r); terms.append(t if f == 1.0 else t * f)
        s = sum(terms) + const
        if record: s.retain_grad(); cache["readers"][r] = s
        return s
    def reg(name, t):
        if record: cache["writers"][name] = t
        return t
    # ---- encoder
    ws = [("embed", reg("embed", m.tokens(obs)))]; const = torch.zeros(1, 1, m.query_emb.shape[1])
    for l, L in enumerate(m.enc_layers):
        a_in = read(ws, f"E{l}.attn", const); h = L.ln1(a_in); A = L.attn
        q, k, v = (_split_heads(t, nh) for t in A.qkv(h).chunk(3, -1)); att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1); y = att @ v      # B,h,L,dh
        dh = y.shape[-1]
        for hh in range(nh): ws.append((f"E{l}.h{hh}", reg(f"E{l}.h{hh}", y[:, hh] @ A.out.weight[:, hh * dh:(hh + 1) * dh].T)))
        const = const + A.out.bias
        m_in = read(ws, f"E{l}.mlp", const); hid = F.gelu(L.mlp.fc1(L.ln2(m_in)))
        if record: hid.retain_grad(); cache["acts"][f"E{l}.mlp"] = hid
        ws.append((f"E{l}.mlp", reg(f"E{l}.mlp", hid @ L.mlp.fc2.weight.T))); const = const + L.mlp.fc2.bias
    mem = m.enc_ln(read(ws, "mem", const))
    # ---- decoder
    B = obs.shape[0]; qs = [("query", reg("query", m.query_emb[None].expand(B, -1, -1)))]; const = torch.zeros(1, 1, m.query_emb.shape[1])
    for l, L in enumerate(m.dec_layers):
        s_in = read(qs, f"D{l}.self", const); h = L.ln1(s_in); A = L.self_attn
        q, k, v = (_split_heads(t, nh) for t in A.qkv(h).chunk(3, -1)); att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1); y = att @ v; dh = y.shape[-1]
        for hh in range(nh): qs.append((f"D{l}.sh{hh}", reg(f"D{l}.sh{hh}", y[:, hh] @ A.out.weight[:, hh * dh:(hh + 1) * dh].T)))
        const = const + A.out.bias
        c_in = read(qs, f"D{l}.cq", const); C = L.cross_attn; qq = _split_heads(C.q(L.ln2(c_in)), nh); kk, vv = (_split_heads(t, nh) for t in C.kv(mem).chunk(2, -1))
        att = torch.softmax(qq @ kk.transpose(-1, -2) / math.sqrt(qq.shape[-1]), -1); y = att @ vv
        for hh in range(nh): qs.append((f"D{l}.xh{hh}", reg(f"D{l}.xh{hh}", y[:, hh] @ C.out.weight[:, hh * dh:(hh + 1) * dh].T)))
        const = const + C.out.bias
        m_in = read(qs, f"D{l}.mlp", const); hid = F.gelu(L.mlp.fc1(L.ln3(m_in)))
        if record: hid.retain_grad(); cache["acts"][f"D{l}.mlp"] = hid
        qs.append((f"D{l}.mlp", reg(f"D{l}.mlp", hid @ L.mlp.fc2.weight.T))); const = const + L.mlp.fc2.bias
    out = m.head(m.dec_ln(read(qs, "final", const)))
    return out, cache


def metric(out, y): return (out - y).abs().mean(dim=(1, 2))           # per-sample L1


@torch.no_grad()
def exact_edge_scores(m, obs, y, edges, keep_frac=0.0):
    """exact damage of scaling each edge's contribution to keep_frac (0 = remove). Divide by (1-keep_frac) to compare with the first-order score."""
    base = metric(run(m, obs)[0], y).mean().item(); return {e: (metric(run(m, obs, {e: keep_frac})[0], y).mean().item() - base) / (1 - keep_frac) for e in edges}


def eap(m, obs, y, rows=None):
    """first-order scores on the sample rows (default all). returns (edge_scores dict, neuron_scores dict name->(256,), base_metric)."""
    m.zero_grad(); out, c = run(m, obs, record=True); per = metric(out, y); rows = torch.arange(len(obs)) if rows is None else torch.as_tensor(rows)
    per[rows].sum().backward(); n = len(rows); es = {}
    for (w, r) in dict.fromkeys(c["edges"]):
        g = c["readers"][r].grad; es[(w, r)] = float(-(c["writers"][w] * g)[rows].sum().item() / n)
    ns = {k: (-(a * a.grad).sum(1)[rows].mean(0)).detach().numpy() for k, a in c["acts"].items()}
    return es, ns, float(per[rows].mean().detach())


def contrast(m, obs_a, y_a, obs_b, y_b, rows_a=None):
    ea, na, ba = eap(m, obs_a, y_a, rows_a); eb, nb, bb = eap(m, obs_b, y_b)
    return {"edge": {e: (ea[e], eb[e], ea[e] - eb[e]) for e in ea}, "neuron": {k: (na[k], nb[k], na[k] - nb[k]) for k in na}, "base": (ba, bb)}


def node_scores(edge_table, which=2):
    """writer-node score = sum of its outgoing edge scores (first-order damage of removing the writer from every reader)."""
    out = {}
    for (w, r), v in edge_table.items(): out[w] = out.get(w, 0.0) + v[which]
    return out


def validate(m, obs, y, keep_frac=0.0):
    """EAP vs EXACT over ALL edges on one metric. returns dict(spearman, sign_agree (|exact|>1e-4), top10_overlap, n_edges, slope)."""
    from scipy.stats import spearmanr
    es, _, _ = eap(m, obs, y); ex = exact_edge_scores(m, obs, y, list(es), keep_frac); keys = list(es)
    a, b = np.array([es[k] for k in keys]), np.array([ex[k] for k in keys]); big = np.abs(b) > 1e-4
    top = lambda v: set(np.argsort(-np.abs(v))[:10])
    return {"spearman": float(spearmanr(a, b)[0]), "sign_agree": float((np.sign(a[big]) == np.sign(b[big])).mean()), "top10_overlap": len(top(a) & top(b)),
            "n_edges": len(keys), "n_big": int(big.sum()), "slope": float((a * b).sum() / (b * b).sum())}
