"""ANALYST TOOLS (what the LLM may call; every call is logged to RUN/tools.jsonl; none reads HOLD/CANARY; none returns closed-loop success).
  inventory                       every editable component, labelled by role, with the op that edits it
  neuron_scan --block enc0|enc1|dec0|dec1 [--top N]   label-free contrast attribution per MLP neuron (damage on TRAINED labels/train states minus damage on expert labels/held-out states)
  circuit [--top N]               EXACT edge-patching contrast over the whole residual-stream graph (+ per-node roll-up): which writer->reader connections carry the memorisation-like signal
  weights --matrix NAME [--rows a:b --cols c:d]   raw weights of any matrix or parameter (biases, LayerNorm), <=4000 values
  qkv --kind enc_self|dec_self|dec_cross --layer L --head H [--dump]   per-head Wq/Wk/Wv/Wo norms, QK and OV circuit spectra, most-read input dims (+raw slices)
  attention --kind K --layer L --head H [--set heldout|train]   mean query x key attention by token, entropy, examples
  activity --kind enc|dec --layer L   MLP unit activations on train vs held-out states
  heads                           how actions are produced: declared action heads (base/nudge/expert) if the model defines `action_heads`, else decoder readouts + exact per-writer split
  diff --a CKPT --b CKPT          per-module relative weight change and teacher-forced output change between two checkpoints (base vs fine-tuned / earlier vs final)
  usage: python agent_tools/analyst_tools.py RUN CMD [...]   (RUN is a core run dir; target policy = that run's target seed)"""
import os, sys, json, argparse, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "nav_planted", "nav_circuits", "nav_sandbox"): sys.path.insert(0, os.path.join(ROOT, p))
import loop as LP, ops as OPS, adapter, policy as P, eap_nav as EN, attrib as AT, planted_metrics as PM
torch.set_num_threads(1); BLOCKS = [("enc", 0), ("enc", 1), ("dec", 0), ("dec", 1)]


def _log(run, rec):
    with open(os.path.join(run, "tools.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")


def _data(seed):
    """(train states, labels the policy was TRAINED on, held-out states, expert labels).  Planted target (seed 7) trains on the planted labels; others on clean labels."""
    obs, clean, planted, _ = PM.build(); idx = np.random.default_rng(0).choice(len(obs), 4000, replace=False); y = planted if seed == 7 else clean
    vo = adapter.probe_obs("VAL", 2000); vy = torch.as_tensor(np.asarray(adapter.backend().expert_chunk(vo.numpy())), dtype=torch.float32)
    return torch.from_numpy(obs[idx]), torch.from_numpy(y[idx]), vo, vy


def inventory(m):
    rows = []; H = m.n_heads
    for k, (kind, layers) in enumerate((("enc", m.enc_layers), ("dec", m.dec_layers))):
        for i, l in enumerate(layers):
            ff = l.mlp.fc1.out_features
            rows.append({"component": f"{kind}{i}.mlp", "role": "feedforward bank", "units": ff, "edit_with": f"scale_neurons/prune_neurons kind={kind} layer={i} (neuron ids 0..{ff - 1}); matrices {kind}{i}.fc1/.fc2 for SVD ops"})
            kinds = ["enc_self"] if kind == "enc" else ["dec_self", "dec_cross"]
            for hk in kinds:
                rows.append({"component": f"{kind}{i}.{hk}", "role": "attention heads", "units": H, "edit_with": f"scale_head kind={hk} layer={i} head=0..{H - 1}; head_temperature (if enabled); matrices in matrix_names"})
    rows += [{"component": "tokens", "role": "input embeddings", "units": 3, "edit_with": "scale_token_embed which=self|goal|obs"}, {"component": "head", "role": "action output head (base head; no separate action expert / residual head in this model)", "units": 1, "edit_with": "matrix 'head' via SVD ops"},
             {"component": "any", "role": "stack", "units": 0, "edit_with": "composite (<=4 sub-edits)"}]
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("cmd", choices=["inventory", "neuron_scan", "circuit", "diff", "heads", "weights", "qkv", "attention", "activity"]); ap.add_argument("--block"); ap.add_argument("--top", type=int, default=20); ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--matrix"); ap.add_argument("--rows"); ap.add_argument("--cols"); ap.add_argument("--kind"); ap.add_argument("--layer", type=int, default=0); ap.add_argument("--head", type=int, default=0); ap.add_argument("--dump", action="store_true"); ap.add_argument("--set", default="heldout")
    a = ap.parse_args(); seed = json.loads(open(os.path.join(a.run, "log.jsonl")).readline())["payload"]["target_seed"]; m = LP.target_model({"target_seed": seed}); out = None
    if a.cmd == "inventory": out = inventory(m)
    elif a.cmd == "neuron_scan":
        o, y, vo, vy = _data(seed); d1 = AT.attr(m, o, y, np.arange(len(o))) - AT.attr(m, vo, vy, np.arange(len(vo))); b = [f"{k}{i}" for k, i in BLOCKS].index(a.block); s = d1[b * 256:(b + 1) * 256]
        order = np.argsort(-s)[:a.top]; out = {"block": a.block, "note": "contrast attribution, first-order (validated vs exact ablation on this policy); larger = more memorisation-like", "top": [{"neuron": int(j), "contrast": round(float(s[j]), 5)} for j in order],
                                                  "block_mean": round(float(s.mean()), 5), "block_top24_sum": round(float(np.sort(s)[::-1][:24].sum()), 4)}
    elif a.cmd == "circuit":
        o, y, vo, vy = _data(seed); _, c = EN.run(m, o[:8], record=True); edges = list(dict.fromkeys(c["edges"])); A_ = EN.exact_edge_scores(m, o, y, edges); B_ = EN.exact_edge_scores(m, vo, vy, edges)
        rows = sorted(({"writer": w, "reader": r, "train_labels_damage": round(float(A_[(w, r)]), 5), "expert_heldout_damage": round(float(B_[(w, r)]), 5), "contrast": round(float(A_[(w, r)] - B_[(w, r)]), 5)} for w, r in edges), key=lambda x: -x["contrast"])
        nodes = {}
        for r_ in rows: nodes[r_["writer"]] = nodes.get(r_["writer"], 0.0) + r_["contrast"]
        out = {"note": "EXACT edge removal (zero-ablation of one writer's contribution to one reader); contrast = needed for training labels but not for expert behaviour on held-out layouts", "top_edges": rows[:a.top],
               "top_nodes": sorted(({"node": k, "sum_edge_contrast": round(v, 5)} for k, v in nodes.items()), key=lambda x: -x["sum_edge_contrast"])[:a.top]}
    elif a.cmd in ("weights", "qkv", "attention", "activity"):
        import read_tools as RT
        if a.cmd == "weights": out = RT.read_weights(m, a.matrix, a.rows, a.cols)
        elif a.cmd == "qkv": out = RT.qkv_view(m, a.kind, a.layer, a.head, a.dump)
        else:
            o, y, vo, vy = _data(seed)
            out = RT.attention_view(m, vo if a.set == "heldout" else o[:1000], a.kind, a.layer, a.head) if a.cmd == "attention" else RT.neuron_activity(m, o[:1000], vo[:1000], a.kind, a.layer)
    elif a.cmd == "heads":
        import action_heads as AH; o, y, vo, vy = _data(seed)
        if hasattr(m, "action_heads"): out = {"declared_action_heads": AH.compare_heads(m, o, y, vo, vy)}
        else:
            out = {"note": "this model has ONE base action head (no action-expert/residual head); showing how the action is built: logit-lens readouts after each decoder layer, and the exact per-writer split of the final action"}
            for tag, ob, yy in (("train_vs_trained_labels", o, y), ("heldout_vs_expert", vo, vy)):
                ro = AH.decoder_readouts(m, ob); dec, err = AH.writer_decomposition(m, ob)
                out[tag] = {"readout_l1": {k: round(AH._l1(v, yy), 4) for k, v in ro.items()}, "decomposition_recon_error": err,
                            "writer_mean_abs_contribution": {k: round(float(v.abs().mean()), 4) for k, v in sorted(dec.items(), key=lambda kv: -float(kv[1].abs().mean()))[:a.top]}}
    else:
        ma, mb = adapter.load_ckpt(seed, int(a.a) if a.a.isdigit() else a.a), adapter.load_ckpt(seed, int(a.b) if a.b.isdigit() else a.b); import research as R; ro = R.research_obs(1000)
        mods = {n: float((pb - pa).norm() / (pa.norm() + 1e-12)) for (n, pa), (_, pb) in zip(ma.named_parameters(), mb.named_parameters())}
        with torch.no_grad(): od = float((ma(ro) - mb(ro)).abs().mean())
        out = {"a": a.a, "b": a.b, "mean_abs_action_change_on_research_states": round(od, 5), "largest_relative_weight_changes": sorted(({"param": k, "rel": round(v, 4)} for k, v in mods.items()), key=lambda x: -x["rel"])[:a.top]}
    _log(a.run, {"t": time.time(), "cmd": a.cmd, "args": {k: v for k, v in vars(a).items() if k not in ("run", "cmd") and v is not None}}); print(json.dumps(out, indent=1))


if __name__ == "__main__": main()
