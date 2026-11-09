"""GREEN RUN: end-to-end proof that the harness works on a completely out-of-the-box transformer (stock HF GPT-2, ~0.1M params, induction task).
Nothing here is mocked: the same CLIs/ledger/accept rule an LLM agent uses.  Two phases, because a (Haiku) agent makes the mechanical edits in between:

  python any_model/green_run.py setup     checks 1-9, starts the edit run, writes OUT/haiku_task.md
  (spawn a Haiku agent with OUT/haiku_task.md  -> it runs the edit commands through the real CLI)
  python any_model/green_run.py verify    checks the ledger the agent produced + controls + hook + agent loop + revert, writes OUT/report.{json,md}; exit 1 unless EVERY check passes

Haiku is used on purpose: the run tests that the MACHINERY works, not that an agent picks good edits.  All pass/fail decisions are taken from the ledger and from files, never from what an agent says.
Every check records its evidence; a check that cannot be verified is FAIL, not skipped."""
import os, sys, json, hashlib, subprocess, shutil, copy, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
OUT = os.environ.get("GREEN_DIR", os.path.join(os.path.dirname(ROOT), "evidence", "green_run")); PY = os.environ.get("HARNESS_PYTHON", sys.executable)
ENV = {**os.environ, "HARNESS_BACKEND": "lm_backend:LMBackend", "HARNESS_MODE": "llm", "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore", "HARNESS_PYTHON": PY}
RUN_A, RUN_B, RUN_C = (os.path.join(OUT, n) for n in ("run_haiku", "run_control", "run_agentloop"))
REQUIRED_TOOLS = {"scale_head", "scale_neurons", "head_temperature", "edge_circuit", "node_circuit_exact", "matrix_ops"}
CHECKS = []

# one edit of EVERY op type (valid for 2 blocks x 4 heads, d=64, d_ff=256), then deliberately invalid ones (must be rejected AND charged)
EDIT_TABLE = [
    ("scale_head", {"layer": 1, "head": 2, "gamma": 0.5}),
    ("scale_neurons", {"layer": 0, "neurons": [1, 2, 3], "gamma": 0.5}),
    ("prune_neurons", {"layer": 1, "neurons": [4, 5]}),
    ("head_temperature", {"layer": 0, "head": 3, "gamma": 1.5}),
    ("nudge_weights", {"matrix": "transformer.h.0.mlp.c_fc.weight", "axis": "col", "index": 7, "positions": [0, 1, 2], "deltas": [0.01, -0.01, 0.02]}),
    ("scale_slice", {"matrix": "transformer.h.1.mlp.c_fc.weight", "axis": "col", "indices": [3, 4], "gamma": 0.8}),
    ("bias_shift", {"bias": "transformer.h.0.mlp.c_fc.bias", "positions": [0, 1], "deltas": [0.02, -0.02]}),
    ("rank1_update", {"matrix": "transformer.h.1.attn.c_proj.weight", "rows": [0, 1], "cols": [0, 1], "u": [0.1, 0.2], "v": [0.1, -0.1], "alpha": 0.1}),
    ("blend_matrix", {"matrix": "transformer.h.0.attn.c_proj.weight", "ref_ckpt": 9, "frac": 0.5}),
    ("blend_slice", {"matrix": "transformer.h.0.attn.c_proj.weight", "axis": "row", "indices": list(range(16)), "ref_ckpt": 9, "frac": 1.0}),
    ("revert_delta", {"frac": 0.5, "ref_ckpt": 9}),
    ("rank_truncate", {"matrix": "transformer.h.1.mlp.c_proj.weight", "rank": 32}),
    ("shrink_subspace", {"matrix": "transformer.h.0.attn.c_attn.weight", "k": 2, "gamma": 0.9}),
    ("project_out_top_svd", {"matrix": "transformer.h.1.mlp.c_fc.weight", "k": 1}),
    ("composite", {"ops": [{"op": "scale_head", "cfg": {"layer": 0, "head": 1, "gamma": 2.0}}, {"op": "scale_neurons", "cfg": {"layer": 1, "neurons": [10, 11], "gamma": 1.1}}]}),
    ("scale_head", {"layer": 7, "head": 0, "gamma": 0.5}),                      # INVALID: layer out of range
    ("scale_head", {"layer": 0, "head": 0, "gamma": 99.0}),                     # INVALID: gamma out of range
]
N_VALID, N_INVALID = 15, 2


def check(cid, name, ok, evidence=""):
    CHECKS.append({"id": cid, "name": name, "status": "PASS" if ok else "FAIL", "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, default=str)[:900]})
    print(("PASS " if ok else "FAIL ") + f"[{cid}] {name}  {CHECKS[-1]['evidence'][:200]}", flush=True); return ok


def sh(args, env=None, timeout=1200):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env or ENV); return p.returncode, p.stdout, p.stderr


def cli(*a, timeout=1200):
    rc, out, err = sh([PY, os.path.join(ROOT, "agent_tools", "run.py")] + [str(x) for x in a], timeout=timeout)
    return rc, out, err


def jlast(out):
    s = out.strip(); i = min([x for x in (s.find("{"), s.find("[")) if x >= 0] or [0]); return json.loads(s[i:])


def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_pairs(path, n, kind="induction", seed=1234):
    import lm_backend as B
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        t = rng.permutation(B.V); seen = [int(x) for x in t[:5]]; unused = [int(x) for x in t[5:]]; j = int(rng.integers(0, 4))
        if kind == "induction":                     # corrupt: the query token never appeared -> nothing to look up
            clean = seen + [seen[j]]; corrupt = seen + [int(rng.choice(unused))]; correct = seen[j + 1]; wrong = int(rng.choice([x for x in seen if x != correct]))
        else:                                       # 'which-token' behaviour: corrupt repeats a DIFFERENT earlier token, so the right answer changes
            k = int(rng.choice([x for x in range(4) if x != j])); clean = seen + [seen[j]]; corrupt = seen + [seen[k]]; correct = seen[j + 1]; wrong = seen[k + 1]
            if wrong == correct: continue
        out.append({"name": f"{kind}{len(out)}", "clean": clean, "corrupt": corrupt, "correct": correct, "wrong": wrong})
    json.dump(out, open(path, "w")); return out


def finish_report():
    n_fail = sum(c["status"] != "PASS" for c in CHECKS); rep = {"all_green": n_fail == 0 and len(CHECKS) > 0, "n_checks": len(CHECKS), "n_fail": n_fail, "checks": CHECKS, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    return rep


def write_report(rep, name):
    json.dump(rep, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    L = [f"# Green run - {name}", "", f"**{'ALL GREEN' if rep['all_green'] else 'NOT GREEN'}**  ({rep['n_checks'] - rep['n_fail']}/{rep['n_checks']} checks pass)", "", "| id | check | status | evidence |", "|---|---|---|---|"]
    L += [f"| {c['id']} | {c['name']} | {c['status']} | {c['evidence'][:160].replace('|', '/').replace(chr(10), ' ')} |" for c in rep["checks"]]
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------------------------------ phase 1
def setup():
    import lm_backend as B, structure as S, spec_map as SM, acclimate as A, patching as P, prompt_pairs as PP, llm_ops
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    root = B.default_root(); metap = os.path.join(root, "policy_s0", "meta.json")
    if not os.path.exists(metap): B.train_and_save(root, quiet=True)
    meta = json.load(open(metap)); be = B.LMBackend(root); m9, mf = be.load_ckpt(0, 9), be.load_ckpt(0, "final")
    check("1.1", "model is out-of-the-box HF GPT-2 with <= 1M parameters", type(m9).__name__ == "GPT2LMHeadModel" and meta["params"] <= 1_000_000, f"{type(m9).__name__}, {meta['params']} params")
    check("1.2", "model learned the task and `final` carries real planted damage", meta["acc_ckpt009"] >= 0.9 and meta["acc_final_damaged"] <= meta["acc_ckpt009"] - 0.2, f"ckpt_009 acc {meta['acc_ckpt009']:.3f}; final {meta['acc_final_damaged']:.3f}")
    check("1.3", "all 11 checkpoints exist", all(os.path.exists(be.ckpt_path(0, w)) for w in list(range(10)) + ["final"]), "ckpt_000..009 + final")
    json.dump({"final_sha": sha(be.ckpt_path(0, "final")), "ckpt9_sha": sha(be.ckpt_path(0, 9))}, open(os.path.join(OUT, "checkpoint_hashes.json"), "w"))
    pa, pb = os.path.join(OUT, "pairs_induction.json"), os.path.join(OUT, "pairs_which.json"); make_pairs(pa, 24, "induction"); make_pairs(pb, 24, "which", seed=77)
    # ---- acclimation (in-process): both checkpoints
    enc = PP.encode(json.load(open(pa))); keep, rep = PP.check(m9, enc); clean, corrupt, mf_ = PP.batch([p for p in keep if p["clean"].shape[0] == 6])
    for nm, mod in (("ckpt_009", m9), ("final", mf)):
        r = A.acclimate(copy.deepcopy(mod), clean, corrupt, mf_, out_path=os.path.join(OUT, f"acclimation_{nm}.json"))
        check(f"2.{nm}", f"acclimation battery on {nm}: no FAIL and every tool verified", r["n_fail"] == 0 and REQUIRED_TOOLS <= set(r["verified_tools"]), f"n_fail={r['n_fail']} verified={sorted(r['verified_tools'])}")
    # ---- out-of-the-box CLI path: save as a plain HF directory, then the one-command protocol
    hf = os.path.join(OUT, "hf_model"); mf.save_pretrained(hf); mp = os.path.join(ROOT, "any_model", "map_tool.py"); mout = os.path.join(OUT, "map_out")
    rc, o, e = sh([PY, mp, f"hf:{hf}", "inspect"]); check("3.1", "map_tool inspect on a saved HF directory", rc == 0 and "c_attn" in o, o[:120])
    rc, o, e = sh([PY, mp, f"hf:{hf}", "propose", "--out", mout]); spec = json.load(open(os.path.join(mout, "spec.json"))) if rc == 0 else None; check("3.2", "map_tool propose writes a ModelMap spec", spec is not None, str(spec)[:150])
    rc, o, e = sh([PY, mp, f"hf:{hf}", "acclimate", "--out", os.path.join(OUT, "map_acc_builtin"), "--pairs", pa]); check("3.3", "map_tool acclimate (built-in family) exits 0", rc == 0, (o + e)[-300:])
    rc, o, e = sh([PY, mp, f"hf:{hf}", "acclimate", "--out", os.path.join(OUT, "map_acc_spec"), "--spec", os.path.join(mout, "spec.json"), "--pairs", pa]); check("3.4", "map_tool acclimate with the PROPOSED spec exits 0", rc == 0, (o + e)[-300:])
    # ---- read tools through the real CLI on a fresh run
    rc, o, e = cli("loop", "start", RUN_A, "--budget", 20); check("4.0", "loop start (budget 20) on the LM backend", rc == 0 and '"budget": 20' in o, (o + e)[-200:])
    rc, o, e = cli("loop", "ops", RUN_A); ops_txt = o
    names = ["scale_head", "scale_neurons", "prune_neurons", "head_temperature", "nudge_weights", "scale_slice", "bias_shift", "rank1_update", "blend_matrix", "blend_slice", "revert_delta", "rank_truncate", "shrink_subspace", "project_out_top_svd", "composite"]
    check("4.1", "op listing is model-true and contains every op", rc == 0 and all(f"- {n}:" in ops_txt for n in names) and "layer: int, 0..1" in ops_txt and "head: int, 0..3" in ops_txt and "Conv1D" in ops_txt, f"{len(ops_txt)} chars; Nav-only ops present: {[n for n in ('scale_token_embed','head_transplant') if n in ops_txt]}")
    check("4.1b", "Nav-only ops are NOT offered in LM mode", "scale_token_embed" not in ops_txt and "head_transplant" not in ops_txt and "ridge_head" not in ops_txt, "")
    rc, o, e = cli("tools", RUN_A, "inventory"); inv = jlast(o) if rc == 0 else {}; check("4.2", "inventory lists blocks, heads, MLP banks, matrices with roles", rc == 0 and inv.get("family") == "gpt2" and len(inv.get("components", [])) == 4 and len(inv.get("matrices", {})) >= 8, f"family={inv.get('family')} comps={len(inv.get('components', []))} mats={len(inv.get('matrices', {}))}")
    rc, o, e = cli("tools", RUN_A, "weights", "--matrix", "transformer.h.0.attn.c_proj.weight", "--rows", "0:2", "--cols", "0:3"); w = jlast(o) if rc == 0 else {}; check("4.3", "raw weights readable (raw view default ON)", rc == 0 and len(w.get("values", [])) == 2, str(w)[:150])
    env0 = {**ENV, "HARNESS_RAW_VIEW": "0"}; rc, o, e = sh([PY, os.path.join(ROOT, "agent_tools", "run.py"), "tools", RUN_A, "weights", "--matrix", "transformer.h.0.attn.c_proj.weight"], env=env0); check("4.4", "raw-view ablation toggle hides the values", rc == 0 and "DISABLED" in o and '"values"' not in o, o[:160])
    rc, o, e = cli("tools", RUN_A, "diff", "--a", 0, "--b", 9); d = jlast(o) if rc == 0 else {}; check("4.5", "checkpoint diff", rc == 0 and len(d.get("matrices", [])) > 0, str(d)[:140])
    rc, o, e = cli("tools", RUN_A, "pairs_check", "--file", pa, "--ckpt", 9); pc = jlast(o) if rc == 0 else {}; check("5.1", "prompt-pair quality check keeps most pairs on the healthy checkpoint", rc == 0 and pc.get("n_usable", 0) >= 15, f"{pc.get('n_usable')}/{pc.get('n_total')} usable")
    bad = [{"name": "bad_len", "clean": [0, 1, 2, 3, 4, 0], "corrupt": [0, 1, 2], "correct": 1, "wrong": 2}, {"name": "same_ans", "clean": [0, 1, 2, 3, 4, 0], "corrupt": [0, 1, 2, 3, 4, 5], "correct": 1, "wrong": 1}]
    json.dump(bad, open(os.path.join(OUT, "pairs_bad.json"), "w")); rc, o, e = cli("tools", RUN_A, "pairs_check", "--file", os.path.join(OUT, "pairs_bad.json")); bc = jlast(o) if rc == 0 else {}
    check("5.2", "bad pairs are rejected with reasons", rc == 0 and bc.get("n_usable") == 0 and all(r.get("reason") for r in bc.get("rows", [])), str([r.get("reason") for r in bc.get("rows", [])])[:200])
    # ---- circuits on the HEALTHY checkpoint and on the damaged target; planted heads must be among the heads found
    planted = {tuple(h) for h in meta["planted_heads"]}
    rc, o, e = cli("tools", RUN_A, "circuit", "--file", pa, "--name", "ind_ckpt9", "--ckpt", 9, "--top_frac", 0.95, "--edges"); c9 = jlast(o) if rc == 0 else {}
    h9 = {(n["layer"], n["index"]) for n in c9.get("circuit", []) if n["kind"] == "head"}; f9 = c9.get("faithfulness", {})
    check("6.1", "node circuit on the healthy checkpoint recovers BOTH planted heads", rc == 0 and planted <= h9, f"found heads {sorted(h9)}; planted {sorted(planted)}; faithfulness {f9}")
    check("6.2", "circuit ablation destroys the behaviour (faithfulness.ablate_drop >= 0.7)", f9.get("ablate_drop", 0) >= 0.7, f9)
    check("6.3", "edge attribution returns edges with exact direct-path check", len(c9.get("top_edges", [])) >= 3 and all("exact_direct_path" in x for x in c9.get("top_edges", [])), str(c9.get("top_edges", [])[:2]))
    rc, o, e = cli("tools", RUN_A, "circuit", "--file", pa, "--name", "ind_target", "--top_frac", 0.95); ct = jlast(o) if rc == 0 else {}; ht = {(n["layer"], n["index"]) for n in ct.get("circuit", []) if n["kind"] == "head"}
    check("6.4", "circuit on the damaged TARGET contains at least one planted head", rc == 0 and len(planted & ht) >= 1, f"found {sorted(ht)}; planted {sorted(planted)}")
    rc, o, e = cli("tools", RUN_A, "circuit", "--file", pb, "--name", "which_ckpt9", "--ckpt", 9, "--top_frac", 0.95); cw = jlast(o) if rc == 0 else {}
    check("6.5", "second behaviour circuit found", rc == 0 and len(cw.get("circuit", [])) >= 1, f"{len(cw.get('circuit', []))} nodes, ablate_drop {cw.get('faithfulness', {}).get('ablate_drop')}")
    rc, o, e = cli("tools", RUN_A, "overlap", "--names", "ind_ckpt9,which_ckpt9"); ov = jlast(o) if rc == 0 else {}
    check("6.6", "overlap reports Jaccard, shared core and specific nodes", rc == 0 and {"jaccard", "core_shared_by_all", "specific"} <= set(ov), f"core={ov.get('core_shared_by_all')}")
    rc, o, e = cli("tools", RUN_A, "proposals", "--name", "ind_target", "--gamma", 0.0); pr = jlast(o) if rc == 0 else {}
    check("6.7", "edit proposals are produced for circuit nodes only and are valid op configs", rc == 0 and len(pr.get("proposals", [])) >= 1 and all(p["op"] in ("scale_head", "scale_neurons") for p in pr.get("proposals", [])), str(pr.get("proposals", [])[:2]))
    # ---- the Haiku task
    task = ["# Task for the editing agent (mechanical)", "", "You are testing a weight-editing harness. Your job is ONLY to run the commands below, in order, exactly as written, and record what the harness prints.", "You are NOT judging whether an edit is good. Do not change any config. Do not run any other command. Do not read or modify files in the repository.", "",
            f"Working directory: {ROOT}", f"Python: {PY}", "Every command needs these environment variables (prefix each command with them):", "  HARNESS_BACKEND=lm_backend:LMBackend HARNESS_MODE=llm OMP_NUM_THREADS=1", "",
            f"Run directory: {RUN_A}", f"Results file you must write: {os.path.join(OUT, 'haiku_results.jsonl')}  (one JSON line per command: {{\"i\": index, \"op\": name, \"output\": first 600 chars of what the command printed}})", "",
            "## Step 0 (read-only, no charge): see the allowed edits", f"  cd {ROOT} && HARNESS_BACKEND=lm_backend:LMBackend HARNESS_MODE=llm OMP_NUM_THREADS=1 {PY} agent_tools/run.py loop ops {RUN_A}", "",
            "## Step 1: one charged edit per line below (17 commands). Each prints a JSON verdict; an 'ERROR' or rejection is a normal, expected outcome for the last two - just record it and continue.", ""]
    for i, (op, cfg) in enumerate(EDIT_TABLE): task.append(f"{i + 1}. op={op}\n   cd {ROOT} && HARNESS_BACKEND=lm_backend:LMBackend HARNESS_MODE=llm OMP_NUM_THREADS=1 {PY} agent_tools/run.py loop propose {RUN_A} --op {op} --cfg '{json.dumps(cfg)}'")
    task += ["", "## Step 2: status", f"  cd {ROOT} && HARNESS_BACKEND=lm_backend:LMBackend HARNESS_MODE=llm OMP_NUM_THREADS=1 {PY} agent_tools/run.py loop status {RUN_A}", "",
             "When done, reply with ONE line: how many of the 17 commands you ran, and how many printed something that looks like an error. Do not run `loop finish`."]
    open(os.path.join(OUT, "haiku_task.md"), "w").write("\n".join(task) + "\n")
    rep = finish_report(); rep["phase"] = "setup"; write_report(rep, "report_setup"); print(f"\nsetup: {rep['n_checks'] - rep['n_fail']}/{rep['n_checks']} pass. Haiku task: {os.path.join(OUT, 'haiku_task.md')}")
    return 0 if rep["all_green"] else 1


# ------------------------------------------------------------------------------------------------------------------------------ phase 2
def verify():
    import lm_backend as B, loop as LP, ops as OPS, llm_ops, more_ops, adapter, train_hook as TH, dispatch as D, agent_loop as AL, torch.nn as nn
    adapter.set_backend(B.LMBackend())      # in-process checks must use the LM backend too (the CLIs get it from HARNESS_BACKEND)
    setup_rep = json.load(open(os.path.join(OUT, "report_setup.json"))); CHECKS.extend(setup_rep["checks"]); meta = json.load(open(os.path.join(B.default_root(), "policy_s0", "meta.json"))); planted = [tuple(h) for h in meta["planted_heads"]]
    # ---- A: the Haiku run, judged from the ledger
    r = LP.Run(RUN_A); ents = r.entries(); its = r.of("iterate"); strikes = r.strikes()
    check("7.1", "ledger hash chain intact after the agent's run", r.verify() == [], str(r.verify())[:200])
    check("7.2", "agent ran all 17 commands (15 valid evaluated + 2 invalid charged)", len(its) == N_VALID and len(strikes) == N_INVALID and r.left() == 20 - 17, f"evaluated={len(its)} rejected_charged={len(strikes)} budget_left={r.left()}")
    ops_seen = set()
    for it in its: ops_seen.add(it.get("op"))
    want = {n for n, _ in EDIT_TABLE[:N_VALID]}
    check("7.3", "every op type was evaluated through the real loop", want <= ops_seen, f"missing={sorted(want - ops_seen)}")
    ok_fields = all({"d_success", "d_success_lo", "d_success_hi", "id_drop", "accepted"} <= set(it.get("metrics", {})) for it in its)
    check("7.4", "every evaluated edit carries a paired report (d_success, CI, id drop, accepted)", ok_fields, str(its[0].get("metrics"))[:300])
    check("7.5", "invalid configs rejected AND charged", len(strikes) == N_INVALID, str(strikes[0])[:200] if strikes else "none")
    hr = [json.loads(l) for l in open(os.path.join(OUT, "haiku_results.jsonl"))] if os.path.exists(os.path.join(OUT, "haiku_results.jsonl")) else []
    check("7.6", "the Haiku agent wrote its results file (17 lines)", len(hr) >= 17, f"{len(hr)} lines")
    hashes = json.load(open(os.path.join(OUT, "checkpoint_hashes.json"))); be = B.default_root()
    check("7.7", "edits never mutate the stored target (final.pt hash unchanged)", sha(os.path.join(be, "policy_s0", "final.pt")) == hashes["final_sha"], "")
    rc, o, e = cli("loop", "status", RUN_A); st = jlast(o) if rc == 0 else {}
    check("7.8", "loop status: counts match and no ledger problems", st.get("problems") == [] and st.get("evaluated") == N_VALID and st.get("rejected") == N_INVALID, str(st))
    n_acc = sum(bool(it["metrics"]["accepted"]) for it in its); rc, o, e = cli("loop", "finish", RUN_A, timeout=1800); fj = jlast(o) if rc == 0 else {}
    check("7.9", "finish scores the selection on HOLD (3 seeds) and reports the paired delta", rc == 0 and "hold_delta_mean" in fj and len(fj.get("per_seed", [])) == 3, f"accepted_edits={n_acc}; chosen={fj.get('op')}; hold_delta={fj.get('hold_delta_mean')}; beats_noise={fj.get('hold_gain_beats_noise')}")
    n_score = len(LP.Run(RUN_A).of("score")); rc2, o2, e2 = cli("loop", "finish", RUN_A, timeout=600)
    check("7.10", "HOLD is read only once: a second finish does not re-score", rc2 == 0 and n_score == 1 and len(LP.Run(RUN_A).of("score")) == 1, f"score entries before/after second finish: {n_score}/{len(LP.Run(RUN_A).of('score'))}")
    # ---- B: controls with a scripted overseer (no agent): known-good repair must be ACCEPTED, a random head edit must be REJECTED
    shutil.rmtree(RUN_B, ignore_errors=True); rc, o, e = cli("loop", "start", RUN_B, "--budget", 6)
    good = {"ops": [{"op": "scale_head", "cfg": {"layer": l, "head": h, "gamma": round(1 / meta["damage_gamma"], 3)}} for l, h in planted]}
    rc, o, e = cli("loop", "propose", RUN_B, "--op", "composite", "--cfg", json.dumps(good)); gj = o
    g1 = jlast(gj) if rc == 0 else {}; check("8.1", "POSITIVE control: undoing the planted damage is ACCEPTED by the real accept rule", g1.get("accepted") is True, f"d_success={g1.get('d_success')} CI=[{g1.get('d_success_lo')},{g1.get('d_success_hi')}] id_drop={g1.get('id_drop')} flags={g1.get('flags')}")
    others = [(l, h) for l in range(2) for h in range(4) if (l, h) not in planted]; l, h = others[-1]
    rc, o, e = cli("loop", "propose", RUN_B, "--op", "scale_head", "--cfg", json.dumps({"layer": l, "head": h, "gamma": 0.0}))
    g2 = jlast(o) if rc == 0 else {}; check("8.2", "NEGATIVE control: ablating an unrelated head is REJECTED", g2.get("accepted") is False, f"head ({l},{h}) d_success={g2.get('d_success')} flags={g2.get('flags')}")
    rc, o, e = cli("loop", "propose", RUN_B, "--op", "revert_delta", "--cfg", json.dumps({"frac": 1.0, "ref_ckpt": 9}))
    g3 = jlast(o) if rc == 0 else {}; check("8.3", "reverting the whole model to the healthy checkpoint is ACCEPTED", g3.get("accepted") is True, f"d_success={g3.get('d_success')}")
    rc, o, e = cli("loop", "finish", RUN_B, timeout=1800); fb = jlast(o) if rc == 0 else {}
    check("8.4", "control run: HOLD (distribution-shifted N=6) improves for the accepted repair, beyond seed noise", rc == 0 and fb.get("hold_delta_mean", 0) > 0.05 and fb.get("hold_gain_beats_noise") is True, f"hold_delta={fb.get('hold_delta_mean')} CI=[{fb.get('hold_delta_lo')},{fb.get('hold_delta_hi')}] base={fb.get('hold_base_mean')} edit={fb.get('hold_edit_mean')}")
    check("8.5", "control ledger: chain intact, one score entry", LP.Run(RUN_B).verify() == [] and len(LP.Run(RUN_B).of("score")) == 1, "")
    # ---- reversibility
    llm_ops.enable(); more_ops.enable_value_ops(); more_ops.enable_more_value_ops()
    try:
        m = adapter.load_ckpt(0, "final"); x = adapter.probe_obs("ID", 64); base = adapter.forward(m, x).detach()
        e1 = OPS.apply("scale_head", {"layer": 0, "head": 1, "gamma": 4.0}, m); e2 = OPS.apply("scale_head", {"layer": 0, "head": 1, "gamma": 0.25}, e1); err = float((adapter.forward(e2, x) - base).abs().max())
        check("9.1", "inverse edit (gamma then 1/gamma) restores outputs", err < 1e-4, f"max |logit diff| = {err:.2e}")
        e3 = OPS.apply("nudge_weights", {"matrix": "transformer.h.0.mlp.c_fc.weight", "axis": "col", "index": 3, "positions": [0, 1], "deltas": [0.05, -0.05]}, m); e4 = OPS.apply("nudge_weights", {"matrix": "transformer.h.0.mlp.c_fc.weight", "axis": "col", "index": 3, "positions": [0, 1], "deltas": [-0.05, 0.05]}, e3)
        err = float((adapter.forward(e4, x) - base).abs().max()); check("9.2", "value-level delta is exactly invertible", err < 1e-5, f"max diff {err:.2e}")
        ref = {k: adapter.load_ckpt(0, k) for k in range(10)}; rv = OPS.apply("revert_delta", {"frac": 1.0, "ref_ckpt": 9}, m, {"ref_models": ref}); err = float((adapter.forward(rv, x) - adapter.forward(ref[9], x)).abs().max())
        check("9.3", "revert_delta frac=1 reproduces the reference checkpoint exactly", err < 1e-5, f"max diff {err:.2e}")
        check("9.4", "edits deep-copy: the input model object is unchanged", torch.equal(adapter.forward(m, x), base), "")
    finally: llm_ops.disable()
    # ---- training hook on an LM
    def train(tmp, agent, ev, unver=False, steps=300):
        torch.manual_seed(0); mm = B.make_model(); opt = torch.optim.Adam(mm.parameters(), 2e-3); rng = np.random.default_rng(0)
        hook = TH.HarnessHook(tmp, steps, [TH.AtFraction(0.2)], agent, ev, watch_metric="val_loss", watch_steps=20, cooldown_steps=10, allow_unverified=unver); applied = 0; vl = None
        for s in range(steps):
            xb, yb = B.make_batch(64, rng); loss = nn.functional.cross_entropy(B.logits_of(mm, xb)[:, -1], yb); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): xv, yv = B.make_batch(256, np.random.default_rng(5)); vl = float(nn.functional.cross_entropy(B.logits_of(mm, xv)[:, -1], yv))
            applied += bool(hook.step(s, {"train_loss": float(loss), "val_loss": vl}, mm, opt))
        return mm, hook, vl, applied
    def ledger_ok(path):
        prev = "0" * 64
        for l in open(path):
            r_ = json.loads(l)
            if r_["prev"] != prev or r_["sha"] != hashlib.sha256(json.dumps({k: r_[k] for k in ("t", "kind", "payload", "prev")}, sort_keys=True, default=str).encode()).hexdigest(): return False
            prev = r_["sha"]
        return True
    val_acc = lambda mm: B.accuracy(mm, "VAL_SEEN", 500)
    def harmless(d, model): sd = {k: v.clone() for k, v in model.state_dict().items()}; sd["transformer.h.0.mlp.c_fc.bias"][0] += 0.0; return sd
    hd = os.path.join(OUT, "hook"); shutil.rmtree(hd, ignore_errors=True)
    mm, hook, vl, applied = train(os.path.join(hd, "accept"), harmless, lambda a, b: {"accept": val_acc(b) >= val_acc(a) - 0.02})
    kinds = [json.loads(l)["kind"] for l in open(os.path.join(hd, "accept", "ledger.jsonl"))]
    check("10.1", "hook fires at 20% of training, edit passes the evaluator gate and is applied", applied == 1 and kinds[:3] == ["trigger", "evaluated", "applied"], str(kinds))
    check("10.2", "hook ledger hash chain valid", ledger_ok(os.path.join(hd, "accept", "ledger.jsonl")), "")
    mm, hook, vl, applied = train(os.path.join(hd, "poison"), lambda d, model: {k: v * 0 for k, v in model.state_dict().items()}, lambda a, b: {"accept": val_acc(b) >= val_acc(a) - 0.02})
    check("10.3", "poison edit (all weights zeroed) is REJECTED by the evaluator gate", applied == 0, "")
    mm, hook, vl, applied = train(os.path.join(hd, "noeval"), harmless, None)
    check("10.4", "no evaluator => edit refused", applied == 0 and "no evaluator" in open(os.path.join(hd, "noeval", "ledger.jsonl")).read(), "")
    def harm(d, model): return {k: (v + 3.0 if v.dtype.is_floating_point else v) for k, v in model.state_dict().items()}
    mm, hook, vl, applied = train(os.path.join(hd, "rollback"), harm, None, unver=True, steps=400)
    kinds = [json.loads(l)["kind"] for l in open(os.path.join(hd, "rollback", "ledger.jsonl"))]
    check("10.5", "damaging edit waved through is AUTOMATICALLY ROLLED BACK by the watch metric", "rollback" in kinds, str(kinds))
    mm2, hook2, _, _ = train(os.path.join(hd, "manual"), harmless, lambda a, b: {"accept": True}); n = hook2.revert(mm2); check("10.6", "manual revert() restores the pre-edit snapshot and is logged", n == 1 and "manual_revert" in open(os.path.join(hd, "manual", "ledger.jsonl")).read(), "")
    # ---- a scripted 'agent' driving the neutral tool layer end-to-end (any chat() callable plugs in here)
    shutil.rmtree(RUN_C, ignore_errors=True); cli("loop", "start", RUN_C, "--budget", 4); pa = os.path.join(OUT, "pairs_induction.json")
    script = [("note", {"kind": "hypothesis", "text": "the damaged target has weak early heads"}), ("inventory", {}), ("pairs_check", {"file": pa}), ("circuit", {"file": pa, "name": "c1", "top_frac": 0.95}),
              ("proposals", {"name": "c1", "gamma": 2.0}), ("weights", {"matrix": "transformer.h.0.attn.c_proj.weight", "rows": "0:2", "cols": "0:3"}),
              ("propose", {"op": "composite", "cfg": good}), ("note", {"kind": "result", "text": "composite repair evaluated"}), ("status", {}), ("state", {})]
    it = iter(script); n_called = {"n": 0}
    def chat(system, msgs, tools):
        try: nm, ar = next(it)
        except StopIteration: return {"text": "done", "tool_calls": []}
        assert nm in {t["name"] for t in tools}; n_called["n"] += 1; return {"text": "", "tool_calls": [{"id": f"c{n_called['n']}", "name": nm, "args": ar}]}
    os.environ["HARNESS_MODE"] = "llm"; os.environ["HARNESS_BACKEND"] = "lm_backend:LMBackend"
    res = AL.run_agent(chat, RUN_C, "system", "task", max_turns=30, reset_every=5)
    tr = [json.loads(l) for l in open(os.path.join(RUN_C, "agent_transcript.jsonl"))]; errs = [t for t in tr if "result" in json.dumps(t) and '"error"' in json.dumps(t)]
    check("11.1", "scripted agent loop (neutral tools, any chat() callable) ran 10 tool calls incl. a context reset", res.get("finished") and n_called["n"] == 10 and any(t.get("event") for t in tr), f"turns={res.get('turns')} errors={len(errs)}")
    check("11.2", "no tool call in the agent loop returned an error", len(errs) == 0, str(errs[:1])[:400])
    check("11.3", "agent-loop run ledger intact and its charged proposal is in it", LP.Run(RUN_C).verify() == [] and len(LP.Run(RUN_C).of("iterate")) == 1, "")
    check("11.4", "tool specs convert to OpenAI and Anthropic formats", len(D.to_openai(D.active())) == len(D.TOOLS_LLM) == len(D.to_anthropic(D.active())), f"{len(D.TOOLS_LLM)} tools")
    # ---- circuit growth (necessary -> sufficient) and macro-circuit union, through the neutral tool layer
    g1 = D.call(RUN_C, "grow", {"file": pa, "name": "g_ind", "target": 0.8}); g2 = D.call(RUN_C, "grow", {"file": os.path.join(OUT, "pairs_which.json"), "name": "g_which", "target": 0.8})
    check("12.1", "grow: circuit made SUFFICIENT (keep-only >= 0.8) and still necessary (ablate_drop >= 0.7)", g1.get("reached_target") is True and g1.get("sufficiency", 0) >= 0.8 and g1.get("ablate_drop", 0) >= 0.7 and g1.get("seed_sufficiency", 1) < 0.3, {k: g1.get(k) for k in ("seed_sufficiency", "sufficiency", "ablate_drop", "n_nodes", "n_nodes_if_pruned_in_random_order")})
    check("12.2", "grow works for a second behaviour", g2.get("reached_target") is True, {k: g2.get(k) for k in ("sufficiency", "n_nodes")})
    u = D.call(RUN_C, "union", {"names": "g_ind,g_which", "min_frac": 1.0}); check("12.3", "union returns the shared macro circuit", isinstance(u.get("core"), list) and len(u["core"]) >= 1, f"{len(u.get('core', []))} shared nodes")
    # ---- baseline activity map + co-active neighbours (candidates are verified against equal-size random units)
    b = D.call(RUN_C, "baseline", {"name": "b_norm", "n": 512}); pl = b.get("per_layer", {})
    check("13.1", "baseline map: 512 normal prompts, per-layer stats for every head bank and MLP bank, clusters", b.get("n_prompts") == 512 and set(pl) == {"head.0", "head.1", "neuron.0", "neuron.1"} and len(b.get("largest_clusters", [])) >= 1 and all(pl[k]["dead"] == 0 and pl[k]["always_on"] == 0 for k in ("head.0", "head.1")), {k: (v["units"], v["dead"], v["always_on"]) for k, v in pl.items()})
    nb = D.call(RUN_C, "neighbors", {"name": "b_norm", "circuit": "g_ind", "file": pa, "min_corr": 0.3, "k": 32}); vv = nb.get("verification", {})
    check("13.2", "neighbors: ranked co-active candidates, none inside the circuit, verified against equal-size random groups", nb.get("n_candidates", 0) >= 8 and {"gain_coactive", "gain_random_mean", "gain_random_std", "informative"} <= set(vv), {k: vv.get(k) for k in ("k", "gain_coactive", "gain_random_mean", "informative")})
    gb = D.call(RUN_C, "grow", {"file": pa, "name": "g_with_base", "target": 0.8, "baseline": "b_norm", "min_corr": 0.3}); check("13.3", "grow accepts the baseline map's co-active groups and still reaches the target", gb.get("reached_target") is True, {k: gb.get(k) for k in ("sufficiency", "n_nodes")})
    # ---- measurement caveats: present by default, one short field per read-tool output, removable by the switch
    r1 = D.call(RUN_C, "inventory", {}); r2 = D.call(RUN_C, "weights", {"matrix": "transformer.h.0.attn.c_proj.weight", "rows": "0:1", "cols": "0:2"}); r3 = D.call(RUN_C, "overlap", {"names": "g_ind,g_which"})
    check("14.1", "caveat field present on read-tool outputs by default (inventory, weights, overlap, baseline, grow, neighbors)", all("caveat" in x and len(x["caveat"].split()) <= 30 for x in (r1, r2, r3, b, gb, nb)), [x.get("caveat", "")[:40] for x in (r1, r2, r3)])
    os.environ["HARNESS_CAVEATS"] = "0"
    try: q1 = D.call(RUN_C, "inventory", {}); q2 = D.call(RUN_C, "weights", {"matrix": "transformer.h.0.attn.c_proj.weight", "rows": "0:1", "cols": "0:2"})
    finally: os.environ.pop("HARNESS_CAVEATS", None)
    check("14.2", "HARNESS_CAVEATS=0 removes every caveat and leaves the data identical", "caveat" not in q1 and "caveat" not in q2 and {k: v for k, v in r1.items() if k != "caveat"} == q1, "")
    check("14.3", "no caveat contains a digit (no toy numbers leak into prompts)", not any(any(ch.isdigit() for ch in v) for v in __import__("caveats").CAVEATS.values()), "")
    # ---- uncharged screening + strength sweep (research episodes; budget and charged ledger untouched)
    left0 = LP.Run(RUN_C).left(); n_it0 = len(LP.Run(RUN_C).of("iterate"))
    sc = D.call(RUN_C, "screen", {"op": "composite", "cfg": good}); sw = D.call(RUN_C, "sweep", {"op": "scale_head", "cfg": {"layer": planted[0][0], "head": planted[0][1], "gamma": 1.0}, "param": "gamma", "values": [0.5, 1.0, 2.0, 3.0, 5.0]})
    charged = LP.Run(RUN_C).of("iterate")[0]["metrics"]["d_success"]
    check("15.1", "screen is free: charged budget and ledger unchanged, screens chain intact", LP.Run(RUN_C).left() == left0 and len(LP.Run(RUN_C).of("iterate")) == n_it0 and LP.Run(RUN_C).verify() == [] and LP._screen_log(RUN_C).verify() == [], f"budget left {LP.Run(RUN_C).left()}; screens logged {len(LP._screen_log(RUN_C).entries())}")
    check("15.2", "screen agrees in sign with the charged judge for the known-good repair", sc.get("d_success", -1) > 0 and charged > 0 and "research" in sc.get("split", ""), f"free {sc.get('d_success')} vs charged {charged}")
    check("15.3", "sweep returns a curve over the research split and names a best value; caveat attached", len(sw.get("curve", [])) == 5 and sw.get("best_value") is not None and sw["best_value"] > 1.0 and "caveat" in sw, {"best": sw.get("best_value"), "curve": [(c["value"], c.get("d_success")) for c in sw.get("curve", [])]})
    check("15.4", "screens never use the judge's splits", LP.SCREEN_SPLIT not in ("VAL_SEEN", "ID_EVAL", "CANARY", "HOLD_OOD", "VAL_OOD"), LP.SCREEN_SPLIT)
    rep = finish_report(); write_report(rep, "report"); print(f"\nFINAL: {rep['n_checks'] - rep['n_fail']}/{rep['n_checks']} pass -> {'ALL GREEN' if rep['all_green'] else 'NOT GREEN'}"); return 0 if rep["all_green"] else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("setup", "verify"): sys.exit(__doc__)
    os.environ.update({k: v for k, v in ENV.items() if k.startswith("HARNESS_") or k == "OMP_NUM_THREADS"}); sys.exit(setup() if sys.argv[1] == "setup" else verify())
