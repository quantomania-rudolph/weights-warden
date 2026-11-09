import os, sys, json, hashlib, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import train_hook as TH

def _train(tmp, agent, evaluator, allow_unverified=False, steps=400, triggers=None):
    torch.manual_seed(0); m = nn.Linear(4, 1); opt = torch.optim.Adam(m.parameters(), 0.05); X = torch.randn(256, 4); y = X @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]); Xv = torch.randn(64, 4); yv = Xv @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    hook = TH.HarnessHook(str(tmp), steps, triggers or [TH.AtFraction(0.25)], agent, evaluator, watch_metric="val_loss", watch_steps=20, cooldown_steps=10, allow_unverified=allow_unverified); applied = 0
    for s in range(steps):
        loss = (m(X) - y).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step(); vl = float((m(Xv) - yv).pow(2).mean())
        applied += bool(hook.step(s, {"train_loss": float(loss), "val_loss": vl}, m, opt))
    return m, hook, vl, applied

def _ledger_ok(path):
    prev = "0" * 64
    for l in open(path):
        r = json.loads(l); assert r["prev"] == prev; assert r["sha"] == hashlib.sha256(json.dumps({k: r[k] for k in ("t", "kind", "payload", "prev")}, sort_keys=True, default=str).encode()).hexdigest(); prev = r["sha"]

def test_fraction_trigger_snapshots_applies_good_edit_and_ledger_chains(tmp_path):
    seen = {}
    def agent(d, model): seen["files"] = sorted(os.listdir(d)); sd = {k: v.clone() for k, v in model.state_dict().items()}; sd["weight"][0, 0] += 0.0; return sd
    m, hook, vl, applied = _train(tmp_path, agent, lambda a, b: {"accept": True}); assert applied == 1 and {"pre.pt", "curves.json", "request.json"} <= set(seen["files"]); _ledger_ok(tmp_path / "ledger.jsonl")
    kinds = [json.loads(l)["kind"] for l in open(tmp_path / "ledger.jsonl")]; assert kinds[:3] == ["trigger", "evaluated", "applied"] and "kept" in kinds

def test_poison_evaluator_rejects_and_nonfinite_and_no_evaluator_refused(tmp_path):
    bad = lambda d, model: {k: v * 0 for k, v in model.state_dict().items()}
    m, hook, vl, applied = _train(tmp_path / "a", bad, lambda a, b: {"accept": False}); assert applied == 0
    nan = lambda d, model: {k: v * float("nan") for k, v in model.state_dict().items()}; _, _, _, applied = _train(tmp_path / "b", nan, lambda a, b: {"accept": True}); assert applied == 0
    _, _, _, applied = _train(tmp_path / "c", bad, None); assert applied == 0 and "no evaluator" in open(tmp_path / "c" / "ledger.jsonl").read()

def test_rollback_when_watched_metric_gets_worse(tmp_path):
    harm = lambda d, model: {k: v + 3.0 for k, v in model.state_dict().items()}           # damaging edit, waved through (allow_unverified) -> post-edit watch must undo it
    m, hook, vl, applied = _train(tmp_path, harm, None, allow_unverified=True, steps=300); kinds = [json.loads(l)["kind"] for l in open(tmp_path / "ledger.jsonl")]; assert "applied" in kinds and "rollback" in kinds

def test_plateau_and_gap_triggers():
    h = {"v": [(i, 1.0) for i in range(40)]}; assert TH.Plateau("v", window=10).check(40, 100, h) and not TH.Plateau("v", window=30).check(40, 100, h)
    assert TH.GapAbove("a", "b", 1.5).check(0, 1, {"a": [(0, 1.0)], "b": [(0, 2.0)]}) and not TH.GapAbove("a", "b", 1.5).check(0, 1, {"a": [(0, 1.0)], "b": [(0, 1.2)]})
    assert TH.Divergence("l").check(0, 1, {"l": [(i, 1.0) for i in range(11)] + [(11, 9.0)]})

def test_cmd_agent_roundtrip(tmp_path):
    script = tmp_path / "ag.py"; script.write_text("import sys, torch\nd = sys.argv[1]; sd = torch.load(d + '/pre.pt'); torch.save(sd, d + '/edit.pt')\n")
    m, hook, vl, applied = _train(tmp_path / "r", TH.cmd_agent([sys.executable, str(script)]), lambda a, b: {"accept": True}); assert applied == 1

def test_manual_revert_restores_pre_edit_weights(tmp_path):
    bump = lambda d, model: {k: v + 0.5 for k, v in model.state_dict().items()}
    m, hook, vl, applied = _train(tmp_path, bump, None, allow_unverified=True, steps=60)
    pre = torch.load(tmp_path / "intervention_01" / "pre.pt"); hook.revert(m); assert all(torch.equal(m.state_dict()[k], pre[k]) for k in pre); assert json.loads(open(tmp_path / "ledger.jsonl").read().strip().splitlines()[-1])["kind"] == "manual_revert"; _ledger_ok(tmp_path / "ledger.jsonl")
