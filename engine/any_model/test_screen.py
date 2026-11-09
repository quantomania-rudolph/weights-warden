import os, sys, json, hashlib, pytest, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import adapter, loop as LP, ops as OPS, lm_backend as B, llm_ops, more_ops
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(B.default_root(), "policy_s0", "ckpt_009.pt")), reason="run any_model/lm_backend.py train first")


@pytest.fixture()
def lm():
    saved = (dict(OPS.REGISTRY), dict(OPS.RANGE_HOOKS), adapter._BACKEND if hasattr(adapter, "_BACKEND") else None)
    adapter.set_backend(B.LMBackend()); llm_ops.enable(); more_ops.enable_value_ops(); more_ops.enable_more_value_ops(); LP._MODELS.clear()
    yield
    llm_ops.disable(); OPS.REGISTRY.clear(); OPS.REGISTRY.update(saved[0]); OPS.RANGE_HOOKS.clear(); OPS.RANGE_HOOKS.update(saved[1]); LP._MODELS.clear()


GOOD = {"ops": [{"op": "scale_head", "cfg": {"layer": 0, "head": h, "gamma": 4.0}} for h in (0, 1)]}


def test_screen_never_uses_the_judges_data_and_is_free(lm, tmp_path):
    assert LP.SCREEN_SPLIT not in ("VAL_SEEN", "ID_EVAL", "CANARY", "HOLD_OOD", "VAL_OOD")
    run = str(tmp_path / "r"); LP.start(run, 4); r = LP.screen(run, "composite", GOOD)
    assert r["d_success"] > 0.1 and "research" in r["split"]
    s = LP.Run(run); assert len(s.of("iterate")) == 0 and s.left() == 4 and s.verify() == []            # the charged ledger is untouched
    assert LP.Run(run).entries()[-1]["kind"] == "start"


def test_screens_are_logged_in_their_own_tamper_evident_chain(lm, tmp_path):
    run = str(tmp_path / "r"); LP.start(run, 4); LP.screen(run, "scale_head", {"layer": 0, "head": 1, "gamma": 2.0}); LP.screen(run, "scale_head", {"layer": 0, "head": 1, "gamma": 3.0})
    sr = LP._screen_log(run); assert len(sr.entries()) == 2 and sr.verify() == []
    lines = open(sr.log).read().splitlines(); lines[0] = lines[0].replace("2.0", "9.0"); open(sr.log, "w").write("\n".join(lines) + "\n"); assert sr.verify() != []     # editing history is detected


def test_cap_is_enforced_atomically_and_invalid_attempts_count(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(LP, "SCREEN_MAX", 4); run = str(tmp_path / "r"); LP.start(run, 4)
    bad = LP.screen(run, "scale_head", {"layer": 9, "head": 0, "gamma": 2.0}); assert "error" in bad and bad["screens_left"] == 3            # invalid tries are not free probes
    with pytest.raises(LP.LoopError): LP.sweep(run, "scale_head", {"layer": 0, "head": 1, "gamma": 1.0}, "gamma", [1, 2, 3, 4])            # needs 4, only 3 left -> nothing runs
    assert len(LP._screen_log(run).entries()) == 1
    LP.sweep(run, "scale_head", {"layer": 0, "head": 1, "gamma": 1.0}, "gamma", [1, 2, 3]); assert len(LP._screen_log(run).entries()) == 4
    with pytest.raises(LP.LoopError): LP.screen(run, "scale_head", {"layer": 0, "head": 1, "gamma": 2.0})


def test_sweep_curve_dotted_path_and_bad_parameters(lm, tmp_path):
    run = str(tmp_path / "r"); LP.start(run, 4); r = LP.sweep(run, "composite", GOOD, "ops.0.cfg.gamma", [1.0, 4.0]); assert [c["value"] for c in r["curve"]] == [1.0, 4.0] and "d_success" in r["curve"][0]
    assert r["curve"][0]["d_success"] < r["curve"][1]["d_success"]                          # gamma 1 on head 0 + 4 on head 1  <  4 on both
    e = LP.sweep(run, "scale_head", {"layer": 0, "head": 1, "gamma": 1.0}, "nonexistent", [1, 2]); assert all("error" in c for c in e["curve"]) and e["best_value"] is None
    with pytest.raises(LP.LoopError): LP.sweep(run, "scale_head", {"layer": 0, "head": 1, "gamma": 1.0}, "gamma", [1.0])
    with pytest.raises(LP.LoopError): LP.sweep(run, "scale_head", {"layer": 0, "head": 1, "gamma": 1.0}, "gamma", list(range(10)))


def test_screen_agrees_in_sign_with_the_charged_judge_and_target_is_unchanged(lm, tmp_path):
    run = str(tmp_path / "r"); LP.start(run, 4); cfg = LP.Run(run).config(); before = hashlib.sha256(open(adapter.backend().ckpt_path(0, "final"), "rb").read()).hexdigest()
    free = LP.screen(run, "composite", GOOD)["d_success"]; paid = LP.propose(run, "composite", GOOD)["payload"]["metrics"]["d_success"]
    assert free > 0 and paid > 0                                                           # a good edit looks good on both
    free_bad = LP.screen(run, "scale_head", {"layer": 0, "head": 1, "gamma": 0.0})["d_success"]; paid_bad = LP.propose(run, "scale_head", {"layer": 0, "head": 1, "gamma": 0.0})["payload"]["metrics"]["d_success"]
    assert free_bad < 0 and paid_bad < 0                                                   # and a bad edit looks bad on both
    assert hashlib.sha256(open(adapter.backend().ckpt_path(0, "final"), "rb").read()).hexdigest() == before
