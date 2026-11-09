import json, os, pytest, torch
import adapter, loop, rewards as RW, ops


@pytest.fixture(autouse=True)
def fast_boot():
    old = RW.BOOT_B; RW.BOOT_B = 300; loop._MODELS.clear(); yield; RW.BOOT_B = old; loop._MODELS.clear()


H1 = ("revert_delta", {"frac": 0.5, "ref_ckpt": 5}); H2 = ("revert_delta", {"frac": 0.7, "ref_ckpt": 7})
BADOP = ("scale_token_embed", {"which": "goal", "gamma": 0.0})       # real op, harmful -> evaluated but rejected


def test_golden_chain_budget_and_charging(backend, tmp_path):
    d = str(tmp_path / "r"); loop.start(d, budget=6)
    e = loop.propose(d, *H1); assert e["payload"]["metrics"]["accepted"] and e["payload"]["index"] == 0
    for bad in [("nope", {}), ("scale_head", {"kind": "enc_self", "layer": 0, "head": 9, "gamma": 1.0}), ("revert_delta", {"frac": 0.5, "ref_ckpt": 5, "x": 1})]:
        with pytest.raises(loop.LoopError, match="charged"): loop.propose(d, *bad)
    with pytest.raises(loop.LoopError, match="duplicate"): loop.propose(d, *H1)
    r = loop.Run(d); assert len(r.of("iterate")) == 1 and len(r.strikes()) == 4 and r.left() == 1          # every rejected proposal is charged
    loop.propose(d, *H2)
    with pytest.raises(loop.LoopError, match="budget exhausted"): loop.propose(d, *BADOP)
    assert r.verify() == []


def test_poison_tampering_breaks_the_chain(backend, tmp_path):
    d = str(tmp_path / "r"); loop.start(d); loop.propose(d, *H1)
    r = loop.Run(d); assert r.verify() == []
    lines = open(r.log).read().split("\n"); e = json.loads(lines[1]); e["payload"]["metrics"]["d_success"] = 0.99; lines[1] = json.dumps(e, sort_keys=True)
    open(r.log, "w").write("\n".join(lines)); assert any("hash mismatch" in p for p in r.verify())
    d2 = str(tmp_path / "r2"); loop.start(d2); loop.propose(d2, *H1); loop.propose(d2, *H2); r2 = loop.Run(d2); head = r2.entries()[-1]["hash"]
    assert r2.verify(head) == [] and r2.verify("0" * 64) != []
    ls = open(r2.log).read().split("\n"); open(r2.log, "w").write("\n".join([ls[0]] + ls[2:])); assert r2.verify()


def test_golden_frozen_selection_rule(backend, tmp_path):
    d = str(tmp_path / "r"); loop.start(d); loop.propose(d, *H1); loop.propose(d, *BADOP); loop.propose(d, *H2)
    its = loop.Run(d).of("iterate"); assert [p["metrics"]["accepted"] for p in its] == [True, False, True]
    ch = loop.choose(d); best = max((p for p in its if p["metrics"]["accepted"]), key=lambda p: p["metrics"]["d_success"])
    assert ch["index"] == best["index"] and not ch["no_accept"]


def test_golden_no_accept_means_unedited_target(backend, tmp_path):
    d = str(tmp_path / "r"); loop.start(d); loop.propose(d, *BADOP)
    assert loop.choose(d)["no_accept"]
    s = loop.finish(d); assert s["no_accept"] and s["chosen"] is None and s["hold_delta_mean"] == 0.0 and not s["hold_gain_beats_noise"]


def test_golden_hold_read_once_with_three_seeds_and_only_in_score(backend, tmp_path):
    d = str(tmp_path / "r"); loop.start(d); backend.calls.clear()
    loop.propose(d, *H1); loop.propose(d, *H2); loop.propose(d, *BADOP)
    assert all(c[0] != "HOLD_OOD" for c in backend.calls)
    s = loop.finish(d)
    hold = [c for c in backend.calls if c[0] == "HOLD_OOD"]; assert len(hold) == 6 and len({c[2] for c in hold}) == 3        # edit+base x 3 episode seeds
    assert s["hold_delta_mean"] > 0.1 and s["hold_gain_beats_noise"] and len(s["per_seed"]) == 3 and s["id_edit_mean"] >= s["id_base_mean"]
    with pytest.raises(loop.LoopError, match="already scored"): loop.score(d)
    with pytest.raises(loop.LoopError, match="edit already written"): loop.propose(d, *H1)
    kinds = [e["kind"] for e in loop.Run(d).entries()]; assert kinds[-3:] == ["edit", "score", "close"] and loop.Run(d).verify() == []


def test_leak_analyst_sees_no_canary_numbers_and_log_has_none(backend, tmp_path):
    d = str(tmp_path / "r"); loop.start(d); e = loop.propose(d, *H1)
    v = loop.analyst_view(e); assert v["canary"] == "pass" and not any("canary" in k and k != "canary" for k in v)
    text = open(loop.Run(d).log).read(); assert "_canary_private" not in text and "canary_d" not in text and "HOLD" not in text
    assert loop.evaluator(d).canary_reads == 1
    loop.propose(d, *BADOP); assert loop.evaluator(d).canary_reads == 1          # canary only for proposals passing every other condition


def test_golden_listing_shows_every_argument_and_range(backend, tmp_path):
    d = str(tmp_path / "r"); loop.start(d); txt = loop.listing(d)
    for name, e in ops.REGISTRY.items():
        assert name in txt
        for k in e["schema"]: assert f"    {k}" in txt
    assert "ref_ckpt" in txt and "0..9" in txt and "enc_self" in txt and "dec_cross" in txt and "gamma" in txt


def test_speed_propose_latency_on_standin(backend, tmp_path):
    import time; d = str(tmp_path / "r"); loop.start(d); t = time.time(); loop.propose(d, *H1); loop.propose(d, *H2); assert time.time() - t < 8
