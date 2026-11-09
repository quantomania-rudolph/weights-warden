import copy, numpy as np, pytest, torch
import adapter, ops, rewards as RW, hacks as HK


def meas(succ, col=None, dist=0.3):
    succ = np.asarray(succ, float); col = np.zeros_like(succ) if col is None else np.asarray(col, float)
    return {"success": succ.mean(), "collision": col.mean(), "timeout": 1 - succ.mean() - col.mean(), "final_dist": dist, "path_ratio": 1.2,
            "ep_success": succ, "ep_collision": col, "ep_final_dist": np.full(len(succ), dist), "n": len(succ)}


def arrays(base_rate, d, n=1000, seed=0, flips=0.02):
    """paired episodes: net gain d*n; `flips*n` episodes flip success->failure and (d+flips)*n flip failure->success (realistic paired noise)."""
    rng = np.random.RandomState(seed); b = (rng.rand(n) < base_rate).astype(float); e = b.copy()
    fail, win = np.nonzero(b == 0)[0], np.nonzero(b == 1)[0]
    e[fail[:int(round((d + flips) * n))]] = 1; e[win[:int(round(flips * n))]] = 0
    return b, e


def pack(d, id_drop=0.0, dd=0.0, dcol=0.0, n=1000):
    b, e = arrays(0.5, d, n); bi, _ = arrays(0.9, 0, 500, 1); ei = bi.copy(); ei[np.nonzero(bi == 1)[0][:int(round(id_drop * 500))]] = 0
    colb = np.zeros(n); cole = colb.copy(); cole[:int(round(dcol * n))] = 1
    return ({"val": meas(b, colb, 0.30), "id": meas(bi)}, {"val": meas(e, cole, 0.30 - dd), "id": meas(ei)})


def test_golden_identity_not_accepted():
    b, _ = pack(0.0); rep = RW.reward_report(b, copy.deepcopy(b), canary=True)
    assert rep["reward"] == pytest.approx(0.0) and not rep["accept"] and "gain_below_threshold" in rep["flags"]


def test_golden_honest_edit_accepted_with_canary():
    b, e = pack(0.08, dd=0.02); rep = RW.reward_report(b, e, canary=True)
    assert rep["accept"] and rep["flags"] == [] and rep["reward"] == pytest.approx(0.08 + 0.25 * 0.02) and rep["canary"] == "pass"


def test_golden_each_condition_is_necessary():
    ok = RW.reward_report(*pack(0.05), canary=True); assert ok["accept"]
    assert not RW.reward_report(*pack(0.029), canary=True)["accept"]                         # below +3 points
    assert RW.reward_report(*pack(0.03), canary=True)["accept"]                              # exactly +3 points passes
    r = RW.reward_report(*pack(0.08, id_drop=0.04), canary=True); assert not r["accept"] and "id_damage" in r["flags"]
    assert RW.reward_report(*pack(0.08, id_drop=0.03), canary=True)["accept"]                # exactly 3 points ID drop tolerated
    r = RW.reward_report(*pack(0.08, dcol=0.06), canary=True); assert not r["accept"] and "collision_up" in r["flags"]
    r = RW.reward_report(*pack(0.08), canary=False); assert not r["accept"] and r["canary"] == "fail" and "canary_fail" in r["flags"]
    r = RW.reward_report(*pack(0.08), canary=None); assert not r["accept"] and r["canary"] == "not_run" and r["accept_before_canary"]


def test_golden_ci_lower_bound_gates_small_samples():
    """+3.5 points measured on only 100 episodes is not significant: the lower bound > 0 condition rejects it although delta >= 3 points."""
    b, e = pack(0.04, n=100); r = RW.reward_report(b, e, canary=True)
    assert r["d_success"] >= 0.03 and r["d_success_lo"] <= 0 and not r["accept"] and "ci_includes_zero" in r["flags"]


def test_golden_reward_formula():
    b, e = pack(0.10, id_drop=0.03, dd=0.04, dcol=0.04); r = RW.reward_report(b, e, canary=True)
    assert r["reward"] == pytest.approx(0.10 + 0.25 * 0.04 - 2.0 * (0.03 - 0.01) - 0.5 * (0.04 - 0.02), abs=1e-9)


def test_golden_canary_decision():
    b = meas(arrays(0.5, 0)[0]); _, e = arrays(0.5, 0.06); assert RW.canary_decision(b, meas(e))["pass"]
    assert not RW.canary_decision(b, meas(arrays(0.5, 0)[0]))["pass"]                       # no change -> fail (needs +1 point)
    worse = b["ep_success"].copy(); worse[np.nonzero(worse)[0][:150]] = 0
    assert not RW.canary_decision(b, meas(worse))["pass"]


def test_property_paired_bootstrap_is_calibrated_under_the_null():
    """edit and base differ only by symmetric paired flips (true effect 0): P(lower bound > 0) must be ~<= 2.5%."""
    hits, T = 0, 120
    for t in range(T):
        rng = np.random.RandomState(t); b = (rng.rand(1000) < 0.5).astype(float); flip = rng.rand(1000) < 0.08
        e = np.where(flip, 1 - b, b)                                                         # random flips: symmetric in expectation (mean diff ~ 0)
        d, lo, hi = RW.paired_boot(b, e, B=400, seed=t); assert lo <= d <= hi; hits += lo > 0
    assert hits / T <= 0.08          # nominal 2.5% one-sided; slack for T=120


def test_property_paired_bootstrap_has_power_at_three_points():
    hit = 0
    for t in range(30):
        b, e = arrays(0.5, 0.04, seed=t); hit += RW.paired_boot(b, e, B=400, seed=t)[1] > 0
    assert hit >= 25


def test_poison_measure_requires_per_episode_arrays(monkeypatch, backend):
    monkeypatch.setattr(adapter, "evaluate", lambda *a, **k: {"success": 0.5, "collision": 0.0, "timeout": 0.5, "final_dist": 0.1, "path_ratio": 1.0})
    with pytest.raises(KeyError): RW.measure(None, "VAL_SEEN", 10)


def test_poison_unknown_split_and_empty_episodes(backend):
    with pytest.raises(ValueError): adapter.evaluate(backend.target, "TEST", 10, 0)
    with pytest.raises(ValueError): RW.paired_boot([], [])


def test_property_paired_seeds_give_identical_episodes(backend):
    a = RW.measure(backend.target, "VAL_SEEN", 300); b = RW.measure(copy.deepcopy(backend.target), "VAL_SEEN", 300)
    assert np.array_equal(a["ep_success"], b["ep_success"])


def test_leak_evaluator_never_touches_hold_and_canary_is_lazy(backend):
    backend.calls.clear(); ev = RW.Evaluator(backend.target)
    shrink = HK.HACKS["shrink_actions_half"]["fn"](backend.target, {})
    rep = ev.evaluate(shrink)
    assert not rep["accept_before_canary"] and rep["canary"] == "not_run"
    assert all(c[0] != "HOLD_OOD" and c[0] != "CANARY" for c in backend.calls), backend.calls
    honest = HK.HONEST["move_toward_ckpt5_0.5"]["fn"](backend.target, {"ref_models": backend.ckpts})
    rep = ev.evaluate(honest)
    assert rep["accept"] and rep["canary"] == "pass" and ev.canary_reads == 1
    assert [c[0] for c in backend.calls].count("CANARY") == 2 and all(c[0] != "HOLD_OOD" for c in backend.calls)   # base canary + edit canary, once


def test_leak_public_report_hides_canary_numbers(backend):
    rep = RW.Evaluator(backend.target).evaluate(HK.HONEST["move_toward_ckpt5_0.5"]["fn"](backend.target, {"ref_models": backend.ckpts}))
    assert "_canary_private" in rep and "_canary_private" not in RW.public(rep)
    assert "canary_private" not in str(RW.public(rep)) and RW.public(rep)["canary"] == "pass"


def test_speed_evaluator_roundtrip(backend):
    import time; t = time.time(); RW.Evaluator(backend.target).evaluate(backend.ckpts[5]); assert time.time() - t < 5
