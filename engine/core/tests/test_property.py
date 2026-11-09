import numpy as np
import env as E
import policy as P
import torch


def test_property_paired_seeds_identical_episodes():
    S = E.splits()
    a = E.sample_episodes(S["VAL_SEEN"], 200, 5)
    b = E.sample_episodes(S["VAL_SEEN"], 200, 5)
    for k in ("layout_idx", "start", "goal"):
        assert np.array_equal(a[k], b[k])
    c = E.sample_episodes(S["VAL_SEEN"], 200, 6)
    assert not np.array_equal(a["start"], c["start"])
    # different policies see identical episodes: expert vs zero policy
    r1 = E.evaluate(None, S["TRAIN"], 100, 3); r2 = E.evaluate(lambda o: np.zeros((len(o), 2)), S["TRAIN"], 100, 3)
    assert np.array_equal(r1["layout_idx"], r2["layout_idx"])
    d = np.linalg.norm(a["goal"] - a["start"], axis=1)
    assert (d > 0.5).all()


def test_property_determinism():
    S = E.splits()
    torch.manual_seed(0)
    m = P.NavPolicy(dict(d=32, n_enc=1, n_dec=1))
    r1 = P.evaluate_model(m, S["VAL_SEEN"], 100, 11)
    r2 = P.evaluate_model(m, S["VAL_SEEN"], 100, 11)
    for k in ("success", "collision", "final_dist", "path_ratio", "success_ci"):
        assert r1[k] == r2[k]
    assert np.array_equal(r1["per_episode"]["final_dist"], r2["per_episode"]["final_dist"])
    e1 = E.evaluate(None, S["TRAIN"], 50, 1); e2 = E.evaluate(None, S["TRAIN"], 50, 1)
    assert e1["final_dist"] == e2["final_dist"]


def test_property_split_disjointness_and_sizes():
    S = E.splits(); K = E.split_keys()
    assert [len(S[k]) for k in ("TRAIN", "VAL_OOD", "VAL_SEEN", "CANARY", "HOLD_OOD")] == [40, 300, 150, 150, 600]
    assert not (K["TRAIN"] & K["VAL_OOD"]) and not (K["TRAIN"] & K["HOLD_OOD"]) and not (K["VAL_OOD"] & K["HOLD_OOD"])
    assert not (K["VAL_SEEN"] & K["CANARY"]) and (K["VAL_SEEN"] | K["CANARY"]) == K["VAL_OOD"]
    assert np.array_equal(S["ID_EVAL"], S["TRAIN"][:E.N_TRAINED]) and E.N_TRAINED == 12   # fixed 2026-11-04: ID_EVAL = the layouts the policies were actually trained on
    assert len(set().union(*[K[k] for k in ("TRAIN", "VAL_OOD", "HOLD_OOD")])) == 940
    for name in ("TRAIN", "VAL_OOD", "HOLD_OOD"):
        for l in S[name]:
            E.validate_layout(l)


def test_property_splits_are_pinned():
    import hashlib
    S = E.splits()
    h = hashlib.sha1(b"".join(S[k].tobytes() for k in ("TRAIN", "VAL_OOD", "HOLD_OOD"))).hexdigest()
    assert h == PINNED, h


def test_property_paired_bootstrap_sign():
    a = np.zeros(200); b = np.ones(200)
    r = E.paired_bootstrap(a, b)
    assert r["mean"] == 1 and r["lo"] > 0.9

PINNED = "84bacd306ddbd78b5dafd98f267d9926ce3409ce"
