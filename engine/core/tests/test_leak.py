import os, sys, tempfile
import numpy as np
import env as E
import data as D
import policy as P
import train as TR


def test_leak_all_sources_train_only():
    S = E.splits(); K = E.split_keys()
    h = D.collect_happy(k=3, eps_per_layout=3, seed=0)
    d = D.collect_dart(k=3, eps_per_layout=2, seed=0)
    m = P.NavPolicy(dict(d=32, n_enc=1, n_dec=1))
    g = D.collect_dagger(P.make_policy_fn(m), 0.5, k=3, eps_per_layout=3, seed=0, round_idx=0)
    ds = D.Dataset.concat([h, d, g])
    assert set(np.unique(ds.source)) == {0, 1, 2}
    D.check_dataset(ds)
    lay = ds.obs[:, 6:].reshape(-1, 5, 3).astype(np.float64)
    tr = S["TRAIN"]
    assert np.abs(lay - tr[ds.layout_id]).max() < 1e-5
    assert ds.layout_id.max() < 3
    for name in ("VAL_OOD", "HOLD_OOD"):
        for l in np.unique(lay, axis=0):
            assert np.abs(S[name] - l[None]).reshape(len(S[name]), -1).max(1).min() > 1e-3


def test_leak_labels_are_clean_expert_chunks():
    d = D.collect_dart(k=2, eps_per_layout=2, seed=1)       # noisy execution, clean labels
    assert np.allclose(d.act, E.expert_chunk(d.obs), atol=1e-6)


def test_leak_tiny_training_produces_checkpoints():
    with tempfile.TemporaryDirectory() as td:
        rec = dict(k=2, eps_per_layout=2, dagger_eps=2, d=32, n_enc=1, n_dec=1, steps0=22, steps_round=11, rounds=3, batch=32, eval_n=10)
        TR.train(0, td, rec, log=lambda *_: None)
        names = sorted(os.listdir(td))
        assert [n for n in names if n.startswith("ckpt_")] == [f"ckpt_{i:03d}.pt" for i in range(10)]
        assert "final.pt" in names
        m = P.load_policy(os.path.join(td, "final.pt"))
        assert m(__import__("torch").zeros(1, 21)).shape == (1, 8, 2)
