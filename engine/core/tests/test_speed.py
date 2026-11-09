import time
import torch
import env as E
import policy as P


def test_speed_1000_episodes_under_30s():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    m = P.NavPolicy(dict(d=32, n_enc=1, n_dec=1))
    t = time.time()
    r = P.evaluate_model(m, E.splits()["VAL_SEEN"], 1000, 0)
    dt = time.time() - t
    assert dt < 30, dt
    assert r["n"] == 1000


def test_speed_default_policy_1000_episodes():
    m = P.NavPolicy()
    assert P.n_params(m) <= 6_000_000
    t = time.time(); P.evaluate_model(m, E.splits()["VAL_SEEN"], 1000, 0)
    assert time.time() - t < 30
