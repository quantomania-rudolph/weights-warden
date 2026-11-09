import numpy as np
import env as E


def _single(lay, start, goal):
    lay = np.asarray(lay, float)[None]
    return dict(layouts=lay, start=np.array([start], float), goal=np.array([goal], float), layout_idx=np.zeros(1, int))


FAR = [[0.1, 0.9, 0.06], [0.9, 0.9, 0.06], [0.9, 0.1, 0.06], [0.1, 0.5, 0.06], [0.5, 0.9, 0.06]]


def test_golden_expert_success_train_and_val():
    S = E.splits()
    for name in ("TRAIN", "VAL_SEEN", "CANARY"):
        r = E.evaluate(None, S[name], 400, 123)
        assert r["success"] >= 0.95, (name, r["success"])
        assert r["success_ci"][0] >= 0.93


def test_golden_free_space_geometry():
    ep = _single(FAR, [0.3, 0.3], [0.7, 0.7])
    r = E.run_episodes(ep, None)
    assert r["status"][0] == 1
    assert abs(r["path_len"][0] - np.hypot(0.4, 0.4)) < 0.06   # straight line within one step


def test_golden_collision_ends_episode():
    lay = [[0.5, 0.5, 0.1]] + FAR[1:]
    ep = _single(lay, [0.2, 0.5], [0.8, 0.5])
    straight = lambda o: np.tile(np.array([1.0, 0.0]), (len(o), 1))
    r = E.run_episodes(ep, straight)
    assert r["status"][0] == 2          # walks into obstacle
    e = E.run_episodes(ep, None)
    assert e["status"][0] == 1          # expert detours


def test_golden_dynamics_and_timeout():
    ep = _single(FAR, [0.5, 0.5], [0.5, 0.0 + 0.99])
    zero = lambda o: np.zeros((len(o), 2))
    r = E.run_episodes(ep, zero)
    assert r["status"][0] == 0 and r["path_len"][0] == 0
    big = lambda o: np.full((len(o), 2), 50.0)          # clipped to 1 => 0.04 per axis
    r = E.run_episodes(_single(FAR, [0.3, 0.3], [0.9, 0.5]), big)
    assert r["path_len"][0] > 0 and r["path_len"][0] <= 0.04 * np.sqrt(2) * E.T + 1e-9


def test_golden_obs_velocity_and_chunk_replan():
    seen = []
    def pol(o):
        seen.append(o.copy())
        return np.tile(np.array([[[1.0, 0.0]] * 8]), (len(o), 1, 1))
    ep = _single(FAR, [0.2, 0.3], [0.2, 0.8])
    E.run_episodes(ep, pol)
    assert len(seen) == 15                              # replan every 4 steps over 60 steps
    assert np.allclose(seen[1][0, 2:4], [1.0, 0.0])     # last displacement / 0.04
    assert np.isclose(seen[1][0, 0], 0.2 + 4 * 0.04)


def test_golden_expert_chunk_shape():
    o = E.make_obs(np.array([[0.3, 0.3]]), np.zeros((1, 2)), np.array([[0.8, 0.8]]), np.array(FAR)[None])
    c = E.expert_chunk(o)
    assert c.shape == (1, 8, 2) and np.allclose(c[0, 0], E.expert_action(o)[0], atol=1e-6)
