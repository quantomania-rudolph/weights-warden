import numpy as np, pytest
import env as E
import data as D

GOOD = np.array(E.splits()["TRAIN"][0])


@pytest.mark.parametrize("mut", ["nan", "radius", "centre", "overlap", "shape"])
def test_poison_bad_layouts(mut):
    l = GOOD.copy()
    if mut == "nan": l[0, 0] = np.nan
    if mut == "radius": l[1, 2] = 0.3
    if mut == "centre": l[2, 0] = 1.5
    if mut == "overlap": l[1, :2] = l[0, :2] + 0.01
    if mut == "shape": l = l[:4]
    with pytest.raises(E.LayoutError):
        E.validate_layout(l)
    with pytest.raises(E.LayoutError):
        E.evaluate(None, np.stack([GOOD, l]) if mut != "shape" else l[None], 10, 0)


def test_poison_dataset_with_val_or_hold_layout_raises():
    S = E.splits()
    for name in ("VAL_OOD", "HOLD_OOD", "CANARY"):
        ds = D.collect_happy(k=2, eps_per_layout=2, seed=0)
        lay = S[name][3].astype(np.float32)
        ds.obs[5:9, 6:] = lay.reshape(-1)
        with pytest.raises(D.LeakError):
            D.check_dataset(ds)
        with pytest.raises(D.LeakError):
            D.check_layouts(S[name][:2])


def test_poison_dataset_nan_or_bad_id_raises():
    ds = D.collect_happy(k=2, eps_per_layout=2, seed=0)
    ds.act[0, 0, 0] = np.nan
    with pytest.raises(D.LeakError):
        D.check_dataset(ds)
    ds = D.collect_happy(k=2, eps_per_layout=2, seed=0)
    ds.layout_id[:] = 1                                  # label claims layout 1 for layout-0 rows
    with pytest.raises(D.LeakError):
        D.check_dataset(ds)
