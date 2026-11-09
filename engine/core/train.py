"""Train the ACT-like Nav2D policy: happy+DART phase, then 3 DAgger rounds. 10 checkpoints + final.
usage: python core/train.py --seed S --out models/nav/policy_sS
Checkpoints ckpt_000..ckpt_009 at global steps round(total*(i+1)/11), plus final.pt (step=total).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env as E
import data as D
import policy as P

RECIPE = dict(k=12, eps_per_layout=40, dagger_eps=30, d=64, n_heads=4, n_enc=2, n_dec=2, steps0=2000,
              steps_round=1500, rounds=3, batch=256, lr=3e-3, wd=0.01, eval_n=200, vel_shuffle=1.0)
BETAS = (0.5, 0.25, 0.0)


def train(seed, out, recipe=None, log=print, do_eval=True):
    R = dict(RECIPE); R.update(recipe or {})
    torch.set_num_threads(1)
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out, exist_ok=True)
    cfg = dict(d=R["d"], n_heads=R["n_heads"], n_enc=R["n_enc"], n_dec=R["n_dec"])
    model = P.NavPolicy(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=R["lr"], weight_decay=R["wd"])
    total = R["steps0"] + R["rounds"] * R["steps_round"]
    ck_steps = {int(round(total * (i + 1) / 11)): i for i in range(10)}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 1 - s / total))
    t0 = time.time()
    ds = D.Dataset.concat([D.collect_happy(R["k"], R["eps_per_layout"], seed),
                           D.collect_dart(R["k"], R["eps_per_layout"], seed)])
    log(f"[data] happy+dart {ds.sizes()} N={len(ds)} t={time.time()-t0:.0f}s")
    gen = torch.Generator().manual_seed(seed)
    S = E.splits()
    ids = S["TRAIN"][:R["k"]]
    hist, step = [], 0

    def save(path):
        torch.save(dict(state_dict=model.state_dict(), cfg=cfg, step=step, recipe=R, seed=seed), path)

    def run_phase(n_steps, tag):
        nonlocal step
        X, Y = torch.from_numpy(ds.obs), torch.from_numpy(ds.act)
        model.train()
        for _ in range(n_steps):
            idx = torch.randint(0, len(X), (R["batch"],), generator=gen)
            xb = X[idx].clone()
            if R["vel_shuffle"] > 0:    # anti-copycat: velocity ~ previous expert action, shuffle it across the batch
                sel = torch.rand(len(xb), generator=gen) < R["vel_shuffle"]
                perm = torch.randperm(len(xb), generator=gen)
                xb[sel, 2:4] = xb[perm][sel, 2:4]
            loss = (model(xb) - Y[idx]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if step in ck_steps:
                i = ck_steps[step]; save(os.path.join(out, f"ckpt_{i:03d}.pt"))
                ent = dict(ckpt=i, step=step, phase=tag, loss=float(loss.detach()))
                if do_eval:
                    model.eval()
                    ent["id"] = P.evaluate_model(model, ids, R["eval_n"], 9001)["success"]
                    ent["val_seen"] = P.evaluate_model(model, S["VAL_SEEN"], R["eval_n"], 9002)["success"]
                    model.train()
                hist.append(ent); log(f"[ckpt] {ent} t={time.time()-t0:.0f}s")
        model.eval()

    run_phase(R["steps0"], "happy+dart")
    for r in range(R["rounds"]):
        fn = P.make_policy_fn(model)
        new = D.collect_dagger(fn, BETAS[r], R["k"], R["dagger_eps"], seed, r)
        ds = D.Dataset.concat([ds, new]); D.check_dataset(ds)
        log(f"[dagger {r}] beta={BETAS[r]} +{len(new)} -> {ds.sizes()} t={time.time()-t0:.0f}s")
        run_phase(R["steps_round"], f"dagger{r}")
    save(os.path.join(out, "final.pt"))
    info = dict(seed=seed, recipe=R, params=P.n_params(model), dataset=ds.sizes(), n_total=len(ds),
                steps=total, history=hist, train_seconds=time.time() - t0)
    if do_eval:
        info["final_id"] = {k: v for k, v in P.evaluate_model(model, ids, 1000, 9101).items() if k in ("success", "success_ci", "collision", "timeout", "final_dist")}
        info["final_val_seen"] = {k: v for k, v in P.evaluate_model(model, S["VAL_SEEN"], 1000, 9102).items() if k in ("success", "success_ci", "collision", "timeout", "final_dist")}
    json.dump(info, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    log(f"[done] {json.dumps({k: info.get(k) for k in ('final_id','final_val_seen','train_seconds')})}")
    return model, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    for k, v in RECIPE.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    rec = {k: getattr(a, k) for k in RECIPE}
    train(a.seed, a.out, rec)
