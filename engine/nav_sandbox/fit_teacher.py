"""Closed-form 'distill an inference-space behaviour into weights' (the AutoResearch-style edit family: build a TEACHER by changing inference, then ridge-fit chosen matrices to match it).
For a linear layer y = W x + b:  W_new = argmin ||X W^T + b - Y||^2 + lam ||W - W0||^2  =  (Y-b)^T X (X^T X + lam I)^-1  blended toward W0 (closed form).  No gradients, no training loop.
GUARD (the failure the post reports: head-only fit RMS 0.005 train -> 0.024 test): the fit is applied only if the HELD-OUT match error is not worse than `max_ratio` x the fit-set error; otherwise refuse."""
import torch


def ridge_to_teacher(W0, X, Y, bias=None, lam=1e-2):
    """W0 (out,in); X (N,in); Y (N,out) teacher outputs. returns W (out,in) minimising ||X W^T + b - Y||^2 + lam*N*||W-W0||^2 (lam scaled by N so it is dataset-size free)."""
    X, Y, W0 = X.double(), Y.double(), W0.double(); b = 0 if bias is None else bias.double()
    N, d = X.shape; A = X.T @ X + lam * N * torch.eye(d, dtype=X.dtype); B = X.T @ (Y - b) + lam * N * W0.T
    return torch.linalg.solve(A, B).T.to(torch.float32)


def fit_with_guard(W0, X_fit, Y_fit, X_held, Y_held, bias=None, lam=1e-2, max_ratio=1.5):
    W = ridge_to_teacher(W0, X_fit, Y_fit, bias, lam); b = 0 if bias is None else bias
    err = lambda Wm, X, Y: float(((X @ Wm.T + b) - Y).pow(2).mean().sqrt()); fit_e, held_e = err(W, X_fit, Y_fit), err(W, X_held, Y_held); base_held = err(W0, X_held, Y_held)
    ok = held_e <= max_ratio * max(fit_e, 1e-9) and held_e < base_held
    return {"W": W if ok else W0.clone(), "applied": bool(ok), "fit_rmse": fit_e, "held_rmse": held_e, "held_rmse_unedited": base_held, "ratio": held_e / max(fit_e, 1e-9)}
