"""Closed-form IPO estimator (Butler & Kwon, 2023) for the linear predictor.

The lower-level problem is relaxed to the budget constraint only (w >= 0 is dropped). The
relaxed decision is affine in the prediction, so the decision cost is a convex quadratic in
theta and its minimiser solves a d x d linear system. The derivation is in docs/formulation.md.
"""

from __future__ import annotations

import numpy as np


def budget_affine_decision(V: np.ndarray, delta_eff: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, c) with argmin_{1'w=1} delta_eff/2 w'Vw - yhat'w = H @ yhat / delta_eff + c."""
    M = np.linalg.inv(V)
    a = M.sum(axis=1)
    H = (np.eye(V.shape[0]) - np.outer(a, np.ones(V.shape[0])) / a.sum()) @ M
    return H, a / a.sum()


def fit_ipo(
    X: np.ndarray, Y: np.ndarray, V: np.ndarray, delta: float, *, ridge: float = 1e-10
) -> np.ndarray:
    """Minimise the budget-relaxed decision cost over theta. Requires 0 < delta < 1.

    The common factor (1 - delta)^2 / delta multiplies both sides of the normal equations and
    the cross term H'Vc vanishes, so the minimiser itself does not depend on delta.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1) for the closed form, got {delta}")
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    V = np.asarray(V, float)
    d = X.shape[1]
    A = np.zeros((d, d))
    b = np.zeros(d)
    for x_t, y_t, V_t in zip(X, Y, V, strict=True):
        H, _ = budget_affine_decision(V_t, 1.0)
        D = np.diag(x_t)
        A += D @ H.T @ V_t @ H @ D
        b += D @ H.T @ y_t
    A = 0.5 * (A + A.T) + ridge * np.eye(d)
    return np.linalg.solve(A, b)
