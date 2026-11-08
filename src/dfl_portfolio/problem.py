"""Mean-variance decision problem shared by every model.

Conventions: ``X``, ``Y`` and predictions are ``(T, d)``; covariances ``V`` are ``(T, d, d)``.
The decision at time t is ``w_t = argmin_{w in simplex} cost(w; yhat_t, V_t)`` with

    cost(w; y, V) = -(1 - delta) * w @ y + delta / 2 * w @ V @ w,   0 <= delta <= 1.
"""

from __future__ import annotations

import numpy as np


def mvo_cost(w: np.ndarray, y: np.ndarray, V: np.ndarray, delta: float) -> np.ndarray:
    """Mean-variance cost, vectorised over any leading dimensions."""
    w = np.asarray(w, float)
    ret = np.einsum("...i,...i->...", w, y)
    risk = np.einsum("...i,...ij,...j->...", w, V, w)
    return -(1.0 - delta) * ret + 0.5 * delta * risk


def solve_mvo_2asset(yhat: np.ndarray, V: np.ndarray, delta: float) -> np.ndarray:
    """Exact long-only MVO for d = 2, vectorised over leading dimensions.

    With w = (w1, 1 - w1) the cost is a convex quadratic in w1, so the minimiser is the
    stationary point clipped to [0, 1]. When the curvature is zero (delta = 0 or a singular
    V with V11 + V22 = 2 V12) the cost is linear and the solution is a corner.
    """
    yhat = np.asarray(yhat, float)
    V = np.asarray(V, float)
    num = (1.0 - delta) * (yhat[..., 0] - yhat[..., 1]) + delta * (V[..., 1, 1] - V[..., 0, 1])
    den = delta * (V[..., 0, 0] + V[..., 1, 1] - 2.0 * V[..., 0, 1])
    flat = den <= 1e-14
    w1 = np.where(flat, (num >= 0).astype(float), np.clip(num / np.where(flat, 1.0, den), 0, 1))
    return np.stack([w1, 1.0 - w1], axis=-1)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1} (sort-based algorithm)."""
    v = np.asarray(v, float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, v.size + 1)
    r = k[u - css / k > 0][-1]
    return np.maximum(v - css[r - 1] / r, 0.0)


def solve_mvo_pgd(
    yhat: np.ndarray, V: np.ndarray, delta: float, *, n_iter: int = 20_000, tol: float = 1e-12
) -> np.ndarray:
    """Long-only MVO for any d by projected gradient. Slow; used as a test oracle."""
    yhat = np.asarray(yhat, float)
    V = np.asarray(V, float)
    d = yhat.size
    if delta == 0.0:
        w = np.zeros(d)
        w[int(np.argmax(yhat))] = 1.0
        return w
    step = 1.0 / (delta * np.linalg.eigvalsh(V)[-1])
    w = np.full(d, 1.0 / d)
    for _ in range(n_iter):
        grad = -(1.0 - delta) * yhat + delta * V @ w
        w_new = project_simplex(w - step * grad)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


def decide(yhat: np.ndarray, V: np.ndarray, delta: float) -> np.ndarray:
    """Optimal long-only portfolios for a batch of predictions ``(T, d)``."""
    yhat = np.asarray(yhat, float)
    if yhat.shape[-1] == 2:
        return solve_mvo_2asset(yhat, V, delta)
    return np.stack([solve_mvo_pgd(y, v, delta) for y, v in zip(yhat, V, strict=True)])


def decision_cost(yhat: np.ndarray, Y: np.ndarray, V: np.ndarray, delta: float) -> float:
    """Average realised cost of the portfolios chosen from ``yhat``."""
    W = decide(yhat, V, delta)
    return float(np.mean(mvo_cost(W, Y, V, delta)))


def regret(yhat: np.ndarray, Y: np.ndarray, V: np.ndarray, delta: float) -> float:
    """Decision cost minus the cost of the decisions made with perfect foresight of ``Y``."""
    return decision_cost(yhat, Y, V, delta) - decision_cost(Y, Y, V, delta)


def mse(yhat: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean((np.asarray(yhat) - np.asarray(Y)) ** 2))
