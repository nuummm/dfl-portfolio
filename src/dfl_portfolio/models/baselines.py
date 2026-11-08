"""Baselines outside the shared no-intercept class, used to judge whether a gain is real.

* OLS with a per-asset intercept: a larger model class than the one the other methods use.
* The best constant portfolio: ignores the features altogether.
"""

from __future__ import annotations

import numpy as np

from dfl_portfolio.problem import decide


def fit_ols_intercept(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Asset-wise OLS ``y_tj ~ alpha_j + beta_j x_tj``; returns ``(beta, alpha)``."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    xc = X - X.mean(axis=0)
    beta = (xc * (Y - Y.mean(axis=0))).sum(axis=0) / (xc**2).sum(axis=0)
    return beta, Y.mean(axis=0) - beta * X.mean(axis=0)


def best_constant_portfolio(Y: np.ndarray, V: np.ndarray, delta: float) -> np.ndarray:
    """The fixed portfolio with the lowest average in-sample cost.

    The cost is linear in y and V, so its average over t is the cost at the averages, and the
    minimiser is the MVO decision for ``(mean Y, mean V)``.
    """
    Y = np.asarray(Y, float)
    V = np.asarray(V, float)
    return decide(Y.mean(axis=0)[None], V.mean(axis=0)[None], delta)[0]
