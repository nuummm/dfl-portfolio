"""Per-asset linear predictor ``yhat_t = x_t * theta`` and its least-squares fit.

There is no intercept on purpose: every method then learns the same d parameters, and for
d = 2 the whole hypothesis class can be drawn as the theta plane.
"""

from __future__ import annotations

import numpy as np


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.asarray(X, float) * np.asarray(theta, float)


def fit_ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Asset-wise OLS without intercept: theta_j = sum_t x_tj y_tj / sum_t x_tj^2."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    return (X * Y).sum(axis=0) / (X**2).sum(axis=0)
