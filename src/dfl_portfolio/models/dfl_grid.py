"""Brute-force reference for decision-focused learning with two assets.

For d = 2 the decision cost L(theta) can be evaluated exactly at any theta (the lower-level
MVO has a closed form), so scanning a grid gives a reference value that does not depend on a
nonconvex solver. It is a reference, not an exact optimum: the answer is only as fine as the
grid. L is piecewise smooth and flat wherever every decision is a corner portfolio, so the
minimiser is generally not unique; compare objective values, not thetas.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from dfl_portfolio.problem import mvo_cost, solve_mvo_2asset


@dataclass(frozen=True)
class GridResult:
    theta: np.ndarray
    cost: float
    theta1: np.ndarray
    theta2: np.ndarray
    costs: np.ndarray  # costs[i, j] = L(theta1[i], theta2[j])
    on_boundary: bool


def decision_cost_at(
    thetas: np.ndarray, X: np.ndarray, Y: np.ndarray, V: np.ndarray, delta: float
) -> np.ndarray:
    """L(theta) for a batch of thetas ``(K, 2)``; returns ``(K,)``."""
    yhat = thetas[:, None, :] * X[None, :, :]
    W = solve_mvo_2asset(yhat, V[None], delta)
    return mvo_cost(W, Y[None], V[None], delta).mean(axis=1)


def grid_bounds(anchors: np.ndarray, scale: float = 3.0, min_width: float = 1.0) -> np.ndarray:
    """Box centred on the anchors' bounding box, ``scale`` times as wide. Returns ``(2, 2)``."""
    lo, hi = anchors.min(axis=0), anchors.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * scale * np.maximum(hi - lo, min_width)
    return np.stack([centre - half, centre + half], axis=1)


def fit_dfl_grid(
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
    delta: float,
    *,
    anchors: np.ndarray,
    n_grid: int = 201,
    scale: float = 3.0,
) -> GridResult:
    """Minimise L over an ``n_grid`` x ``n_grid`` grid plus the anchor points themselves.

    Including the anchors (typically the OLS and IPO estimates) guarantees that the
    reference is never worse than either of them.
    """
    X, Y, V = (np.asarray(a, float) for a in (X, Y, V))
    anchors = np.atleast_2d(np.asarray(anchors, float))
    if X.shape[1] != 2:
        raise ValueError("dfl_grid only supports two assets")
    bounds = grid_bounds(anchors, scale)
    theta1 = np.linspace(*bounds[0], n_grid)
    theta2 = np.linspace(*bounds[1], n_grid)
    costs = np.empty((n_grid, n_grid))
    for i, t1 in enumerate(theta1):
        row = np.column_stack([np.full(n_grid, t1), theta2])
        costs[i] = decision_cost_at(row, X, Y, V, delta)

    i, j = np.unravel_index(np.argmin(costs), costs.shape)
    best_theta, best_cost = np.array([theta1[i], theta2[j]]), float(costs[i, j])
    on_boundary = bool(best_cost < costs[1:-1, 1:-1].min() - 1e-12)
    anchor_costs = decision_cost_at(anchors, X, Y, V, delta)
    k = int(np.argmin(anchor_costs))
    if anchor_costs[k] < best_cost:
        best_theta, best_cost, on_boundary = anchors[k].copy(), float(anchor_costs[k]), False
    if on_boundary:
        warnings.warn(
            "grid minimum lies on the boundary; widen the grid", RuntimeWarning, stacklevel=2
        )
    return GridResult(best_theta, best_cost, theta1, theta2, costs, on_boundary)
