"""Decision-focused learning as a single nonconvex program (Pyomo).

The lower-level MVO is a convex QP, so its KKT conditions are necessary and sufficient and
can replace it inside the learning problem (the single-level reformulation used for
contextual decision-making by Munoz, Pineda & Morales, 2022). Two variants are provided:

* ``"kkt"``  - stationarity, which is linear in (theta, w, lambda, mu), plus the T * d
  complementarity products ``w * lambda = 0``; these are the only nonconvex constraints.
* ``"dual"`` - the same stationarity plus one aggregated equality per time step stating
  zero duality gap, ``delta w'Vw - (1 - delta) yhat'w = mu``. Given stationarity it is
  equivalent to ``lambda'w = 0``; it is quadratic and bilinear in (theta, w).

Both are nonconvex. ``build_model`` needs no solver; ``solve`` needs Gurobi or Knitro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyomo.environ as pyo

from dfl_portfolio.models.ipo_closed_form import fit_ipo
from dfl_portfolio.solvers import solve


def build_model(
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
    delta: float,
    *,
    formulation: str = "kkt",
    theta_init: np.ndarray | None = None,
    lambda_anchor: float = 0.0,
    theta_anchor: np.ndarray | None = None,
) -> pyo.ConcreteModel:
    """Pyomo model with variables theta (d), w and lambda (T x d) and mu (T)."""
    if formulation not in ("kkt", "dual"):
        raise ValueError("formulation must be 'kkt' or 'dual'")
    X, Y, V = (np.asarray(a, float) for a in (X, Y, V))
    T, d = X.shape
    if lambda_anchor > 0.0 and theta_anchor is None:
        raise ValueError("theta_anchor is required when lambda_anchor > 0")

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.J = pyo.RangeSet(0, d - 1)
    m.theta = pyo.Var(m.J, initialize=lambda m, j: 0.0 if theta_init is None else theta_init[j])
    m.w = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals, initialize=1.0 / d)
    m.lam = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals, initialize=0.0)
    m.mu = pyo.Var(m.T, initialize=0.0)

    def yhat(m, t, j):
        return float(X[t, j]) * m.theta[j]

    def risk_grad(m, t, j):
        return sum(float(V[t, j, k]) * m.w[t, k] for k in m.J)

    m.budget = pyo.Constraint(m.T, rule=lambda m, t: sum(m.w[t, j] for j in m.J) == 1.0)
    m.stationarity = pyo.Constraint(
        m.T,
        m.J,
        rule=lambda m, t, j: (
            delta * risk_grad(m, t, j) - (1.0 - delta) * yhat(m, t, j) - m.mu[t] - m.lam[t, j]
            == 0.0
        ),
    )
    if formulation == "kkt":
        m.complementarity = pyo.Constraint(
            m.T, m.J, rule=lambda m, t, j: m.w[t, j] * m.lam[t, j] == 0.0
        )
    else:
        m.zero_gap = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                delta * sum(m.w[t, j] * risk_grad(m, t, j) for j in m.J)
                - (1.0 - delta) * sum(yhat(m, t, j) * m.w[t, j] for j in m.J)
                == m.mu[t]
            ),
        )

    def objective(m):
        cost = (
            sum(
                -(1.0 - delta) * sum(float(Y[t, j]) * m.w[t, j] for j in m.J)
                + 0.5 * delta * sum(m.w[t, j] * risk_grad(m, t, j) for j in m.J)
                for t in m.T
            )
            / T
        )
        if lambda_anchor > 0.0:
            cost += (
                0.5 * lambda_anchor * sum((m.theta[j] - float(theta_anchor[j])) ** 2 for j in m.J)
            )
        return cost

    m.obj = pyo.Objective(rule=objective, sense=pyo.minimize)
    return m


@dataclass(frozen=True)
class OneStageResult:
    theta: np.ndarray
    W: np.ndarray
    objective: float
    info: dict = field(default_factory=dict)


def fit_dfl_onestage(
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
    delta: float,
    *,
    formulation: str = "kkt",
    theta_init: np.ndarray | None = None,
    solver: str = "gurobi",
    time_limit: float = 600.0,
    options: dict | None = None,
    **model_kwargs,
) -> OneStageResult:
    """Build and solve. theta starts from the IPO closed form unless ``theta_init`` is given."""
    if theta_init is None:
        theta_init = fit_ipo(X, Y, V, delta)
    m = build_model(X, Y, V, delta, formulation=formulation, theta_init=theta_init, **model_kwargs)
    info = solve(m, solver, time_limit=time_limit, options=options)
    theta = np.array([pyo.value(m.theta[j]) for j in m.J])
    W = np.array([[pyo.value(m.w[t, j]) for j in m.J] for t in m.T])
    return OneStageResult(theta, W, float(pyo.value(m.obj)), info)
