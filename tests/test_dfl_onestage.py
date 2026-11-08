import numpy as np
import pyomo.environ as pyo
import pytest

from dfl_portfolio import solvers
from dfl_portfolio.models.dfl_grid import fit_dfl_grid
from dfl_portfolio.models.dfl_onestage import build_model, fit_dfl_onestage
from dfl_portfolio.models.ipo_closed_form import fit_ipo
from dfl_portfolio.models.linear import fit_ols
from dfl_portfolio.problem import decide, decision_cost

DELTA = 0.5


def count(m, ctype):
    return sum(1 for c in m.component_data_objects(ctype, active=True))


@pytest.mark.parametrize(
    ("formulation", "n_constraints"),
    [("kkt", 26 + 2 * 26 * 2), ("dual", 26 + 26 * 2 + 26)],
)
def test_model_size(small_data, formulation, n_constraints):
    m = build_model(*small_data, DELTA, formulation=formulation)
    assert count(m, pyo.Var) == 2 + 26 * 2 + 26 * 2 + 26  # theta, w, lambda, mu = 132
    assert count(m, pyo.Constraint) == n_constraints


@pytest.mark.parametrize("formulation", ["kkt", "dual"])
def test_true_lower_level_solution_is_feasible(small_data, formulation):
    """At any theta, (w*, lambda*, mu*) from the closed form satisfy every constraint."""
    X, Y, V = small_data
    theta = np.array([0.8, 0.3])
    m = build_model(X, Y, V, DELTA, formulation=formulation, theta_init=theta)
    W = decide(X * theta, V, DELTA)
    for t in m.T:
        grad = DELTA * V[t] @ W[t] - (1 - DELTA) * X[t] * theta
        support = W[t] > 1e-12
        mu = grad[support].mean()
        for j in m.J:
            m.w[t, j].value = W[t, j]
            m.lam[t, j].value = 0.0 if support[j] else grad[j] - mu
        m.mu[t].value = mu
    for con in m.component_data_objects(pyo.Constraint, active=True):
        assert abs(pyo.value(con.body) - pyo.value(con.upper)) < 1e-9
    assert all(v.value >= -1e-12 for v in m.lam.values())
    assert pyo.value(m.obj) == pytest.approx(decision_cost(X * theta, Y, V, DELTA), rel=1e-10)


def test_anchor_requires_target(small_data):
    with pytest.raises(ValueError):
        build_model(*small_data, DELTA, lambda_anchor=1.0)


def _solver_or_skip():
    for name in ("gurobi", "knitro"):
        if solvers.available(name):
            return name
    pytest.skip("neither Gurobi nor Knitro is available")


@pytest.mark.solver
@pytest.mark.filterwarnings("ignore:grid minimum lies on the boundary")
@pytest.mark.parametrize("formulation", ["kkt", "dual"])
def test_solver_matches_brute_force_reference(small_data, formulation):
    """Compare objective values, not thetas: the minimiser is not unique."""
    name = _solver_or_skip()
    X, Y, V = small_data
    res = fit_dfl_onestage(X, Y, V, DELTA, formulation=formulation, solver=name, time_limit=300)
    # the model's w must be the true lower-level decision for the returned theta
    assert res.objective == pytest.approx(decision_cost(X * res.theta, Y, V, DELTA), abs=1e-6)
    anchors = np.stack([fit_ols(X, Y), fit_ipo(X, Y, V, DELTA), res.theta])
    ref = fit_dfl_grid(X, Y, V, DELTA, anchors=anchors, n_grid=401)
    if name == "gurobi" and res.info["gap"] is not None and res.info["gap"] < 1e-4:
        assert res.objective <= ref.cost + 1e-6
    assert res.objective <= decision_cost(X * anchors[1], Y, V, DELTA) + 1e-6
