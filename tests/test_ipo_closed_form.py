import numpy as np
import pytest
from scipy.optimize import minimize

from dfl_portfolio.models.ipo_closed_form import budget_affine_decision, fit_ipo
from dfl_portfolio.problem import mvo_cost


def relaxed_cost(theta, X, Y, V, delta):
    """Decision cost when the lower level keeps only the budget constraint."""
    delta_eff = delta / (1 - delta)
    total = 0.0
    for x, y, v in zip(X, Y, V, strict=True):
        H, c = budget_affine_decision(v, delta_eff)
        w = H @ (x * theta) / delta_eff + c
        total += mvo_cost(w, y, v, delta)
    return total / len(X)


def test_affine_decision_is_the_budget_constrained_optimum(rng):
    V = np.array([[1.0, 0.2, 0.1], [0.2, 0.8, 0.0], [0.1, 0.0, 1.5]])
    yhat, delta_eff = rng.standard_normal(3), 2.0
    H, c = budget_affine_decision(V, delta_eff)
    w = H @ yhat / delta_eff + c
    assert w.sum() == pytest.approx(1.0)
    # stationarity of the Lagrangian: delta_eff V w - yhat is parallel to the ones vector
    g = delta_eff * V @ w - yhat
    np.testing.assert_allclose(g, g.mean(), atol=1e-10)


@pytest.mark.parametrize("delta", [0.2, 0.5, 0.8])
def test_closed_form_matches_numerical_minimiser(small_data, delta):
    X, Y, V = small_data
    theta = fit_ipo(X, Y, V, delta)
    res = minimize(
        relaxed_cost, x0=np.zeros(2), args=(X, Y, V, delta), method="BFGS", options={"gtol": 1e-10}
    )
    assert relaxed_cost(theta, X, Y, V, delta) <= res.fun + 1e-10
    np.testing.assert_allclose(theta, res.x, atol=1e-4)


def test_minimiser_does_not_depend_on_delta(small_data):
    X, Y, V = small_data
    np.testing.assert_allclose(fit_ipo(X, Y, V, 0.2), fit_ipo(X, Y, V, 0.8), rtol=1e-8)


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1])
def test_rejects_delta_outside_open_interval(small_data, delta):
    with pytest.raises(ValueError):
        fit_ipo(*small_data, delta)
