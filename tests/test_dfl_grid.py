import numpy as np
import pytest

from dfl_portfolio.models.dfl_grid import decision_cost_at, fit_dfl_grid
from dfl_portfolio.models.ipo_closed_form import fit_ipo
from dfl_portfolio.models.linear import fit_ols
from dfl_portfolio.problem import decision_cost

DELTA = 0.5


def test_batched_cost_matches_problem_module(small_data, rng):
    X, Y, V = small_data
    thetas = rng.uniform(-1, 3, size=(10, 2))
    expected = [decision_cost(X * t, Y, V, DELTA) for t in thetas]
    np.testing.assert_allclose(decision_cost_at(thetas, X, Y, V, DELTA), expected, rtol=1e-12)


def test_reference_is_no_worse_than_ols_and_ipo(small_data):
    X, Y, V = small_data
    anchors = np.stack([fit_ols(X, Y), fit_ipo(X, Y, V, DELTA)])
    res = fit_dfl_grid(X, Y, V, DELTA, anchors=anchors, n_grid=81)
    for theta in anchors:
        assert res.cost <= decision_cost(X * theta, Y, V, DELTA) + 1e-12
    assert res.cost == pytest.approx(decision_cost(X * res.theta, Y, V, DELTA), rel=1e-12)


def test_grid_contains_the_anchor_box(small_data):
    X, Y, V = small_data
    anchors = np.array([[0.5, -0.2], [1.5, 0.4]])
    res = fit_dfl_grid(X, Y, V, DELTA, anchors=anchors, n_grid=11)
    assert res.theta1[0] < 0.5 and res.theta1[-1] > 1.5
    assert res.theta2[0] < -0.2 and res.theta2[-1] > 0.4
    assert res.costs.shape == (11, 11)


def test_rejects_more_than_two_assets():
    X = np.ones((5, 3))
    with pytest.raises(ValueError):
        fit_dfl_grid(X, X, np.broadcast_to(np.eye(3), (5, 3, 3)), DELTA, anchors=np.ones((1, 3)))
