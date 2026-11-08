import numpy as np

from dfl_portfolio.models.baselines import best_constant_portfolio, fit_ols_intercept
from dfl_portfolio.problem import mvo_cost


def test_ols_intercept_matches_polyfit(small_data):
    X, Y, _ = small_data
    beta, alpha = fit_ols_intercept(X, Y)
    for j in range(2):
        np.testing.assert_allclose([beta[j], alpha[j]], np.polyfit(X[:, j], Y[:, j], 1))


def test_best_constant_portfolio_beats_every_fixed_portfolio(small_data):
    _, Y, V = small_data
    w = best_constant_portfolio(Y, V, 0.5)
    assert w.min() >= 0 and np.isclose(w.sum(), 1.0)
    best = mvo_cost(np.broadcast_to(w, Y.shape), Y, V, 0.5).mean()
    for w1 in np.linspace(0, 1, 501):
        fixed = np.broadcast_to([w1, 1 - w1], Y.shape)
        assert best <= mvo_cost(fixed, Y, V, 0.5).mean() + 1e-12
