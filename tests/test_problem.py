import numpy as np
import pytest

from dfl_portfolio.problem import (
    decide,
    mvo_cost,
    project_simplex,
    regret,
    solve_mvo_2asset,
    solve_mvo_pgd,
)
from dfl_portfolio.synthetic import SyntheticConfig, generate


def test_project_simplex_is_feasible_and_idempotent(rng):
    for _ in range(50):
        w = project_simplex(3 * rng.standard_normal(5))
        assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(project_simplex(w), w, atol=1e-12)


@pytest.mark.parametrize("delta", [0.05, 0.3, 0.5, 0.9])
def test_two_asset_closed_form_matches_projected_gradient(rng, random_spd, delta):
    for _ in range(30):
        yhat, V = rng.standard_normal(2), random_spd(2)
        w = solve_mvo_2asset(yhat, V, delta)
        assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
        w_ref = solve_mvo_pgd(yhat, V, delta)
        expected = mvo_cost(w_ref, yhat, V, delta)
        assert mvo_cost(w, yhat, V, delta) == pytest.approx(expected, abs=1e-9)


def test_closed_form_is_never_beaten_by_a_feasible_point(rng, random_spd):
    yhat, V, delta = rng.standard_normal(2), random_spd(2), 0.4
    best = mvo_cost(solve_mvo_2asset(yhat, V, delta), yhat, V, delta)
    w1 = np.linspace(0, 1, 1001)
    costs = mvo_cost(np.column_stack([w1, 1 - w1]), yhat, V, delta)
    assert best <= costs.min() + 1e-12


def test_zero_curvature_gives_a_corner():
    """V11 + V22 = 2 V12 makes the cost linear in w1 even for delta > 0."""
    V = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(solve_mvo_2asset(np.array([0.3, 0.1]), V, 0.5), [1.0, 0.0])
    np.testing.assert_array_equal(solve_mvo_2asset(np.array([0.1, 0.3]), V, 0.5), [0.0, 1.0])


def test_delta_to_zero_picks_the_best_predicted_asset(rng, random_spd):
    for _ in range(20):
        yhat, V = rng.standard_normal(2), random_spd(2)
        np.testing.assert_array_equal(solve_mvo_2asset(yhat, V, 0.0), np.eye(2)[np.argmax(yhat)])
        np.testing.assert_allclose(
            solve_mvo_2asset(yhat, V, 1e-9), np.eye(2)[np.argmax(yhat)], atol=1e-6
        )


def test_general_d_decisions_are_feasible(rng, random_spd):
    yhat = rng.standard_normal((4, 3))
    V = np.stack([random_spd(3) for _ in range(4)])
    W = decide(yhat, V, 0.5)
    assert W.min() >= 0
    np.testing.assert_allclose(W.sum(axis=1), 1.0)


def test_perfect_foresight_has_zero_regret():
    X, Y, V = generate(50, SyntheticConfig(), np.random.default_rng(1))
    assert regret(Y, Y, V, 0.5) == 0.0
    assert regret(X, Y, V, 0.5) >= 0.0


def test_offset_along_ones_changes_no_regret():
    """Adding the same constant to every asset's return moves neither decisions nor regret."""
    base = generate(80, SyntheticConfig(rho=0.0), np.random.default_rng(2))
    shifted = generate(
        80, SyntheticConfig(misspec="offset", rho=0.7, b=(1.0, 1.0)), np.random.default_rng(2)
    )
    np.testing.assert_allclose(base[0], shifted[0])
    rng = np.random.default_rng(3)
    for theta in rng.uniform(-2, 3, size=(25, 2)):
        r0 = regret(base[0] * theta, base[1], base[2], 0.5)
        r1 = regret(shifted[0] * theta, shifted[1], shifted[2], 0.5)
        assert r1 == pytest.approx(r0, abs=1e-12)
