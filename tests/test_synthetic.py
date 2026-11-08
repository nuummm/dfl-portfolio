import numpy as np
import pytest

from dfl_portfolio.synthetic import SyntheticConfig, generate, misspecification

N = 200_000


@pytest.fixture(scope="module")
def large_sample():
    cfg = SyntheticConfig(misspec="nonlinear", rho=2.0)
    return cfg, generate(N, cfg, np.random.default_rng(7))


def test_features_follow_the_stationary_ar1(large_sample):
    cfg, (X, _, _) = large_sample
    np.testing.assert_allclose(X.mean(axis=0), cfg.mu_x, atol=0.01)
    np.testing.assert_allclose(X.var(axis=0), cfg.x_std**2, rtol=0.03)
    lag1 = [np.corrcoef(X[1:, j], X[:-1, j])[0, 1] for j in range(2)]
    np.testing.assert_allclose(lag1, cfg.phi, atol=0.01)


def test_noise_covariance_is_sigma(large_sample):
    cfg, (X, Y, V) = large_sample
    eps = Y - np.asarray(cfg.a) * X - misspecification(X, cfg)
    np.testing.assert_allclose(np.cov(eps.T), cfg.sigma, atol=0.005)
    np.testing.assert_array_equal(V[0], cfg.sigma)


def test_nonlinear_term_has_zero_mean(large_sample):
    """An intercept cannot absorb it, unlike the offset."""
    cfg, (X, _, _) = large_sample
    np.testing.assert_allclose(misspecification(X, cfg).mean(axis=0), 0.0, atol=0.01)


def test_offset_term_is_rho_times_b():
    cfg = SyntheticConfig(misspec="offset", rho=0.5)
    np.testing.assert_allclose(misspecification(np.zeros((3, 2)), cfg), [[0.5, -0.5]] * 3)


def test_same_seed_same_data():
    cfg = SyntheticConfig(misspec="nonlinear", rho=2.0)
    a = generate(50, cfg, np.random.default_rng(3))
    b = generate(50, cfg, np.random.default_rng(3))
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(u, v)


def test_rho_zero_is_the_same_for_both_misspecifications():
    a = generate(50, SyntheticConfig(misspec="offset"), np.random.default_rng(4))
    b = generate(50, SyntheticConfig(misspec="nonlinear"), np.random.default_rng(4))
    np.testing.assert_array_equal(a[1], b[1])


def test_rejects_unknown_misspecification():
    with pytest.raises(ValueError):
        SyntheticConfig(misspec="quadratic")
