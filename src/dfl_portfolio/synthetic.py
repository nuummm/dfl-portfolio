"""Synthetic returns with a controllable misspecification.

    x_t = mu_x + phi (x_{t-1} - mu_x) + eta_t           (AR(1) feature, one per asset)
    y_t = a * x_t + rho * b * g(x_t) + eps_t,  eps_t ~ N(0, Sigma)

The predictor ``yhat = x * theta`` cannot represent ``rho * b * g(x)``. Two choices of g:

* ``"offset"``: g = 1. A constant tilt between the assets; adding an intercept removes it.
* ``"nonlinear"``: g(x) = |x - mu_x| - E|x - mu_x|, a V-shaped term with zero mean under the
  stationary distribution of x, so an intercept cannot absorb it.

``b = (+1, -1)`` makes the term change the ranking of the assets. A term proportional to the
ones vector would change nothing: under the budget constraint it adds the same amount to the
cost of every portfolio, so it moves neither the decisions nor the regret.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MISSPECS = ("offset", "nonlinear")


@dataclass(frozen=True)
class SyntheticConfig:
    a: tuple[float, ...] = (1.0, 1.0)
    b: tuple[float, ...] = (1.0, -1.0)
    rho: float = 0.0
    misspec: str = "nonlinear"
    mu_x: tuple[float, ...] = (1.0, 1.0)
    phi: float = 0.7
    x_std: float = 0.5
    noise_std: tuple[float, ...] = (0.5, 0.5)
    noise_corr: float = 0.3

    def __post_init__(self) -> None:
        if self.misspec not in MISSPECS:
            raise ValueError(f"misspec must be one of {MISSPECS}, got {self.misspec!r}")

    @property
    def sigma(self) -> np.ndarray:
        s = np.asarray(self.noise_std, float)
        corr = np.full((s.size, s.size), self.noise_corr)
        np.fill_diagonal(corr, 1.0)
        return corr * np.outer(s, s)


def misspecification(X: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    """The part of E[y | x] outside the model class, ``rho * b * g(x)``."""
    X = np.asarray(X, float)
    if cfg.misspec == "offset":
        g = np.ones_like(X)
    else:
        g = np.abs(X - np.asarray(cfg.mu_x)) - cfg.x_std * np.sqrt(2.0 / np.pi)
    return cfg.rho * np.asarray(cfg.b) * g


def generate(
    n: int, cfg: SyntheticConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X, Y`` of shape ``(n, d)`` and the true covariance ``V`` repeated ``(n, d, d)``."""
    mu_x = np.asarray(cfg.mu_x, float)
    d = mu_x.size
    innov_std = cfg.x_std * np.sqrt(1.0 - cfg.phi**2)
    X = np.empty((n, d))
    x = mu_x + cfg.x_std * rng.standard_normal(d)
    for t in range(n):
        x = mu_x + cfg.phi * (x - mu_x) + innov_std * rng.standard_normal(d)
        X[t] = x
    Sigma = cfg.sigma
    eps = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    Y = np.asarray(cfg.a) * X + misspecification(X, cfg) + eps
    return X, Y, np.broadcast_to(Sigma, (n, d, d)).copy()
