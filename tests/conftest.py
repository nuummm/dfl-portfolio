import numpy as np
import pytest

from dfl_portfolio.synthetic import SyntheticConfig, generate


@pytest.fixture
def small_data():
    """26 training periods of the two-asset generator with a strong offset (rho = 1)."""
    return generate(26, SyntheticConfig(rho=1.0), np.random.default_rng(0))


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def random_spd(rng):
    def make(d):
        A = rng.standard_normal((d, d))
        return A @ A.T / d + 0.1 * np.eye(d)

    return make
