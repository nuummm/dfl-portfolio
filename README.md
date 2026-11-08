# dfl-portfolio

[![CI](https://github.com/nuummm/dfl-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/nuummm/dfl-portfolio/actions/workflows/ci.yml)

**What**: decision-focused learning (DFL) for long-only mean-variance portfolios: train a return predictor on the cost of the portfolio it induces instead of on squared error.
**How**: the lower-level portfolio QP is replaced by its KKT conditions (Pyomo), and every result is checked against closed forms and a brute-force reference that need no solver.
**Result**: on a misspecified two-asset synthetic problem, the decision-optimal predictor has lower test regret than OLS in 18 of 20 seeds while its test MSE is about 6x higher; on a well-specified one it is worse than OLS.

This is a cleaned, partial re-implementation of the code from my undergraduate thesis. Only the synthetic part is public. The thesis also compared SPO+ and a gradient-based IPO variant and evaluated the methods on real market data; none of that is included here.

![Decision-regret landscape](docs/figures/landscape.png)

*In-sample regret over the parameter plane (log scale, darker is worse) with training-MSE contours. With the offset (right), the regret-minimising theta is far from the MSE minimum. Without it (left), OLS sits in the regret valley and the grid minimum drifts to the box edge, a sign of overfitting rather than a better rule.*

## Problem setup

At each period t, a linear model predicts asset returns from features, `yhat_t = x_t * theta` (element-wise, no intercept), and the portfolio is chosen by

```
w*(yhat, V) = argmin_{w >= 0, sum(w) = 1}  -(1 - delta) w'yhat + delta/2 w'Vw .
```

Training objectives:

| Objective | Definition |
|---|---|
| MSE | mean of `(yhat_t - y_t)^2` |
| Decision cost `L(theta)` | mean realised cost `-(1 - delta) w*'y_t + delta/2 w*'V_t w*` |
| Regret | `L(theta)` minus the cost of perfect-foresight portfolios `w*(y_t, V_t)` |

Synthetic data (`synthetic.py`): AR(1) features, `y_t = a * x_t + rho * b + eps_t` with `b = (+1, -1)`. The predictor has no intercept, so `rho` controls how misspecified it is. An offset along `(1, 1)` would not matter: under the budget constraint it shifts the cost of every portfolio by the same constant (tested).

Full derivations: [docs/formulation.md](docs/formulation.md).

## Methods

| Method | Module | Trains on | Notes |
|---|---|---|---|
| OLS | `models/linear.py` | MSE | asset-wise, closed form |
| IPO (closed form) | `models/ipo_closed_form.py` | decision cost with `w >= 0` dropped | Butler & Kwon (2023); a d x d linear solve whose solution does not depend on delta |
| DFL one-stage | `models/dfl_onestage.py` | decision cost, exact | KKT (`kkt`) or zero-duality-gap (`dual`) single-level model; Gurobi or Knitro |
| DFL-grid | `models/dfl_grid.py` | decision cost, exact | two assets only; exhaustive grid using the closed-form two-asset MVO, used as a solver-free reference |

## Quick start

```bash
uv sync
uv run python experiments/synthetic_demo.py     # about 10 s on a laptop CPU, no solver needed
uv run pytest
```

The one-stage model needs a nonconvex solver. With the `gurobipy` wheel (its bundled license limits model size, which T = 26 respects):

```bash
uv sync --extra gurobi
uv run pytest -m solver
uv run python experiments/solver_check.py
```

For Knitro, set `KNITROAMPL` to the path of the `knitroampl` executable.

## Results (synthetic, two assets)

From [results/table.md](results/table.md): delta = 0.5, 100 training and 2000 test periods per seed, 20 seeds, mean (standard deviation).

| rho | method | test MSE | test regret | seeds with lower test regret than OLS |
|---:|---|---:|---:|---:|
| 0 | OLS | 0.251 (0.005) | 0.0424 (0.0030) | - |
| 0 | IPO | 0.257 (0.010) | 0.0425 (0.0028) | 11/20 |
| 0 | DFL-grid | 1.731 (1.079) | 0.0450 (0.0032) | 1/20 |
| 1 | OLS | 0.463 (0.020) | 0.0183 (0.0050) | - |
| 1 | IPO | 0.509 (0.077) | 0.0142 (0.0048) | 17/20 |
| 1 | DFL-grid | 2.755 (1.034) | 0.0126 (0.0042) | 18/20 |

- With the misspecification (rho = 1), training on decision cost lowers test regret by about 30% relative to OLS (0.0126 vs 0.0183) while test MSE rises roughly sixfold. With two assets the decision depends only on the predicted return difference, not on its level, so the fitted theta can be far from the true one.
- With a well-specified model (rho = 0), OLS is already the decision-optimal rule in the population, and DFL-grid only overfits the 100 training periods (test regret 0.0450 vs 0.0424). In 6 of the 20 seeds its minimum sits on the grid boundary.
- The IPO closed form ignores `w >= 0`; with rho = 1 it recovers part of the gain (0.0142).

The scatter of test MSE against test regret per seed is in [docs/figures/mse_vs_regret.png](docs/figures/mse_vs_regret.png).

**Solver check** ([results/solver_check.md](results/solver_check.md); Gurobi 13.0.3, T = 26, 10 instances, 60 s limit): the `kkt` model closed the optimality gap on all 10 instances (the slowest took 16 s), and its objective was never above the brute-force reference (up to 2.6e-3 below it, where the grid is coarse or too small). The `dual` model reached the same objective whenever it closed the gap, but it did not close it on 5 of the 10 instances (4 hit the 60 s limit, 1 stopped with status `other`), and on 2 of those its objective was worse than the reference.

## Design decisions

- **No intercept.** All methods then learn the same d parameters, and with two assets the whole hypothesis class is the plane in the figure.
- **Brute-force reference instead of trusting the solver.** For d = 2 the lower-level MVO has a closed form, so `L(theta)` is exact anywhere. The grid always includes the OLS and IPO points, so the reference is never worse than them.
- **Compare objective values, not parameters.** `L` is flat wherever all decisions are corner portfolios, so the minimiser is not unique; on one instance the `dual` model returned theta = (454, 558) with the same objective as the `kkt` model's (2.8, 3.3).
- **Two single-level variants.** `kkt` puts all nonconvexity in T x d complementarity products. `dual` aggregates them into one quadratic equality per period. Both have the same feasible set; on these instances Gurobi handled `kkt` much better.
- **theta starts from the IPO solution.** It is cheap and already accounts for the decision problem.
- **Solvers are optional.** Gurobi is an extra and Knitro is found through an environment variable, so CI runs everything except the solver tests.

## What the tests check

| File | Checks |
|---|---|
| `test_problem.py` | simplex projection; the two-asset closed form matches projected gradient and a fine scan; delta -> 0 picks the best predicted asset; perfect foresight has zero regret; an offset along `(1, 1)` leaves every regret unchanged |
| `test_ipo_closed_form.py` | the affine decision satisfies the budget KKT conditions; the closed-form theta matches a numerical (BFGS) minimiser for several delta; independence of delta; invalid delta is rejected |
| `test_dfl_grid.py` | batched evaluation matches `problem.decision_cost`; the result is never worse than OLS or IPO; the grid covers the anchors |
| `test_dfl_onestage.py` | variable and constraint counts (132 variables for T = 26); the closed-form lower-level solution plus its multipliers satisfies every constraint and reproduces `L(theta)` (no solver needed); with a solver (`-m solver`): the objective equals `L(theta)` and is no worse than the reference |

The solver tests are skipped automatically when neither Gurobi nor Knitro is available, and CI runs `pytest -m "not solver"`. They were run locally with Gurobi 13.0.3.

## Repository layout

```
src/dfl_portfolio/
  problem.py            mean-variance cost, two-asset closed form, projected-gradient oracle, regret
  synthetic.py          AR(1) features with a controllable offset
  solvers.py            Gurobi / Knitro wrappers
  models/               linear (OLS), ipo_closed_form, dfl_grid, dfl_onestage
experiments/            synthetic_demo.py, solver_check.py
docs/                   formulation.md, figures/
results/                tables written by the experiments
tests/
```

## Limitations

- Synthetic data only, and the brute-force reference is limited to two assets.
- The covariance is the true one and fixed over time; estimation error in `V` is not modelled.
- The one-stage model is nonconvex. With more periods or assets, Gurobi's global search slows down quickly, and Knitro only returns local solutions.
- DFL-grid has no regularisation, so its in-sample optimum can drift (see the rho = 0 row). The `lambda_anchor` option of the one-stage model is the hook for this, but it is not studied here.
- Runtimes are from one laptop and are only indicative.

## References

- Butler, A., & Kwon, R. H. (2023). Integrating prediction in mean-variance portfolio optimization. *Quantitative Finance*, 23(3), 429-452. arXiv:2102.09287
- Elmachtoub, A. N., & Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science*, 68(1), 9-26. arXiv:1710.08005
- Amos, B., & Kolter, J. Z. (2017). OptNet: Differentiable optimization as a layer in neural networks. *ICML 2017*. arXiv:1703.00443
- Muñoz, M. A., Pineda, S., & Morales, J. M. (2022). A bilevel framework for decision-making under uncertainty with contextual information. *Omega*, 108, 102575.
- Scheel, H., & Scholtes, S. (2000). Mathematical programs with complementarity constraints: Stationarity, optimality, and sensitivity. *Mathematics of Operations Research*, 25(1), 1-22.
- Bynum, M. L., et al. (2021). *Pyomo: Optimization Modeling in Python* (3rd ed.). Springer.

## Acknowledgements

Thanks to my thesis advisor for supervising the research this code comes from.

## License

MIT. See [LICENSE](LICENSE).

---

**日本語の要約**: 卒業研究（Decision-Focused Learning × 平均分散ポートフォリオ）のコードを、合成データの部分だけ整理して書き直したものです。予測の二乗誤差ではなく、その予測で組んだポートフォリオのコストで学習します。下位の最適化問題を KKT 条件で置き換えた一段化モデル（Pyomo）を、ソルバー不要の閉形式と総当たりの参照解で検証しています。上の数値はすべて、このリポジトリのスクリプトを実行して得たものです。
