# Formulation

Notation follows the code: $T$ periods, $d$ assets, features $x_t \in \mathbb{R}^d$,
realised returns $y_t \in \mathbb{R}^d$, covariance $V_t \succ 0$, risk weight
$\delta \in [0, 1]$ and $\alpha = 1 - \delta$.

## 1. Decision problem

The predictor is linear per asset and has no intercept,

$$\hat y_t = x_t \odot \theta, \qquad \theta \in \mathbb{R}^d .$$

Given a prediction, the decision is the long-only mean-variance portfolio

$$
w^\star(\hat y, V) = \arg\min_{w \in \Delta} \; c(w; \hat y, V), \qquad
c(w; y, V) = -\alpha\, w^\top y + \tfrac{\delta}{2}\, w^\top V w, \qquad
\Delta = \{ w \ge 0,\ \mathbf 1^\top w = 1 \}.
$$

For $\delta > 0$ the problem is strictly convex, so $w^\star$ is unique. For $\delta = 0$
it is a linear program whose solution puts everything on the asset with the largest
prediction (`problem.solve_mvo_2asset`, `problem.solve_mvo_pgd`).

## 2. Learning objectives

* **Prediction error** (OLS): $\frac{1}{Td} \sum_t \lVert \hat y_t - y_t \rVert^2$, the mean over periods and assets (`problem.mse`).
* **Decision cost** (DFL): $L(\theta) = \frac1T \sum_t c\big(w^\star(x_t \odot \theta, V_t);\ y_t, V_t\big)$.
* **Regret**: $L(\theta)$ minus the same average for the perfect-foresight decisions
  $w^\star(y_t, V_t)$. It is non-negative and zero for perfect predictions.

Because $c$ is linear in $y$, the best decision given $x$ is $w^\star(\mathbb E[y \mid x], V)$.
When the model class contains $\mathbb E[y \mid x]$, accurate prediction is therefore also
the best decision rule, and decision-focused training can only help through finite-sample
effects. The interesting case is misspecification. `synthetic.py` generates

$$y_t = a \odot x_t + \rho\, b \odot g(x_t) + \varepsilon_t$$

with one of two terms the model cannot represent:

* **offset**, $g = 1$: a constant tilt between the assets. A per-asset intercept removes it,
  so the fair comparison is OLS with an intercept, not a cleverer loss.
* **nonlinear**, $g(x) = \lvert x - \mu_x \rvert - \mathbb E \lvert x - \mu_x \rvert$:
  a V-shaped term with zero mean under the stationary distribution of $x$. An intercept
  cannot absorb it, so the best predictor within any linear class is still misspecified.

**A shift along $\mathbf 1$ changes nothing.** For $w \in \Delta$,
$c(w; y + \kappa \mathbf 1, V) = c(w; y, V) - \alpha \kappa$. A common offset on all assets
moves every portfolio's cost by the same constant, so it changes neither $w^\star$ nor the
regret of any $\theta$ (`tests/test_problem.py::test_offset_along_ones_changes_no_regret`).
This is why the synthetic term uses $b = (+1, -1)$.

## 3. Two assets in closed form

With $w = (w_1, 1 - w_1)$,

$$
c(w_1) = \tfrac{\delta}{2} s\, w_1^2
- \big[\alpha(\hat y_1 - \hat y_2) + \delta(V_{22} - V_{12})\big] w_1 + \text{const},
\qquad s = V_{11} + V_{22} - 2V_{12} \ge 0 ,
$$

so

$$
w_1^\star = \operatorname{clip}_{[0,1]}
\frac{\alpha(\hat y_1 - \hat y_2) + \delta(V_{22} - V_{12})}{\delta s}.
$$

If $\delta s = 0$ the cost is linear in $w_1$ and the solution is a corner. This makes
$L(\theta)$ cheap to evaluate anywhere in the $\theta$ plane, which is what the brute-force
reference (`models/dfl_grid.py`) and the landscape figure rely on.

## 4. IPO closed form (Butler & Kwon, 2023)

Drop $w \ge 0$ and keep only the budget constraint. With $M = V^{-1}$, $a = M \mathbf 1$
and $\delta_{\text{eff}} = \delta / \alpha$ (which requires $0 < \delta < 1$), the relaxed
decision is affine in the prediction:

$$
w(\hat y) = \frac{1}{\delta_{\text{eff}}} H \hat y + c, \qquad
H = \Big(I - \frac{a \mathbf 1^\top}{\mathbf 1^\top a}\Big) M, \qquad
c = \frac{a}{\mathbf 1^\top a}.
$$

$H \mathbf 1 = 0$ and $H^\top \mathbf 1 = 0$. Substituting $\hat y_t = D_t \theta$ with
$D_t = \operatorname{diag}(x_t)$ and $S_t = H_t^\top V_t H_t$,

$$
L_{\text{relaxed}}(\theta) = \frac{\alpha^2}{2\delta T}\, \theta^\top \Big(\sum_t D_t S_t D_t\Big) \theta
- \frac{\alpha^2}{\delta T}\, \theta^\top \sum_t D_t H_t^\top y_t + \text{const}.
$$

The cross term $\theta^\top D_t H_t^\top V_t c_t$ vanishes because $V_t c_t \propto \mathbf 1$.
The common factor $\alpha^2/(\delta T)$ cancels in the normal equations, so

$$
\theta_{\text{IPO}} = \Big(\sum_t D_t S_t D_t\Big)^{-1} \sum_t D_t H_t^\top y_t ,
$$

which does not depend on $\delta$ (`tests/test_ipo_closed_form.py`). The estimator is
exact for the relaxed problem only. Once $w \ge 0$ binds, it no longer minimises $L$; that
gap is the reason for the one-stage model below.

## 5. One-stage model (`models/dfl_onestage.py`)

The lower-level problem is a convex QP with linear constraints, so its KKT conditions are
necessary and sufficient. Replacing it by them gives a single-level problem in
$(\theta, w, \lambda, \mu)$ (the single-level approach to contextual decision-making is
studied by Muñoz, Pineda & Morales, 2022):

$$
\begin{aligned}
\min_{\theta, w, \lambda, \mu}\quad & \frac1T \sum_t c(w_t; y_t, V_t)
  + \frac{\lambda_{\text{anchor}}}{2} \lVert \theta - \theta_{\text{anchor}} \rVert^2 \\
\text{s.t.}\quad & \delta V_t w_t - \alpha\, x_t \odot \theta - \mu_t \mathbf 1 - \lambda_t = 0
  && \text{(stationarity, } Td \text{ linear equalities)} \\
& \mathbf 1^\top w_t = 1, \quad w_t \ge 0, \quad \lambda_t \ge 0 && \text{(feasibility)} \\
& w_{tj}\, \lambda_{tj} = 0 && \text{(complementarity, } Td \text{ bilinear equalities)}
\end{aligned}
$$

There are $d + 2Td + T$ variables: 132 for $d = 2$, $T = 26$. $\lambda_{\text{anchor}} = 0$
by default. $\theta$ starts from $\theta_{\text{IPO}}$ and has no bounds, so with
$\lambda_{\text{anchor}} = 0$ the bilinear terms in $(\theta, w)$ are unbounded too.

**`dual` variant.** Multiplying stationarity by $w_t^\top$ and using the budget gives
$\lambda_t^\top w_t = \delta\, w_t^\top V_t w_t - \alpha\, \hat y_t^\top w_t - \mu_t$. Since
$w_t, \lambda_t \ge 0$, the $d$ complementarity products can therefore be replaced by one
zero-duality-gap equality per period,

$$
\delta\, w_t^\top V_t w_t - \alpha\, (x_t \odot \theta)^\top w_t = \mu_t ,
$$

which contains both a quadratic term in $w_t$ and bilinear terms in $(\theta, w_t)$. The
two variants have the same feasible set but behave differently in a solver.

**Solvers.** Both variants are nonconvex. Complementarity constraints violate the usual
constraint qualifications at every feasible point (see Scheel & Scholtes, 2000), so a local
NLP solver such as Knitro returns a local solution without the usual guarantees (the
Knitro wrapper in `solvers.py` was not run for the results in this repository). Gurobi's
spatial branch-and-bound (`NonConvex=2`) certifies a global optimum when the reported gap
closes. `experiments/solver_check.py` compares both against the brute-force reference on
small instances.

## 6. What the reference can and cannot check

$L$ depends on $\theta$ only through the decisions it induces. It is piecewise smooth, and
flat wherever all decisions sit at the same corners, so the minimiser is generally not
unique. Scaling $\theta$ up pushes more decisions to corners, and once most of them are
corners $L$ changes little along such a direction. The grid minimum therefore often lies
on the edge of the default box (three times the spread of the OLS and IPO anchors,
at least 3 wide): in 6, 9 and 8 of 20 seeds for the well-specified, offset and nonlinear
scenarios of `experiments/synthetic_demo.py`. Rerunning with `--scale 10` (a box about
3.3 times wider) moves the minimiser further out and inflates the test MSE of DFL-grid
roughly 8 to 11 times, but its mean test regret changes by less than 0.001 in every
scenario. Even then the minimum is on the edge in 7, 4 and 3 of 20 seeds, so the
in-sample optimum is not bracketed in every seed; the test regret is simply insensitive
to how far out along the valley it lies. The location of the minimiser is weakly
identified; the test regret it achieves is not. Comparisons between methods and
solvers therefore use objective values, never $\theta$ itself.
