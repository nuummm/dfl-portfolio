"""Two-asset synthetic demo: where MSE-optimal and decision-optimal predictors part ways.

    uv run python experiments/synthetic_demo.py              # figures + results/table.md

Writes docs/figures/landscape.png, docs/figures/mse_vs_regret.png and results/table.md.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dfl_portfolio.models.baselines import best_constant_portfolio, fit_ols_intercept
from dfl_portfolio.models.dfl_grid import fit_dfl_grid
from dfl_portfolio.models.ipo_closed_form import fit_ipo
from dfl_portfolio.models.linear import fit_ols, predict
from dfl_portfolio.problem import decide, decision_cost, mse, mvo_cost
from dfl_portfolio.synthetic import SyntheticConfig, generate

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "well-specified": SyntheticConfig(rho=0.0),
    "offset": SyntheticConfig(misspec="offset", rho=0.5),
    "nonlinear": SyntheticConfig(misspec="nonlinear", rho=2.0),
}
STYLE = {  # fixed categorical order plus a distinct marker per method
    "OLS": dict(color="#2a78d6", marker="o"),
    "IPO": dict(color="#eb6834", marker="s"),
    "DFL-grid": dict(color="#1baf7a", marker="*"),
    "OLS + intercept": dict(color="#eda100", marker="^"),
}
TABLE_ORDER = ["OLS", "OLS + intercept", "best constant", "IPO", "DFL-grid"]
PNG_META = {"Software": None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--grid", type=int, default=201, help="grid points per axis for DFL-grid")
    p.add_argument("--scale", type=float, default=3.0, help="DFL-grid box width / anchor spread")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    p.add_argument("--out", type=Path, default=ROOT)
    return p.parse_args()


def fit_all(Xtr, Ytr, Vtr, args):
    """Return the decision rule of every method as ``X, V -> (yhat or None, W)``."""
    delta = args.delta
    thetas = {"OLS": fit_ols(Xtr, Ytr), "IPO": fit_ipo(Xtr, Ytr, Vtr, delta)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # counted in the table instead
        grid = fit_dfl_grid(
            Xtr,
            Ytr,
            Vtr,
            delta,
            anchors=np.stack([thetas["OLS"], thetas["IPO"]]),
            n_grid=args.grid,
            scale=args.scale,
        )
    thetas["DFL-grid"] = grid.theta

    def linear_rule(theta):
        return lambda X, V: (predict(X, theta), decide(predict(X, theta), V, delta))

    rules = {name: linear_rule(theta) for name, theta in thetas.items()}
    beta, alpha = fit_ols_intercept(Xtr, Ytr)
    rules["OLS + intercept"] = lambda X, V: (X * beta + alpha, decide(X * beta + alpha, V, delta))
    w_const = best_constant_portfolio(Ytr, Vtr, delta)
    rules["best constant"] = lambda X, V: (None, np.broadcast_to(w_const, X.shape))
    return rules, thetas, grid


def evaluate(rule, X, Y, V, delta):
    """(regret, MSE or nan, share of decisions strictly inside the simplex)."""
    yhat, W = rule(X, V)
    reg = float(np.mean(mvo_cost(W, Y, V, delta))) - decision_cost(Y, Y, V, delta)
    err = np.nan if yhat is None else mse(yhat, Y)
    interior = float(np.mean(np.all(W > 1e-9, axis=1)))
    return reg, err, interior


def run(args: argparse.Namespace):
    records = []  # (scenario, seed, method, train_regret, test_mse, test_regret, interior)
    examples = {}
    boundary = {name: 0 for name in args.scenarios}
    for name in args.scenarios:
        cfg = SCENARIOS[name]
        for seed in range(args.seeds):
            X, Y, V = generate(args.n_train + args.n_test, cfg, np.random.default_rng(seed))
            tr, te = slice(0, args.n_train), slice(args.n_train, None)
            rules, thetas, grid = fit_all(X[tr], Y[tr], V[tr], args)
            boundary[name] += grid.on_boundary
            for method, rule in rules.items():
                train_reg, _, _ = evaluate(rule, X[tr], Y[tr], V[tr], args.delta)
                test_reg, test_mse, interior = evaluate(rule, X[te], Y[te], V[te], args.delta)
                records.append((name, seed, method, train_reg, test_mse, test_reg, interior))
            if seed == 0:
                examples[name] = (X[tr], Y[tr], V[tr], thetas, grid)
    return records, examples, boundary


def plot_landscape(examples, delta, path: Path) -> None:
    fig, axes = plt.subplots(1, len(examples), figsize=(5.2 * len(examples), 4.6), squeeze=False)
    for ax, (name, (X, Y, V, thetas, grid)) in zip(axes[0], examples.items(), strict=True):
        oracle = decision_cost(Y, Y, V, delta)
        reg = np.maximum(grid.costs - oracle, 1e-4)
        T1, T2 = np.meshgrid(grid.theta1, grid.theta2, indexing="ij")
        im = ax.pcolormesh(T1, T2, reg, cmap="Greys", norm=LogNorm(), shading="auto")
        sse = ((X[None, None] * np.stack([T1, T2], -1)[:, :, None, :] - Y) ** 2).mean(axis=(2, 3))
        levels = np.quantile(sse, [0.02, 0.1, 0.25, 0.5])
        cs = ax.contour(T1, T2, sse, levels=levels, colors="#52514e", linewidths=1.0)
        ax.clabel(cs, fmt="%.2f", fontsize=7)
        for method, theta in thetas.items():  # decreasing size keeps coincident points visible
            ms = {"OLS": 13, "IPO": 9}.get(method, 14)
            ax.plot(
                *theta, ls="none", ms=ms, mec="white", mew=1.5, clip_on=False, zorder=5,
                label=method, **STYLE[method],
            )  # fmt: skip
        ax.set_title(f"{name} (seed 0, training set)", fontsize=10)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        fig.colorbar(im, ax=ax, label="in-sample regret (log scale)")
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Decision regret over the parameter plane; contours = training MSE", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata=PNG_META)
    plt.close(fig)


def plot_mse_vs_regret(records, path: Path) -> None:
    names = list(dict.fromkeys(r[0] for r in records))
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.0), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        for method in STYLE:
            pts = np.array([(r[4], r[5]) for r in records if r[0] == name and r[2] == method])
            if pts.size:
                ax.plot(
                    pts[:, 0], pts[:, 1], ls="none", ms=6, alpha=0.8, label=method, **STYLE[method]
                )
        ax.set_title(f"{name}, one point per seed", fontsize=10)
        ax.set_xlabel("test MSE")
        ax.set_ylabel("test regret")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata=PNG_META)
    plt.close(fig)


def write_table(records, boundary, args, path: Path) -> None:
    lines = [
        "# Synthetic two-asset results",
        "",
        f"Generated by `experiments/synthetic_demo.py` with delta = {args.delta}, "
        f"n_train = {args.n_train}, n_test = {args.n_test}, {args.seeds} seeds, "
        f"{args.grid}x{args.grid} grid (scale {args.scale:g}). Scenarios (`SyntheticConfig`): "
        + "; ".join(
            f"{name}: rho = 0"
            if SCENARIOS[name].rho == 0
            else f"{name}: misspec = {SCENARIOS[name].misspec}, rho = {SCENARIOS[name].rho:g}"
            for name in boundary
        )
        + ".",
        "Mean (standard deviation) over seeds. Regret is the average cost above the "
        "perfect-foresight portfolio; lower is better. Interior = share of test periods whose "
        "portfolio holds both assets. OLS, IPO and DFL use the same no-intercept model "
        "`yhat = x * theta`; OLS + intercept has a larger model class and the best constant "
        "portfolio ignores the features.",
        "",
        "| scenario | method | train regret | test MSE | test regret | interior | "
        "lower test regret than OLS | lower than OLS + intercept |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in boundary:
        by_method = {
            m: {r[1]: r for r in records if r[0] == name and r[2] == m} for m in TABLE_ORDER
        }
        for method in TABLE_ORDER:
            rows = by_method[method]
            if not rows:
                continue
            a = np.array([r[3:] for r in rows.values()])
            m, s = a.mean(axis=0), a.std(axis=0, ddof=1)
            err = "-" if np.isnan(m[1]) else f"{m[1]:.3f} ({s[1]:.3f})"
            wins = []
            for base in ("OLS", "OLS + intercept"):
                if method == base:
                    wins.append("-")
                else:
                    n = sum(r[5] < by_method[base][seed][5] for seed, r in rows.items())
                    wins.append(f"{n}/{len(rows)}")
            lines.append(
                f"| {name} | {method} | {m[0]:.4f} ({s[0]:.4f}) | {err} "
                f"| {m[2]:.4f} ({s[2]:.4f}) | {m[3]:.2f} | {wins[0]} | {wins[1]} |"
            )
    lines += [
        "",
        "DFL-grid minimum on the grid boundary (the in-sample optimum may lie further out): "
        + ", ".join(f"{name}: {n}/{args.seeds} seeds" for name, n in boundary.items())
        + ".",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    records, examples, boundary = run(args)
    (args.out / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    plot_landscape(examples, args.delta, args.out / "docs" / "figures" / "landscape.png")
    plot_mse_vs_regret(records, args.out / "docs" / "figures" / "mse_vs_regret.png")
    write_table(records, boundary, args, args.out / "results" / "table.md")
    print((args.out / "results" / "table.md").read_text())
    print(f"elapsed: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
