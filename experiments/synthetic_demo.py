"""Two-asset synthetic demo: where MSE-optimal and decision-optimal predictors part ways.

    uv run python experiments/synthetic_demo.py              # figures + results/table.md
    uv run python experiments/synthetic_demo.py --with-solver gurobi --n-train 26

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

from dfl_portfolio.models.dfl_grid import fit_dfl_grid
from dfl_portfolio.models.ipo_closed_form import fit_ipo
from dfl_portfolio.models.linear import fit_ols, predict
from dfl_portfolio.problem import decision_cost, mse, regret
from dfl_portfolio.synthetic import SyntheticConfig, generate

ROOT = Path(__file__).resolve().parents[1]
STYLE = {  # categorical slots 1-3 plus a distinct marker per method
    "OLS": dict(color="#2a78d6", marker="o"),
    "IPO": dict(color="#eb6834", marker="s"),
    "DFL-grid": dict(color="#1baf7a", marker="*"),
    "DFL-onestage": dict(color="#6250d6", marker="D"),
}
PNG_META = {"Software": None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--grid", type=int, default=201, help="grid points per axis for DFL-grid")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--with-solver", choices=["gurobi", "knitro"], default=None)
    p.add_argument("--out", type=Path, default=ROOT)
    return p.parse_args()


def fit_all(Xtr, Ytr, Vtr, delta, n_grid, solver):
    thetas = {"OLS": fit_ols(Xtr, Ytr), "IPO": fit_ipo(Xtr, Ytr, Vtr, delta)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # counted in the table instead
        grid = fit_dfl_grid(
            Xtr, Ytr, Vtr, delta, anchors=np.stack([thetas["OLS"], thetas["IPO"]]), n_grid=n_grid
        )
    thetas["DFL-grid"] = grid.theta
    if solver is not None:
        from dfl_portfolio.models.dfl_onestage import fit_dfl_onestage

        thetas["DFL-onestage"] = fit_dfl_onestage(
            Xtr, Ytr, Vtr, delta, theta_init=thetas["IPO"], solver=solver
        ).theta
    return thetas, grid


def run(args: argparse.Namespace):
    records = []  # (rho, seed, method, train_regret, test_mse, test_regret)
    examples = {}
    boundary = {rho: 0 for rho in args.rhos}
    for rho in args.rhos:
        cfg = SyntheticConfig(rho=rho)
        for seed in range(args.seeds):
            X, Y, V = generate(args.n_train + args.n_test, cfg, np.random.default_rng(seed))
            tr, te = slice(0, args.n_train), slice(args.n_train, None)
            thetas, grid = fit_all(X[tr], Y[tr], V[tr], args.delta, args.grid, args.with_solver)
            boundary[rho] += grid.on_boundary
            for name, theta in thetas.items():
                records.append(
                    (
                        rho,
                        seed,
                        name,
                        regret(predict(X[tr], theta), Y[tr], V[tr], args.delta),
                        mse(predict(X[te], theta), Y[te]),
                        regret(predict(X[te], theta), Y[te], V[te], args.delta),
                    )
                )
            if seed == 0:
                examples[rho] = (X[tr], Y[tr], V[tr], thetas, grid)
    return records, examples, boundary


def plot_landscape(examples, delta, path: Path) -> None:
    fig, axes = plt.subplots(1, len(examples), figsize=(5.2 * len(examples), 4.6), squeeze=False)
    for ax, (rho, (X, Y, V, thetas, grid)) in zip(axes[0], examples.items(), strict=True):
        oracle = decision_cost(Y, Y, V, delta)
        reg = np.maximum(grid.costs - oracle, 1e-4)
        T1, T2 = np.meshgrid(grid.theta1, grid.theta2, indexing="ij")
        im = ax.pcolormesh(T1, T2, reg, cmap="Greys", norm=LogNorm(), shading="auto")
        sse = ((X[None, None] * np.stack([T1, T2], -1)[:, :, None, :] - Y) ** 2).mean(axis=(2, 3))
        levels = np.quantile(sse, [0.02, 0.1, 0.25, 0.5])
        cs = ax.contour(T1, T2, sse, levels=levels, colors="#52514e", linewidths=1.0)
        ax.clabel(cs, fmt="%.2f", fontsize=7)
        for name, theta in thetas.items():  # decreasing size keeps coincident points visible
            ms = {"OLS": 13, "IPO": 9}.get(name, 14)
            ax.plot(*theta, ls="none", ms=ms, mec="white", mew=1.5, label=name, **STYLE[name])
        ax.set_title(f"rho = {rho:g} (seed 0, training set)", fontsize=10)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        fig.colorbar(im, ax=ax, label="in-sample regret (log scale)")
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Decision regret over the parameter plane; contours = training MSE", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata=PNG_META)
    plt.close(fig)


def plot_mse_vs_regret(records, path: Path) -> None:
    rhos = sorted({r[0] for r in records})
    fig, axes = plt.subplots(1, len(rhos), figsize=(5.2 * len(rhos), 4.0), squeeze=False)
    for ax, rho in zip(axes[0], rhos, strict=True):
        for name in STYLE:
            pts = np.array([(r[4], r[5]) for r in records if r[0] == rho and r[2] == name])
            if pts.size:
                ax.plot(pts[:, 0], pts[:, 1], ls="none", ms=6, alpha=0.8, label=name, **STYLE[name])
        ax.set_title(f"rho = {rho:g}, one point per seed", fontsize=10)
        ax.set_xlabel("test MSE")
        ax.set_ylabel("test regret")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata=PNG_META)
    plt.close(fig)


def write_table(records, boundary, args, elapsed: float, path: Path) -> None:
    lines = [
        "# Synthetic two-asset results",
        "",
        f"Generated by `experiments/synthetic_demo.py` with delta = {args.delta}, "
        f"n_train = {args.n_train}, n_test = {args.n_test}, default `SyntheticConfig`, "
        f"{args.seeds} seeds, {args.grid}x{args.grid} grid ({elapsed:.0f} s).",
        "Mean (standard deviation) over seeds. Regret is the average cost above the "
        "perfect-foresight portfolio; lower is better.",
        "",
        "| rho | method | train regret | test MSE | test regret | seeds with lower test "
        "regret than OLS |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    names = [n for n in STYLE if any(r[2] == n for r in records)]
    for rho in sorted({r[0] for r in records}):
        ols = {r[1]: r[5] for r in records if r[0] == rho and r[2] == "OLS"}
        for name in names:
            rows = [r for r in records if r[0] == rho and r[2] == name]
            a = np.array([r[3:] for r in rows])
            m, s = a.mean(axis=0), a.std(axis=0, ddof=1)
            wins = "-" if name == "OLS" else f"{sum(r[5] < ols[r[1]] for r in rows)}/{len(rows)}"
            lines.append(
                f"| {rho:g} | {name} | {m[0]:.4f} ({s[0]:.4f}) | {m[1]:.3f} ({s[1]:.3f}) "
                f"| {m[2]:.4f} ({s[2]:.4f}) | {wins} |"
            )
    lines += [
        "",
        "DFL-grid minimum on the grid boundary (the in-sample optimum may lie further out): "
        + ", ".join(f"rho = {rho:g}: {n}/{args.seeds} seeds" for rho, n in boundary.items())
        + ".",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    records, examples, boundary = run(args)
    elapsed = time.perf_counter() - start
    (args.out / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    plot_landscape(examples, args.delta, args.out / "docs" / "figures" / "landscape.png")
    plot_mse_vs_regret(records, args.out / "docs" / "figures" / "mse_vs_regret.png")
    write_table(records, boundary, args, elapsed, args.out / "results" / "table.md")
    print((args.out / "results" / "table.md").read_text())


if __name__ == "__main__":
    main()
