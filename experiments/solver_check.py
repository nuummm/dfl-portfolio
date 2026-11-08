"""Check the one-stage model against the brute-force reference on small instances.

    uv sync --extra gurobi
    uv run python experiments/solver_check.py

T = 26 keeps the KKT model (132 variables, 52 bilinear constraints) within the size limits
of the license bundled with the ``gurobipy`` wheel. Writes results/solver_check.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dfl_portfolio.models.dfl_grid import fit_dfl_grid
from dfl_portfolio.models.dfl_onestage import fit_dfl_onestage
from dfl_portfolio.models.ipo_closed_form import fit_ipo
from dfl_portfolio.models.linear import fit_ols
from dfl_portfolio.problem import decision_cost
from dfl_portfolio.synthetic import SyntheticConfig, generate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-train", type=int, default=26)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "solver_check.md")
    args = p.parse_args()

    import gurobipy

    lines = [
        "# One-stage model vs. brute-force reference",
        "",
        f"Gurobi {'.'.join(map(str, gurobipy.gurobi.version()))}, 1 thread, "
        f"time limit {args.time_limit:g} s, T = {args.n_train}, delta = {args.delta}. "
        "theta starts at the IPO solution. `ref` is the best of a 401 x 401 grid and the "
        "OLS / IPO points. `L(theta)` re-evaluates the returned theta with the exact "
        "lower-level solution. Runtimes are wall-clock on one laptop and only indicative.",
        "",
        "| rho | seed | formulation | theta | objective | L(theta) | ref | objective - ref | gap | "
        "time [s] | status |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rho in args.rhos:
        for seed in range(args.seeds):
            X, Y, V = generate(args.n_train, SyntheticConfig(rho=rho), np.random.default_rng(seed))
            anchors = np.stack([fit_ols(X, Y), fit_ipo(X, Y, V, args.delta)])
            ref = fit_dfl_grid(X, Y, V, args.delta, anchors=anchors, n_grid=401).cost
            for formulation in ("kkt", "dual"):
                res = fit_dfl_onestage(
                    X, Y, V, args.delta, formulation=formulation, time_limit=args.time_limit
                )
                exact = decision_cost(X * res.theta, Y, V, args.delta)
                info = res.info
                lines.append(
                    f"| {rho:g} | {seed} | {formulation} "
                    f"| ({res.theta[0]:.3g}, {res.theta[1]:.3g}) "
                    f"| {res.objective:.6f} | {exact:.6f} "
                    f"| {ref:.6f} | {res.objective - ref:+.1e} | {info['gap']:.1e} "
                    f"| {info['runtime']:.1f} | {info['termination']} |"
                )
                print(lines[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
