"""Thin wrappers for Gurobi and Knitro.

* Gurobi (``gurobipy``, optional extra): spatial branch-and-bound with ``NonConvex=2``; the
  result is a global optimum only when the reported gap is closed.
* Knitro through its AMPL driver, located by the ``KNITROAMPL`` environment variable; the
  result is a local solution. This path has not been run against the results in this
  repository, and option names may differ between Knitro versions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyomo.environ as pyo

GUROBI_DEFAULTS = {"NonConvex": 2, "Threads": 1, "Seed": 0, "MIPGap": 1e-6}
KNITRO_DEFAULTS = {"feastol": 1e-8, "opttol": 1e-8, "numthreads": 1, "outlev": 0}


def available(name: str) -> bool:
    if name == "gurobi":
        return bool(pyo.SolverFactory("gurobi_persistent").available(exception_flag=False))
    if name == "knitro":
        exe = os.environ.get("KNITROAMPL")
        return bool(exe) and Path(exe).is_file()
    raise ValueError(f"unknown solver {name!r}")


def solve(
    model: pyo.ConcreteModel,
    name: str,
    *,
    time_limit: float = 600.0,
    options: dict | None = None,
    tee: bool = False,
) -> dict:
    """Solve in place; report status, objective and, for Gurobi, best bound, gap and runtime."""
    if name == "gurobi":
        return _solve_gurobi(model, time_limit, options, tee)
    if name == "knitro":
        return _solve_knitro(model, time_limit, options, tee)
    raise ValueError(f"unknown solver {name!r}")


def _solve_gurobi(model, time_limit, options, tee) -> dict:
    opt = pyo.SolverFactory("gurobi_persistent")
    opt.set_instance(model)
    params = {"OutputFlag": int(tee), **GUROBI_DEFAULTS, "TimeLimit": time_limit, **(options or {})}
    for key, value in params.items():
        opt.set_gurobi_param(key, value)
    res = opt.solve(tee=tee, load_solutions=False)
    if opt.get_model_attr("SolCount") == 0:
        raise RuntimeError(f"gurobi found no solution ({res.solver.termination_condition})")
    opt.load_vars()
    return {
        "solver": "gurobi",
        "termination": str(res.solver.termination_condition),
        "objective": float(pyo.value(model.obj)),
        "bound": float(opt.get_model_attr("ObjBound")),
        "gap": float(opt.get_model_attr("MIPGap")),
        "runtime": float(opt.get_model_attr("Runtime")),
    }


def _solve_knitro(model, time_limit, options, tee) -> dict:
    if not available("knitro"):
        raise RuntimeError("set KNITROAMPL to the path of the knitroampl executable")
    opt = pyo.SolverFactory("asl", executable=os.environ["KNITROAMPL"])
    opt.options.update({**KNITRO_DEFAULTS, "maxtime_real": time_limit, **(options or {})})
    res = opt.solve(model, tee=tee, load_solutions=False)
    if len(res.solution) == 0:
        raise RuntimeError(f"knitro returned no solution ({res.solver.termination_condition})")
    model.solutions.load_from(res)
    return {
        "solver": "knitro",
        "termination": str(res.solver.termination_condition),
        "objective": float(pyo.value(model.obj)),
    }
