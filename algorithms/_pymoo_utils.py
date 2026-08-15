"""Shared plumbing for the pymoo evolutionary multi-objective baselines.

These baselines exist to *showcase* that BoCoDe's multi-objective problems drive
pymoo's evolutionary optimizers unchanged. The optimization logic lives entirely in
pymoo; this module only supplies the two adapters every pymoo baseline needs:

* :class:`_BoCoDePymooProblem` -- wraps a BoCoDe :class:`MultiObjectiveProblem` as a
  ``pymoo.core.problem.Problem`` over the unit cube ``[0, 1]^d``. BoCoDe **maximizes**
  and its constraints are feasible when ``c <= 0``; pymoo **minimizes** and its
  inequality constraints are feasible when ``g <= 0``. So the objectives are negated
  for pymoo and the constraints pass straight through. The wrapper also *logs every
  point pymoo evaluates, in evaluation order*, which is what lets us report a running
  hypervolume comparable with the BO baselines.

* :func:`run_pymoo` -- sizes the pymoo run so the total number of objective
  evaluations matches the BO baselines' budget (``n_init + iters``), runs it, then
  computes the **running hypervolume** over the evaluated stream with the *exact* same
  ``DominatedPartitioning`` + fixed ``ref_point`` convention the qNEHVI / qNParEGO
  baselines use (:class:`MultiObjectiveProblem.hv_ref_point`), so the numbers are
  directly comparable. Entry ``0`` of the trace is the HV of the first ``n_init``
  evaluated points (the "initial design"), entry ``k`` the HV after ``n_init + k``
  points -- identical to how the BO baselines index their trace.

Sources:
J. Blank and K. Deb. pymoo: Multi-Objective Optimization in Python. IEEE Access, 2020. https://pymoo.org
"""

from __future__ import annotations

import math

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from ._bo_utils import (
    DTYPE,
    MultiObjectiveProblem,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    make_problem,
    set_seed,
)


class _BoCoDePymooProblem(Problem):
    """BoCoDe multi-objective problem as a pymoo ``Problem`` over the unit cube.

    Objectives are negated (BoCoDe maximizes, pymoo minimizes); constraints pass
    through unchanged (both use ``<= 0`` feasible). Every evaluated batch is logged in
    order into ``Y_hist`` (BoCoDe's raw maximization values), ``X_hist`` (the unit-cube
    points) and ``C_hist`` (constraints, when constrained).
    """

    def __init__(self, obj: MultiObjectiveProblem, constrained: bool):
        n_constr = obj.num_constraints if constrained else 0
        super().__init__(
            n_var=obj.dim,
            n_obj=obj.num_objectives,
            n_ieq_constr=n_constr,
            xl=0.0,
            xu=1.0,
        )
        self._obj = obj
        self._constrained = constrained
        self.X_hist: list = []
        self.Y_hist: list = []
        self.C_hist: list = []

    def _evaluate(self, X, out, *args, **kwargs):
        Xt = torch.from_numpy(np.asarray(X)).to(DTYPE)
        values, constraints = self._obj.evaluate_raw(Xt)
        values_np = values.detach().cpu().numpy()
        out["F"] = -values_np  # pymoo minimizes; BoCoDe maximizes
        self.X_hist.append(np.asarray(X, dtype=float).copy())
        self.Y_hist.append(values_np.copy())
        if self.n_ieq_constr > 0:
            c_np = constraints.detach().cpu().numpy()
            out["G"] = c_np  # feasible when <= 0 in both conventions
            self.C_hist.append(c_np.copy())


def run_pymoo(
    problem,
    method: str,
    algo_factory,
    n_init: int | None,
    iters: int,
    seed: int,
    constrained: bool,
) -> Result:
    """Drive a pymoo evolutionary optimizer on a BoCoDe multi-objective problem.

    ``algo_factory(pop_size)`` returns the pymoo algorithm to run. The population size
    is set to ``n_init`` and the number of generations chosen so the total number of
    objective evaluations (``pop_size * n_gen``) covers ``n_init + iters`` -- the same
    budget the BO baselines use. The reported ``per_iteration_value`` is the running
    hypervolume over the first ``n_init + k`` evaluated points, in BoCoDe's maximization
    frame with the problem's fixed reference point.
    """
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    if constrained:
        assert obj.num_constraints > 0, f"{method} requires a constrained problem"
    total = n_init + iters
    pop_size = n_init
    # +2 generations of head-room so duplicate elimination never leaves us short of
    # the target budget; we only ever read the first `total` evaluated points.
    n_gen = math.ceil(total / pop_size) + 2

    pymoo_prob = _BoCoDePymooProblem(obj, constrained)
    minimize(
        pymoo_prob,
        algo_factory(pop_size),
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    Y = np.concatenate(pymoo_prob.Y_hist, axis=0)
    X = np.concatenate(pymoo_prob.X_hist, axis=0)
    C = np.concatenate(pymoo_prob.C_hist, axis=0) if constrained else None
    use = min(total, Y.shape[0])
    Y, X = Y[:use], X[:use]
    C = None if C is None else C[:use]

    ref_point = obj.hv_ref_point(torch.from_numpy(Y).to(DTYPE))

    def hv(k: int) -> float:
        k = min(k, use)
        Yk = torch.from_numpy(Y[:k]).to(DTYPE)
        if constrained:
            feas = (torch.from_numpy(C[:k]).to(DTYPE) <= 0).all(dim=1)
            if not feas.any():
                return 0.0
            Yk = Yk[feas]
        return (
            DominatedPartitioning(ref_point=ref_point.cpu(), Y=Yk.cpu())
            .compute_hypervolume()
            .item()
        )

    res = Result(method, type(problem).__name__, seed, acquisition_function="none")
    res.start(hv(n_init))
    for k in range(1, iters + 1):
        res.record(hv(n_init + k))
    res.set_history(X, Y, n_init, c=C)
    return res


def pymoo_main(description: str, method: str, optimize_problem) -> None:
    """Shared CLI for the pymoo baselines (CPU only -- no ``--device``)."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size / pymoo pop_size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=50)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args), args.init, args.iters, args.seed
    )
    finalize(res, args)
