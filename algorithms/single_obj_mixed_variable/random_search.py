"""Random search baseline for mixed-variable single-objective problems.

The mandatory baseline for the mixed-variable category: draw points uniformly at
random over the unit cube, snap the integer/categorical dimensions to their nearest
allowed values (from the problem's ``variable_types``), and keep the best. Any
constraints are folded into a penalized objective, as in the other baselines.

Run::

    python -m algorithms.single_obj_mixed_variable.random_search --problem PressureVessel --iters 100

Sources:
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse

from .._bo_utils import (
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    initial_design,
    make_problem,
    set_seed,
)
from .single_task_gp import _discrete_grids, _snap


def optimize_problem(problem, iters: int = 100, seed: int = 0) -> Result:
    """Random search over valid mixed-variable points."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result(
        "random_search", type(problem).__name__, seed, acquisition_function="none"
    )

    grids = _discrete_grids(problem)
    X = _snap(initial_design(iters, obj.dim, seed), grids)
    Y = obj(X)

    best = -float("inf")
    for i in range(iters):
        best = max(best, Y[i].item())
        res.record(best)
    res.set_history(X, Y, 0)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--iters", type=int, default=100)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(make_problem(args.problem, args), args.iters, args.seed)
    finalize(res, args)


if __name__ == "__main__":
    main()
