"""Random search baseline for multi-objective problems (hypervolume tracking).

Draws Sobol points and records the dominated hypervolume of the running Pareto
set (in BoCoDe's maximization frame). The mandatory baseline for the
multi-objective categories.

Run::

    python -m algorithms.multi_obj.random_search --problem Penicillin --iters 200

Sources:
M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse

from .._bo_utils import (
    MultiObjectiveProblem,
    Result,
    add_common_args,
    finalize,
    initial_design,  # noqa: E501
    make_problem,
    set_seed,
)
from .._hv_utils import hypervolume_trace


def optimize_problem(problem, iters: int = 200, seed: int = 0) -> Result:
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    res = Result(
        "random_search", type(problem).__name__, seed, acquisition_function="none"
    )

    X = initial_design(iters, obj.dim, seed)
    Y, _ = obj.evaluate_raw(X)
    ref_point = obj.hv_ref_point(Y)

    # Exact HV trace, computed incrementally with pymoo's compiled-C WFG kernel.
    # The old code rebuilt BoTorch's box decomposition from scratch on every one of
    # the `iters` iterations, which cost ~340 s/seed on RE61 (m=6) and starved tables
    # T3/T4 out of the campaign. Same numbers (agreement ~1e-16), orders faster.
    for hv in hypervolume_trace(Y[:iters], ref_point):
        res.record(hv)
    res.set_history(X, Y, 0)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--iters", type=int, default=200)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(make_problem(args.problem, args), args.iters, args.seed)
    finalize(res, args)


if __name__ == "__main__":
    main()
