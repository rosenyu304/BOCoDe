"""Random search baseline for constrained multi-objective problems.

Draws Latin-hypercube points and records the dominated hypervolume of the running
*feasible* Pareto set (in BoCoDe's maximization frame): infeasible points (any
``c > 0``) are dropped before the Pareto partition, exactly as in
``constrained_qnehvi`` / ``constrained_qparego``. Zero until the first feasible
point is found. The mandatory baseline for the constrained multi-objective category.

Run::

    python -m algorithms.multi_obj_constrained.random_search --problem WeldedBeam --iters 200

Sources:
M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
BoTorch constrained multi-objective tutorial: https://botorch.org/docs/tutorials/constrained_multi_objective_bo
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
    assert obj.num_constraints > 0, "expected a constrained problem"
    res = Result(
        "random_search", type(problem).__name__, seed, acquisition_function="none"
    )

    X = initial_design(iters, obj.dim, seed)
    Y, C = obj.evaluate_raw(X)
    ref_point = obj.hv_ref_point(Y)
    feasible = (C <= 0).all(dim=1)

    # Exact FEASIBLE HV trace, computed incrementally with pymoo's compiled-C kernel
    # (infeasible points never enter the front, so the trace stays 0 until the first
    # feasible point). Replaces an O(iters) rebuild of BoTorch's box decomposition.
    for hv in hypervolume_trace(Y[:iters], ref_point, feasible=feasible[:iters]):
        res.record(hv)
    res.set_history(X, Y, 0, c=C)
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
