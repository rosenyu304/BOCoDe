"""Random search baseline for single-objective constrained problems.

Draws Sobol points and tracks the best *feasible* objective seen. If no feasible
point has been found yet, it tracks the least-infeasible point's penalized
objective, so the best-history is always defined.

Run::

    python -m algorithms.single_obj_constrained.random_search --problem PressureVessel --iters 200

Sources:
M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse

import torch

from .._bo_utils import (
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    initial_design,  # noqa: E501
    make_problem,
    set_seed,
)


def optimize_problem(problem, iters: int = 200, seed: int = 0) -> Result:
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result(
        "random_search", type(problem).__name__, seed, acquisition_function="none"
    )

    X = initial_design(iters, obj.dim, seed)
    values, constraints = obj.evaluate_raw(X)
    feasible = (
        torch.ones(iters, dtype=torch.bool)
        if constraints is None or constraints.numel() == 0
        else (constraints <= 0).all(dim=1)
    )

    best = -float("inf")
    for i in range(iters):
        if feasible[i]:
            best = max(best, values[i].item())
        res.record(best)
    res.set_history(X, values, 0, c=constraints)
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
