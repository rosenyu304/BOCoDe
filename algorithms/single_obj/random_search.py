"""Random search baseline for single-objective problems.

The mandatory baseline for every problem category: draw points uniformly at
random (Sobol for low discrepancy) and keep the best. Provides both a
problem-optimization loop (continuous, over the unit cube) and a
dataset-optimization loop (discrete, over a candidate pool).

Run::

    python -m algorithms.single_obj.random_search --problem Car --iters 100
    python -m algorithms.single_obj.random_search --dataset AgNP --iters 50

Sources:
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse

import torch

from .._bo_utils import (
    DatasetObjective,
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    initial_design,  # noqa: E501
    make_problem,
    set_seed,
)


def optimize_problem(problem, iters: int = 100, seed: int = 0) -> Result:
    """Random (Sobol) search over the unit cube for a continuous problem."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result(
        "random_search", type(problem).__name__, seed, acquisition_function="none"
    )

    X = initial_design(iters, obj.dim, seed)
    Y = obj(X)

    best = -float("inf")
    for i in range(iters):
        best = max(best, Y[i].item())
        res.record(best)
    return res


def optimize_dataset(dataset_problem, iters: int = 100, seed: int = 0) -> Result:
    """Random selection (without replacement) from a discrete candidate pool."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result(
        "random_search",
        type(dataset_problem).__name__,
        seed,
        acquisition_function="none",
    )

    iters = min(iters, data.n_candidates)
    perm = torch.randperm(data.n_candidates)[:iters]

    best = -float("inf")
    for idx in perm:
        best = max(best, data.select(idx).item())
        res.record(best)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", help="continuous problem name, e.g. Car")
    group.add_argument("--dataset", help="dataset problem name, e.g. AgNP")
    parser.add_argument("--iters", type=int, default=100)
    add_common_args(parser)
    args = parser.parse_args()

    if args.problem:
        res = optimize_problem(make_problem(args.problem, args), args.iters, args.seed)
    else:
        res = optimize_dataset(make_problem(args.dataset, args), args.iters, args.seed)
    finalize(res, args)


if __name__ == "__main__":
    main()
