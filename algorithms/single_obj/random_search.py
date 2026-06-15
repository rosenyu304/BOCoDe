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
from torch.quasirandom import SobolEngine

import bocode

from .._bo_utils import DTYPE, DatasetObjective, ProblemObjective, Result, set_seed


def optimize_problem(problem, iters: int = 100, seed: int = 0) -> Result:
    """Random (Sobol) search over the unit cube for a continuous problem."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result("random_search", type(problem).__name__, seed)

    sobol = SobolEngine(dimension=obj.dim, scramble=True, seed=seed)
    X = sobol.draw(iters).to(DTYPE)
    Y = obj(X)

    best = -float("inf")
    for i in range(iters):
        best = max(best, Y[i].item())
        res.log(best)
    return res


def optimize_dataset(dataset_problem, iters: int = 100, seed: int = 0) -> Result:
    """Random selection (without replacement) from a discrete candidate pool."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result("random_search", type(dataset_problem).__name__, seed)

    iters = min(iters, data.n_candidates)
    perm = torch.randperm(data.n_candidates)[:iters]

    best = -float("inf")
    for idx in perm:
        best = max(best, data.select(idx).item())
        res.log(best)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", help="continuous problem name, e.g. Car")
    group.add_argument("--dataset", help="dataset problem name, e.g. AgNP")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.problem:
        res = optimize_problem(bocode.get_problem(args.problem)(), args.iters, args.seed)
    else:
        res = optimize_dataset(bocode.get_problem(args.dataset)(), args.iters, args.seed)
    print(f"{res.algorithm} on {res.problem}: best={res.best:.6g} after {len(res.best_history)} evals")


if __name__ == "__main__":
    main()
