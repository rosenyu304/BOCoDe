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

from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

import bocode

from .._bo_utils import (
    MultiObjectiveProblem,
    Result,
    initial_design,  # noqa: E501
    set_seed,
)


def optimize_problem(problem, iters: int = 200, seed: int = 0) -> Result:
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    res = Result("random_search", type(problem).__name__, seed)

    X = initial_design(iters, obj.dim, seed)
    Y, _ = obj.evaluate_raw(X)
    ref_point = obj.infer_ref_point(Y)

    for i in range(1, iters + 1):
        part = DominatedPartitioning(ref_point=ref_point, Y=Y[:i])
        res.log(part.compute_hypervolume().item())
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = optimize_problem(bocode.get_problem(args.problem)(), args.iters, args.seed)
    print(f"{res.algorithm} on {res.problem}: final hypervolume={res.best:.6g} after {len(res.best_history)} evals")


if __name__ == "__main__":
    main()
