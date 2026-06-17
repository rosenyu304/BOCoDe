"""SingleTaskGP Bayesian optimization (vanilla BoTorch getting-started setting).

The plain BoTorch baseline from https://botorch.org/docs/getting_started : fit a
``SingleTaskGP`` with input normalization and output standardization, then propose
the next point with LogExpectedImprovement optimized by ``optimize_acqf`` using
BoTorch's default settings. No high-dimensional tricks — the reference "just use a
SingleTaskGP" loop. Use ``vanilla_highdim_bo`` for the high-dimensional variant.

Run::

    python -m algorithms.single_obj.single_task_gp --problem Branin --init 10 --iters 40
    python -m algorithms.single_obj.single_task_gp --dataset AgNP --init 10 --iters 40

Sources:
BoTorch getting started: https://botorch.org/docs/getting_started
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from .._bo_utils import (
    DatasetObjective,
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    fit_gp,
    gp_stats,
    initial_design,
    make_problem,
    set_seed,
)


def optimize_problem(
    problem, n_init: int = 10, iters: int = 40, seed: int = 0
) -> Result:
    """Continuous SingleTaskGP + LogEI BO over the unit cube."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result(
        "single_task_gp", type(problem).__name__, seed, acquisition_function="LogEI"
    )

    train_X = initial_design(n_init, obj.dim, seed)
    train_Y = obj(train_X)
    best = train_Y.max().item()
    res.start(best)

    for _ in range(iters):
        model = fit_gp(train_X, train_Y)  # SingleTaskGP with Normalize + Standardize
        acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
        candidate, acq_value = optimize_acqf(
            acqf, bounds=obj.bounds, q=1, num_restarts=10, raw_samples=512
        )
        mean, var = gp_stats(model, candidate)
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(best, mean=mean, variance=var, acq_value=acq_value.item())
    return res


def optimize_dataset(
    dataset_problem, n_init: int = 10, iters: int = 40, seed: int = 0
) -> Result:
    """Discrete SingleTaskGP + LogEI BO over a candidate pool."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result(
        "single_task_gp",
        type(dataset_problem).__name__,
        seed,
        acquisition_function="LogEI",
    )

    perm = torch.randperm(data.n_candidates)
    observed = perm[:n_init].tolist()
    best = data.select(torch.tensor(observed)).max().item()
    res.start(best)

    for _ in range(iters):
        obs = torch.tensor(observed)
        model = fit_gp(data.X[obs], data.Y[obs])
        mask = torch.ones(data.n_candidates, dtype=torch.bool)
        mask[obs] = False
        pool_idx = mask.nonzero(as_tuple=True)[0]
        if pool_idx.numel() == 0:
            break
        acqf = LogExpectedImprovement(model=model, best_f=data.Y[obs].max())
        with torch.no_grad():
            scores = acqf(data.X[pool_idx].unsqueeze(1))
        choice = pool_idx[scores.argmax()].item()
        mean, var = gp_stats(model, data.X[choice].unsqueeze(0))
        observed.append(choice)
        best = max(best, data.select(torch.tensor(choice)).item())
        res.record(best, mean=mean, variance=var, acq_value=scores.max().item())
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem")
    group.add_argument("--dataset")
    parser.add_argument("--init", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    add_common_args(parser)
    args = parser.parse_args()

    if args.problem:
        res = optimize_problem(
            make_problem(args.problem, args), args.init, args.iters, args.seed
        )
    else:
        res = optimize_dataset(
            make_problem(args.dataset, args), args.init, args.iters, args.seed
        )
    finalize(res, args)


if __name__ == "__main__":
    main()
