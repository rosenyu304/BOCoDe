"""Vanilla Bayesian optimization for single-objective problems.

A faithful reimplementation of the "Vanilla BO" baseline from Hvarfner et al.,
2024 ("Vanilla Bayesian Optimization Performs Great in High Dimensions"), using
the settings the authors emphasize as load-bearing — getting these wrong makes
the baseline silently underperform (see Research_Plan.md §10):

* **LogExpectedImprovement** (Ament et al., 2023), not the numerically unstable
  ``ExpectedImprovement``.
* Acquisition optimized with ``raw_samples=512`` and ``num_restarts=4`` (L-BFGS
  from the best 4 of the raw samples), not a handful of restarts.
* ``sample_around_best=True``: half the raw samples are drawn from a Gaussian
  around the best observation so far, perturbing only a subset of dimensions
  (BoTorch's ``optimize_acqf`` option). This matters a lot in high dimensions.
* ``best_f`` for the acquisition is taken in the **same (standardized) space** as
  the GP's training targets — here the GP applies a ``Standardize`` outcome
  transform internally, so ``best_f`` is the max of the raw ``train_Y`` and the
  transform is applied consistently by BoTorch's acquisition.

Run::

    python -m algorithms.single_obj.vanilla_bo --problem Car --init 10 --iters 50
    python -m algorithms.single_obj.vanilla_bo --dataset AgNP --init 10 --iters 40

Sources:
C. Hvarfner, E. Hellsten, and L. Nardi. Vanilla Bayesian Optimization Performs Great in High Dimensions. ICML 2024. https://arxiv.org/abs/2402.02229
S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. Unexpected Improvements to Expected Improvement for Bayesian Optimization. NeurIPS 2023. https://arxiv.org/abs/2310.20708
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine

import bocode

from .._bo_utils import (
    DTYPE,
    DatasetObjective,
    ProblemObjective,
    Result,
    fit_gp,
    set_seed,
)

# Acquisition-optimization settings from the original Vanilla BO paper.
RAW_SAMPLES = 512
NUM_RESTARTS = 4


def optimize_problem(
    problem, n_init: int = 10, iters: int = 50, seed: int = 0
) -> Result:
    """Continuous Vanilla BO over the unit cube for a problem."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result("vanilla_bo", type(problem).__name__, seed)

    train_X = SobolEngine(obj.dim, scramble=True, seed=seed).draw(n_init).to(DTYPE)
    train_Y = obj(train_X)

    best = train_Y.max().item()
    for _ in range(n_init):
        res.log(best)

    for _ in range(iters):
        model = fit_gp(train_X, train_Y)
        acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
        candidate, _ = optimize_acqf(
            acqf,
            bounds=obj.bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"sample_around_best": True},
        )
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.log(best)
    return res


def optimize_dataset(
    dataset_problem, n_init: int = 10, iters: int = 50, seed: int = 0
) -> Result:
    """Discrete Vanilla BO: maximize LogEI over the unobserved candidate pool."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result("vanilla_bo", type(dataset_problem).__name__, seed)

    perm = torch.randperm(data.n_candidates)
    observed = perm[:n_init].tolist()
    best = data.select(torch.tensor(observed)).max().item()
    for _ in range(n_init):
        res.log(best)

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
            scores = acqf(data.X[pool_idx].unsqueeze(1))  # (m, 1, d) -> (m,)
        choice = pool_idx[scores.argmax()].item()
        observed.append(choice)
        best = max(best, data.select(torch.tensor(choice)).item())
        res.log(best)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", help="continuous problem name, e.g. Car")
    group.add_argument("--dataset", help="dataset problem name, e.g. AgNP")
    parser.add_argument("--init", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.problem:
        res = optimize_problem(
            bocode.get_problem(args.problem)(), args.init, args.iters, args.seed
        )
    else:
        res = optimize_dataset(
            bocode.get_problem(args.dataset)(), args.init, args.iters, args.seed
        )
    print(f"{res.algorithm} on {res.problem}: best={res.best:.6g} after {len(res.best_history)} evals")


if __name__ == "__main__":
    main()
