"""Standard GP BO for single-objective problems (Xu et al., 2025).

"Standard Gaussian Process is All You Need for High-Dimensional Bayesian
Optimization" argues that a *standard* GP — with a sensible, dimension-aware
lengthscale prior — matched with LogEI is competitive with specialized
high-dimensional BO methods. This script implements that baseline: a
``SingleTaskGP`` whose RBF lengthscale prior is scaled so the prior mean
lengthscale grows with ``sqrt(dim)`` (keeping the effective signal variance
roughly dimension-invariant), optimized with LogExpectedImprovement.

The difference from ``vanilla_highdim_bo`` is the modeling choice (the dimension-scaled
prior) rather than the acquisition-optimization tricks.

Run::

    python -m algorithms.single_obj.standard_gp --problem Rover --init 20 --iters 100
    python -m algorithms.single_obj.standard_gp --dataset P3HT --init 10 --iters 40

Sources:
Z. Xu, H. Wang, J. M. Phillips, and S. Zhe. Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization. ICLR 2025. https://openreview.net/forum?id=kX8h23UG6v
S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. Unexpected Improvements to Expected Improvement for Bayesian Optimization. NeurIPS 2023. https://arxiv.org/abs/2310.20708
M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse
import math

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

import bocode

from .._bo_utils import (
    DTYPE,
    DatasetObjective,
    ProblemObjective,
    Result,
    initial_design,  # noqa: E501
    set_seed,
)


def _standard_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """A standard GP with a dimension-scaled Matern-5/2 lengthscale prior.

    The lengthscale prior mean is scaled by ``sqrt(dim)`` so that, after input
    normalization to the unit cube, the prior expects smoother functions in higher
    dimensions — the modeling choice highlighted by Xu et al. (2025).
    """
    dim = train_X.shape[-1]
    ls_scale = math.sqrt(dim)
    kernel = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_prior=GammaPrior(3.0, 6.0 / ls_scale),
        ),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    model = SingleTaskGP(
        train_X=train_X.to(DTYPE),
        train_Y=train_Y.to(DTYPE),
        covar_module=kernel,
        input_transform=Normalize(d=dim),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(problem, n_init: int = 20, iters: int = 100, seed: int = 0) -> Result:
    """Continuous standard-GP BO over the unit cube."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result("standard_gp", type(problem).__name__, seed)

    train_X = initial_design(n_init, obj.dim, seed)
    train_Y = obj(train_X)
    best = train_Y.max().item()
    for _ in range(n_init):
        res.log(best)

    for _ in range(iters):
        model = _standard_gp(train_X, train_Y)
        acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
        candidate, _ = optimize_acqf(
            acqf, bounds=obj.bounds, q=1, num_restarts=10, raw_samples=512
        )
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.log(best)
    return res


def optimize_dataset(dataset_problem, n_init: int = 10, iters: int = 50, seed: int = 0) -> Result:
    """Discrete standard-GP BO over a candidate pool."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result("standard_gp", type(dataset_problem).__name__, seed)

    perm = torch.randperm(data.n_candidates)
    observed = perm[:n_init].tolist()
    best = data.select(torch.tensor(observed)).max().item()
    for _ in range(n_init):
        res.log(best)

    for _ in range(iters):
        obs = torch.tensor(observed)
        model = _standard_gp(data.X[obs], data.Y[obs])
        mask = torch.ones(data.n_candidates, dtype=torch.bool)
        mask[obs] = False
        pool_idx = mask.nonzero(as_tuple=True)[0]
        if pool_idx.numel() == 0:
            break
        acqf = LogExpectedImprovement(model=model, best_f=data.Y[obs].max())
        with torch.no_grad():
            scores = acqf(data.X[pool_idx].unsqueeze(1))
        choice = pool_idx[scores.argmax()].item()
        observed.append(choice)
        best = max(best, data.select(torch.tensor(choice)).item())
        res.log(best)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem")
    group.add_argument("--dataset")
    parser.add_argument("--init", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.problem:
        res = optimize_problem(bocode.get_problem(args.problem)(), args.init, args.iters, args.seed)
    else:
        res = optimize_dataset(bocode.get_problem(args.dataset)(), args.init, args.iters, args.seed)
    print(f"{res.algorithm} on {res.problem}: best={res.best:.6g} after {len(res.best_history)} evals")


if __name__ == "__main__":
    main()
