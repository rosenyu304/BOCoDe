"""Vanilla high-dimensional Bayesian optimization for single-objective problems.

A faithful reimplementation of the "Vanilla BO" method from Hvarfner et al., 2024
("Vanilla Bayesian Optimization Performs Great in High Dimensions"). The paper's
contribution is a **dimension-scaled log-normal lengthscale prior** on an ordinary
ARD RBF GP -- nothing else about the GP changes -- plus the acquisition-optimization
settings that make a plain acquisition function work in high dimensions:

* ``lengthscale ~ LogNormal(sqrt(2) + log(D) / 2, sqrt(3))`` (Sec. 4; reference
  ``benchmarking/gp_priors.py::get_covar_module`` with ``configs/model/default.yaml``:
  ``ls: loc 1.4, scale 1.73205`` and ``DIM_SCALING["default"] = (0.5, 0)``, i.e.
  ``loc + 0.5 * log(D)``). The prior grows the lengthscales like ``sqrt(D)``, which
  counteracts the vanishing signal-to-noise of a fixed prior in high dimensions.
  It is constructed **explicitly** here: BoTorch >= 0.12 happens to use the same
  prior as its ``SingleTaskGP`` default, but relying on that default would let the
  method silently degrade to the Gamma(3, 6) baseline the paper argues against.
* noise ``~ LogNormal(-4, 1)`` with ``GreaterThan(1e-4)`` (same config), no outputscale.
* **qLogNoisyExpectedImprovement** with ``prune_baseline=True`` (``configs/acq/qlognei.yaml``).
* Acquisition optimized with ``raw_samples=512``, ``num_restarts=4``, and
  ``sample_around_best=True`` with ``sample_around_best_sigma=0.1``, ``maxiter=300``,
  ``batch_limit=64`` (``configs/acq_opt/highdim.yaml``). BoTorch's default sigma is
  1e-3, at which the perturbations are indistinguishable from the incumbent and the
  feature the paper relies on is effectively disabled.

``n_init`` defaults to BoCoDe's dimension-scaled :func:`default_n_init` so every
algorithm in the suite gets the same initial-design budget; the paper's own rule is
``ceil(3 * sqrt(D))`` (``configs/conf.yaml``: ``init: sqrt``, ``init_factor: 3``;
``main.py:54-56``), which ``--init`` can reproduce.

Run::

    python -m algorithms.single_obj.vanilla_highdim_bo --problem Car --init 10 --iters 50
    python -m algorithms.single_obj.vanilla_highdim_bo --dataset AgNP --init 10 --iters 40

Sources:
C. Hvarfner, E. Hellsten, and L. Nardi. Vanilla Bayesian Optimization Performs Great in High Dimensions. ICML 2024. https://arxiv.org/abs/2402.02229
Official implementation: https://github.com/hvarfner/vanilla_bo_in_highdim
S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. Unexpected Improvements to Expected Improvement for Bayesian Optimization. NeurIPS 2023. https://arxiv.org/abs/2310.20708
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from .._bo_utils import (
    DTYPE,
    DatasetObjective,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    gp_stats,
    initial_design,  # noqa: E501
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)

# Acquisition-optimization settings from the reference (configs/acq_opt/highdim.yaml).
RAW_SAMPLES = 512
NUM_RESTARTS = 4
ACQ_OPT_OPTIONS = {
    "nonnegative": False,
    "sample_around_best": True,
    "sample_around_best_sigma": 0.1,
    "maxiter": 300,
    "batch_limit": 64,
}
# GP hyperparameter constraints (configs: gp_constraints default 1e-4).
MIN_LENGTHSCALE = 1e-4
MIN_NOISE = 1e-4


def _vanilla_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """ARD RBF GP with the paper's dimension-scaled LogNormal lengthscale prior."""
    dim = train_X.shape[-1]
    ls_prior = LogNormalPrior(
        loc=math.sqrt(2.0) + math.log(dim) / 2.0, scale=math.sqrt(3.0)
    )
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    covar_module = RBFKernel(
        ard_num_dims=dim,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=GreaterThan(
            MIN_LENGTHSCALE, transform=None, initial_value=ls_prior.mode
        ),
    )
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_NOISE, transform=None, initial_value=noise_prior.mode
        ),
    )
    model = SingleTaskGP(
        train_X=train_X.to(DTYPE),
        train_Y=train_Y.to(DTYPE),
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=Normalize(d=dim),
        outcome_transform=Standardize(m=1),
    ).to(train_X)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Continuous Vanilla BO over the unit cube for a problem."""
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "vanilla_highdim_bo",
        type(problem).__name__,
        seed,
        acquisition_function="qLogNEI",
    )

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
        best = train_Y.max().item()
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        start_it = 0

    bounds_dev = obj.bounds.to(dev)
    for it in range(start_it, iters):
        model = _vanilla_gp(train_X.to(dev), train_Y.to(dev))
        acqf = qLogNoisyExpectedImprovement(
            model=model, X_baseline=train_X.to(dev), prune_baseline=True
        )
        candidate, acq_value = optimize_acqf(
            acqf,
            bounds=bounds_dev,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options=ACQ_OPT_OPTIONS,
        )
        mean, var = gp_stats(model, candidate)
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(best, mean=mean, variance=var, acq_value=acq_value.item())
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(checkpoint, train_X, train_Y, res, it + 1)
    res.set_history(train_X, train_Y, n_init)
    return res


def optimize_dataset(
    dataset_problem, n_init: int = 10, iters: int = 50, seed: int = 0
) -> Result:
    """Discrete Vanilla BO: maximize qLogNEI over the unobserved candidate pool."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result(
        "vanilla_highdim_bo",
        type(dataset_problem).__name__,
        seed,
        acquisition_function="qLogNEI",
    )

    perm = torch.randperm(data.n_candidates)
    observed = perm[:n_init].tolist()
    best = data.select(torch.tensor(observed)).max().item()
    res.start(best)

    for _ in range(iters):
        obs = torch.tensor(observed)
        model = _vanilla_gp(data.X[obs], data.Y[obs])
        mask = torch.ones(data.n_candidates, dtype=torch.bool)
        mask[obs] = False
        pool_idx = mask.nonzero(as_tuple=True)[0]
        if pool_idx.numel() == 0:
            break

        acqf = qLogNoisyExpectedImprovement(
            model=model, X_baseline=data.X[obs], prune_baseline=True
        )
        with torch.no_grad():
            scores = acqf(data.X[pool_idx].unsqueeze(1))  # (m, 1, d) -> (m,)
        choice = pool_idx[scores.argmax()].item()
        mean, var = gp_stats(model, data.X[choice].unsqueeze(0))
        observed.append(choice)
        best = max(best, data.select(torch.tensor(choice)).item())
        res.record(best, mean=mean, variance=var, acq_value=scores.max().item())
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", help="continuous problem name, e.g. Car")
    group.add_argument("--dataset", help="dataset problem name, e.g. AgNP")
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--checkpoint", default=None, help="resumable checkpoint .npz path"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()

    if args.problem:
        res = optimize_problem(
            make_problem(args.problem, args),
            args.init,
            args.iters,
            args.seed,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    else:
        res = optimize_dataset(
            make_problem(args.dataset, args),
            args.init if args.init is not None else 10,
            args.iters,
            args.seed,
        )
    finalize(res, args)


if __name__ == "__main__":
    main()
