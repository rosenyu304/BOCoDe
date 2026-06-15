"""qNEHVI for multi-objective problems.

q-Noisy Expected Hypervolume Improvement (Daulton et al., 2021), using the
numerically stable log form. Each objective is modeled with an independent GP; the
acquisition proposes the point that maximally expands the dominated hypervolume
above a reference point (in BoCoDe's maximization frame).

Run::

    python -m algorithms.multi_obj.qnehvi --problem Penicillin --init 10 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
BoTorch multi-objective tutorial: https://botorch.org/docs/tutorials/multi_objective_bo
"""

from __future__ import annotations

import argparse

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from gpytorch.mlls import SumMarginalLogLikelihood

import bocode

from .._bo_utils import (
    MultiObjectiveProblem,
    Result,
    add_common_args,
    finalize,
    initial_design,  # noqa: E501
    set_seed,
)


def _fit(train_X, train_Y):
    dim = train_X.shape[-1]
    models = [
        SingleTaskGP(
            train_X, train_Y[:, i : i + 1],
            input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1),
        )
        for i in range(train_Y.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(problem, n_init: int = 10, iters: int = 50, seed: int = 0) -> Result:
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    res = Result("qnehvi", type(problem).__name__, seed, acquisition_function="qLogNEHVI")

    train_X = initial_design(n_init, obj.dim, seed)
    train_Y, _ = obj.evaluate_raw(train_X)
    ref_point = obj.infer_ref_point(train_Y)

    def hv(Y):
        return DominatedPartitioning(ref_point=ref_point, Y=Y).compute_hypervolume().item()

    res.start(hv(train_Y))

    for _ in range(iters):
        model = _fit(train_X, train_Y)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
        )
        candidate, _ = optimize_acqf(
            acqf, bounds=obj.bounds, q=1, num_restarts=10, raw_samples=256
        )
        y, _ = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        res.record(hv(train_Y))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(bocode.get_problem(args.problem)(), args.init, args.iters, args.seed)
    finalize(res, args)


if __name__ == "__main__":
    main()
