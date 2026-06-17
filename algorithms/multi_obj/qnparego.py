"""qNParEGO for multi-objective problems.

ParEGO with noisy expected improvement: at each step a random augmented-Chebyshev
scalarization of the (standardized) objectives is drawn, and the point maximizing
q-Log Noisy Expected Improvement on that scalarization is proposed. Cheaper than
qNEHVI and a strong multi-objective baseline.

Run::

    python -m algorithms.multi_obj.qnparego --problem Penicillin --init 10 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
J. Knowles. ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. IEEE TEVC 2006.
BoTorch multi-objective tutorial: https://botorch.org/docs/tutorials/multi_objective_bo
"""

from __future__ import annotations

import argparse

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from gpytorch.mlls import SumMarginalLogLikelihood

from .._bo_utils import (
    DTYPE,
    MultiObjectiveProblem,
    Result,
    add_common_args,
    finalize,
    initial_design,  # noqa: E501
    make_problem,
    set_seed,
)


def _fit(train_X, train_Y):
    dim = train_X.shape[-1]
    models = [
        SingleTaskGP(
            train_X,
            train_Y[:, i : i + 1],
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        for i in range(train_Y.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(
    problem, n_init: int = 10, iters: int = 50, seed: int = 0
) -> Result:
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    m = obj.num_objectives
    res = Result(
        "qnparego", type(problem).__name__, seed, acquisition_function="qLogNEI"
    )

    train_X = initial_design(n_init, obj.dim, seed)
    train_Y, _ = obj.evaluate_raw(train_X)
    ref_point = obj.infer_ref_point(train_Y)

    def hv(Y):
        return (
            DominatedPartitioning(ref_point=ref_point, Y=Y).compute_hypervolume().item()
        )

    res.start(hv(train_Y))

    for _ in range(iters):
        model = _fit(train_X, train_Y)
        weights = sample_simplex(m, dtype=DTYPE).squeeze()
        scalarization = get_chebyshev_scalarization(weights=weights, Y=train_Y)
        mc_objective = GenericMCObjective(lambda Z, X=None, s=scalarization: s(Z))
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_X,
            objective=mc_objective,
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
    res = optimize_problem(
        make_problem(args.problem, args), args.init, args.iters, args.seed
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
