"""Constrained qParEGO for constrained multi-objective problems.

ParEGO with feasibility: each step draws a random augmented-Chebyshev
scalarization of the (standardized) objectives and maximizes q-Log Noisy Expected
Improvement on that scalarization, restricted to points predicted to satisfy every
constraint (``c <= 0``). Objectives and constraints are each modeled with an
independent GP. Cheaper than constrained qNEHVI.

Run::

    python -m algorithms.multi_obj_constrained.constrained_qparego --problem WeldedBeam --init 12 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
J. Knowles. ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. IEEE TEVC 2006.
BoTorch constrained multi-objective tutorial: https://botorch.org/docs/tutorials/constrained_multi_objective_bo
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
    initial_design,
    make_problem,
    set_seed,
)


def _fit(train_X, train_Y, train_C):
    dim = train_X.shape[-1]
    outs = [train_Y[:, i : i + 1] for i in range(train_Y.shape[-1])]
    outs += [train_C[:, i : i + 1] for i in range(train_C.shape[-1])]
    models = [
        SingleTaskGP(
            train_X,
            o,
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        for o in outs
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(
    problem, n_init: int = 12, iters: int = 50, seed: int = 0
) -> Result:
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    m, nc = obj.num_objectives, obj.num_constraints
    assert nc > 0, "constrained_qparego requires a constrained problem"
    res = Result(
        "constrained_qparego",
        type(problem).__name__,
        seed,
        acquisition_function="qLogNEI",
    )

    train_X = initial_design(n_init, obj.dim, seed)
    train_Y, train_C = obj.evaluate_raw(train_X)
    ref_point = obj.infer_ref_point(train_Y)

    # Constraint callables index the constraint model outputs (m..m+nc-1); feasible
    # when the value is <= 0, matching BoCoDe's convention.
    constraint_callables = [(lambda Z, i=m + j: Z[..., i]) for j in range(nc)]

    def feasible_hv(Y, C):
        feas = (C <= 0).all(dim=1)
        if not feas.any():
            return 0.0
        return (
            DominatedPartitioning(ref_point=ref_point, Y=Y[feas])
            .compute_hypervolume()
            .item()
        )

    res.start(feasible_hv(train_Y, train_C))

    for _ in range(iters):
        model = _fit(train_X, train_Y, train_C)
        weights = sample_simplex(m, dtype=DTYPE).squeeze()
        scalarization = get_chebyshev_scalarization(weights=weights, Y=train_Y)

        # Apply the scalarization only to the objective outputs (0..m-1).
        def scalarized_objective(Z, X=None, s=scalarization, mm=m):
            return s(Z[..., :mm])

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_X,
            objective=GenericMCObjective(scalarized_objective),
            constraints=constraint_callables,
            sampler=sampler,
            prune_baseline=True,
        )
        candidate, _ = optimize_acqf(
            acqf, bounds=obj.bounds, q=1, num_restarts=10, raw_samples=256
        )
        y, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        train_C = torch.cat([train_C, c], dim=0)
        res.record(feasible_hv(train_Y, train_C))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=12)
    parser.add_argument("--iters", type=int, default=50)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args), args.init, args.iters, args.seed
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
