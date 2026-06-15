"""Constrained qNEHVI for constrained multi-objective problems.

Models every objective and every inequality constraint with an independent GP and
maximizes q-Log Noisy Expected Hypervolume Improvement with feasibility
constraints: the hypervolume improvement is only credited for points predicted to
satisfy all constraints (``c <= 0``). Reference point is in BoCoDe's maximization
frame.

Run::

    python -m algorithms.multi_obj_constrained.constrained_qnehvi --problem WeldedBeam --init 12 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
BoTorch constrained multi-objective tutorial: https://botorch.org/docs/tutorials/constrained_multi_objective_bo
"""

from __future__ import annotations

import argparse

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from gpytorch.mlls import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine

import bocode

from .._bo_utils import DTYPE, MultiObjectiveProblem, Result, set_seed


def _fit(train_X, train_Y, train_C):
    """Fit GPs for the objectives followed by the constraints (one ModelListGP)."""
    dim = train_X.shape[-1]
    outs = [train_Y[:, i : i + 1] for i in range(train_Y.shape[-1])]
    outs += [train_C[:, i : i + 1] for i in range(train_C.shape[-1])]
    models = [
        SingleTaskGP(
            train_X, o,
            input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1),
        )
        for o in outs
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(problem, n_init: int = 12, iters: int = 50, seed: int = 0) -> Result:
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    m, nc = obj.num_objectives, obj.num_constraints
    assert nc > 0, "constrained_qnehvi requires a constrained problem"
    res = Result("constrained_qnehvi", type(problem).__name__, seed)

    train_X = SobolEngine(obj.dim, scramble=True, seed=seed).draw(n_init).to(DTYPE)
    train_Y, train_C = obj.evaluate_raw(train_X)
    ref_point = obj.infer_ref_point(train_Y)

    # Constraints are model outputs m..m+nc-1; BoTorch convention is feasible when
    # the constraint callable returns <= 0, matching BoCoDe's c <= 0.
    constraint_callables = [
        (lambda Z, i=m + j: Z[..., i]) for j in range(nc)
    ]

    def feasible_hv(Y, C):
        feas = (C <= 0).all(dim=1)
        if not feas.any():
            return 0.0
        return DominatedPartitioning(
            ref_point=ref_point, Y=Y[feas]
        ).compute_hypervolume().item()

    for _ in range(n_init):
        res.log(feasible_hv(train_Y, train_C))

    for _ in range(iters):
        model = _fit(train_X, train_Y, train_C)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
            # select the objective outputs (0..m-1) from the combined model;
            # outputs m..m+nc-1 are the constraints used by the callables.
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(m))),
            constraints=constraint_callables,
        )
        candidate, _ = optimize_acqf(
            acqf, bounds=obj.bounds, q=1, num_restarts=10, raw_samples=256
        )
        y, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        train_C = torch.cat([train_C, c], dim=0)
        res.log(feasible_hv(train_Y, train_C))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=12)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = optimize_problem(bocode.get_problem(args.problem)(), args.init, args.iters, args.seed)
    print(f"{res.algorithm} on {res.problem}: final feasible hypervolume={res.best:.6g} after {len(res.best_history)} evals")


if __name__ == "__main__":
    main()
