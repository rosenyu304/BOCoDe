"""Constrained Expected Improvement for single-objective constrained problems.

Models the objective and each inequality constraint with independent GPs (a
``ModelListGP``) and maximizes the log Constrained Expected Improvement: EI on the
objective weighted by the probability that every constraint is satisfied
(``c <= 0``). This is the analytic constrained-BO baseline from the BoTorch
closed-loop tutorial, using the numerically stable log form.

Run::

    python -m algorithms.single_obj_constrained.constrained_ei --problem PressureVessel --init 10 --iters 50

Sources:
M. Gelbart, J. Snoek, and R. P. Adams. Bayesian Optimization with Unknown Constraints. UAI 2014. https://arxiv.org/abs/1403.5607
S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. Unexpected Improvements to Expected Improvement for Bayesian Optimization. NeurIPS 2023. https://arxiv.org/abs/2310.20708
BoTorch closed-loop tutorial: https://botorch.org/docs/tutorials/closed_loop_botorch_only
"""

from __future__ import annotations

import argparse

import torch
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood

import bocode

from .._bo_utils import (
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    initial_design,  # noqa: E501
    set_seed,
)


def _fit_model_list(train_X, train_obj, train_con):
    """Fit one GP for the objective and one per constraint."""
    dim = train_X.shape[-1]
    models = [
        SingleTaskGP(
            train_X,
            train_obj,
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
    ]
    for i in range(train_con.shape[-1]):
        models.append(
            SingleTaskGP(
                train_X,
                train_con[:, i : i + 1],
                input_transform=Normalize(d=dim),
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(
    problem, n_init: int = 10, iters: int = 50, seed: int = 0
) -> Result:
    set_seed(seed)
    obj = ProblemObjective(problem)
    assert obj.num_constraints > 0, "constrained_ei requires a constrained problem"
    res = Result(
        "constrained_ei", type(problem).__name__, seed, acquisition_function="LogCEI"
    )

    train_X = initial_design(n_init, obj.dim, seed)
    train_obj, train_con = obj.evaluate_raw(train_X)

    def best_feasible(obj_v, con_v):
        feas = (con_v <= 0).all(dim=1)
        return obj_v[feas].max().item() if feas.any() else -float("inf")

    best = best_feasible(train_obj, train_con)
    res.start(best)

    # BoTorch's constraint dict maps each constraint model's output index to
    # (lower, upper) feasible bounds; here constraints are feasible when <= 0.
    constraint_dict = {i + 1: (None, 0.0) for i in range(obj.num_constraints)}

    for _ in range(iters):
        model = _fit_model_list(train_X, train_obj, train_con)
        # best_f must be in the model's standardized objective space; use a safe
        # value (0.0) when no feasible point exists yet so EI still explores.
        feas = (train_con <= 0).all(dim=1)
        best_f = train_obj[feas].max() if feas.any() else train_obj.max()
        acqf = LogConstrainedExpectedImprovement(
            model=model,
            best_f=best_f,
            objective_index=0,
            constraints=constraint_dict,
        )
        candidate, _ = optimize_acqf(
            acqf, bounds=obj.bounds, q=1, num_restarts=10, raw_samples=512
        )
        o, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_obj = torch.cat([train_obj, o], dim=0)
        train_con = torch.cat([train_con, c], dim=0)
        best = max(best, best_feasible(o, c))
        res.record(best)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        bocode.get_problem(args.problem)(), args.init, args.iters, args.seed
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
