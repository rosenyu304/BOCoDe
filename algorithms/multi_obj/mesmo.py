"""MESMO: Max-value Entropy Search for Multi-objective Optimization.

Information-theoretic multi-objective BO: it selects the point expected to most reduce
the entropy of the Pareto-optimal objective values (the max-values), using BoTorch's
lower-bound multi-objective max-value entropy acquisition. Each objective is modeled
with an independent GP; progress is tracked by dominated hypervolume.

Run::

    python -m algorithms.multi_obj.mesmo --problem Penicillin --init 10 --iters 40

Sources:
S. Belakaria, A. Deshwal, J. R. Doppa. Max-value Entropy Search for Multi-Objective Bayesian Optimization. NeurIPS 2019. https://arxiv.org/abs/2011.01542
B. Tu, A. Gandy, N. Kantas, B. Shafei. Joint Entropy Search for Multi-objective Bayesian Optimization. NeurIPS 2022 (BoTorch implementation).
"""

from __future__ import annotations

import argparse

import torch
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from gpytorch.mlls import SumMarginalLogLikelihood

from .._bo_utils import (
    MultiObjectiveProblem,
    Result,
    add_common_args,
    finalize,
    initial_design,
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
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def optimize_problem(
    problem, n_init: int = 10, iters: int = 40, seed: int = 0
) -> Result:
    set_seed(seed)
    obj = MultiObjectiveProblem(problem)
    res = Result("mesmo", type(problem).__name__, seed, acquisition_function="MESMO")

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
        # Sample Pareto-optimal fronts from the model, then the box decomposition the
        # entropy acquisition integrates over.
        _, pareto_fronts = sample_optimal_points(
            model=model,
            bounds=obj.bounds,
            num_samples=8,
            num_points=10,
            optimizer=random_search_optimizer,
            maximize=True,
        )
        hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)
        acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model=model, hypercell_bounds=hypercell_bounds
        )
        candidate, _ = optimize_acqf(
            acqf, bounds=obj.bounds, q=1, num_restarts=5, raw_samples=128
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
    parser.add_argument("--iters", type=int, default=40)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args), args.init, args.iters, args.seed
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
