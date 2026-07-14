"""MESMO: Max-value Entropy Search for Multi-objective Optimization.

Information-theoretic multi-objective BO: it selects the point expected to most reduce
the entropy of the Pareto-optimal objective values (the max-values), using BoTorch's
lower-bound multi-objective max-value entropy acquisition (BoTorch's own MESMO
implementation). Each objective is modeled with an independent GP; progress is tracked
by dominated hypervolume.

Per the paper, each iteration draws ``S`` Pareto fronts from the GP posteriors and
averages the max-value entropy reduction over them; the reference code uses ``S = 10``
(``sample_number`` in its CLI). BoTorch samples those fronts from posterior sample
paths with a random-search inner optimizer and uses the Tu et al. (2022) lower-bound
estimator, where the paper uses RFF sample paths + NSGA-II and a closed-form
truncated-Gaussian bound -- same quantity, a more modern estimator.

Run::

    python -m algorithms.multi_obj.mesmo --problem Penicillin --init 10 --iters 40

Sources:
S. Belakaria, A. Deshwal, J. R. Doppa. Max-value Entropy Search for Multi-Objective Bayesian Optimization. NeurIPS 2019. https://papers.nips.cc/paper/8997-max-value-entropy-search-for-multi-objective-bayesian-optimization.pdf
Official MESMO code (sample_number = 10): https://github.com/belakaria/MESMO
B. Tu, A. Gandy, N. Kantas, B. Shafei. Joint Entropy Search for Multi-objective Bayesian Optimization. NeurIPS 2022 (BoTorch implementation).
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    DTYPE,
    MultiObjectiveProblem,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    initial_design,
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)

# Number of Pareto-front samples S averaged over by the acquisition, and the number of
# Pareto-optimal points per sampled front. S = 10 matches the reference implementation.
NUM_PARETO_SAMPLES = 10
NUM_PARETO_POINTS = 10


def _sample_pareto_fronts(model, bounds):
    """Sample ``S`` Pareto fronts from the GP posterior sample paths.

    BoTorch's random-search inner optimizer *requires* exactly ``num_points``
    non-dominated points per sampled path and raises ``RuntimeError`` when a path's
    front is smaller than that (which happens routinely early in a run). The
    reference MESMO has no such requirement -- it solves each sampled problem with
    NSGA-II and uses whatever front it returns -- so back off to a smaller front
    rather than killing the run.
    """
    num_points = NUM_PARETO_POINTS
    while True:
        try:
            _, pareto_fronts = sample_optimal_points(
                model=model,
                bounds=bounds,
                num_samples=NUM_PARETO_SAMPLES,
                num_points=num_points,
                optimizer=random_search_optimizer,
                maximize=True,
            )
            return pareto_fronts
        except RuntimeError:
            if num_points == 1:
                raise
            num_points = max(1, num_points // 2)


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
    problem,
    n_init: int | None = None,
    iters: int = 40,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """MESMO multi-objective BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    GP fits + acquisition optimization run on ``device`` (default cuda when available);
    the objective is evaluated on CPU. With ``checkpoint`` set, the run is resumable:
    it restores ``(X, Y, completed_iters, RNG, ref_point)`` and saves after every iter
    (the inferred ``ref_point`` is checkpointed so resume keeps the same reference).
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = MultiObjectiveProblem(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("mesmo", type(problem).__name__, seed, acquisition_function="MESMO")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, data = load_checkpoint(checkpoint, res)
        ref_point = torch.tensor(data["ref_point"], dtype=DTYPE)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_Y, _ = obj.evaluate_raw(train_X)
        ref_point = obj.hv_ref_point(train_Y)
        start_it = 0

    def hv(Y):
        return (
            DominatedPartitioning(ref_point=ref_point.cpu(), Y=Y.cpu())
            .compute_hypervolume()
            .item()
        )

    if start_it == 0:
        res.start(hv(train_Y))

    bounds_dev = obj.bounds.to(dev)
    for it in range(start_it, iters):
        model = _fit(train_X.to(dev), train_Y.to(dev))
        # Sample Pareto-optimal fronts from the model, then the box decomposition the
        # entropy acquisition integrates over.
        pareto_fronts = _sample_pareto_fronts(model, bounds_dev)
        hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)
        acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model=model, hypercell_bounds=hypercell_bounds
        )
        candidate, _ = optimize_acqf(
            acqf,
            bounds=bounds_dev,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)
        y, _ = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        res.record(hv(train_Y))
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(
                checkpoint,
                train_X,
                train_Y,
                res,
                it + 1,
                extra={"ref_point": ref_point.cpu().numpy()},
            )
    res.set_history(train_X, train_Y, n_init)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument(
        "--checkpoint", default=None, help="resumable checkpoint .npz path"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
