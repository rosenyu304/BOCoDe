"""Constrained qParEGO for constrained multi-objective problems.

ParEGO with feasibility: each step draws a random augmented-Chebyshev
scalarization of the objectives and maximizes q-Log Noisy Expected Improvement on
that scalarization, feasibility-weighted by the constraint outputs (``c <= 0`` is
feasible). Objectives and constraints are each modeled with an independent GP.
Cheaper than constrained qNEHVI.

Follows BoTorch's constrained MOBO pattern: objectives and constraints go into one
``ModelListGP`` (objectives first); the scalarization is applied only to the objective
outputs ``Z[..., :m]``; the constraint outputs are passed as ``constraints=`` callables.
The Chebyshev scalarization is normalized with the GP **posterior mean at the training
inputs**, following BoTorch's MOBO tutorial (``optimize_qnparego_and_get_observation``)
and matching ``multi_obj/qnparego`` -- the constrained tutorial passes the raw
observations instead, but the two BoCoDe ParEGO variants must agree with each other.

Run::

    python -m algorithms.multi_obj_constrained.constrained_qparego --problem WeldedBeam --init 12 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
J. Knowles. ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. IEEE TEVC 2006.
BoTorch constrained multi-objective tutorial: https://botorch.org/docs/tutorials/constrained_multi_objective_bo
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    default_n_init,
    finalize,
    initial_design,
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
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
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Constrained qParEGO multi-objective BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    GP fits + acquisition optimization run on ``device`` (default cuda when available);
    the objective is evaluated on CPU. With ``checkpoint`` set, the run is resumable:
    it restores ``(X, Y, C, completed_iters, RNG, ref_point)`` and saves after every
    iter (the inferred ``ref_point`` is checkpointed so resume keeps the same reference).
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = MultiObjectiveProblem(problem)
    m, nc = obj.num_objectives, obj.num_constraints
    assert nc > 0, "constrained_qparego requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "constrained_qparego",
        type(problem).__name__,
        seed,
        acquisition_function="qLogNEI",
    )

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, data = load_checkpoint(checkpoint, res)
        train_C = torch.tensor(data["C"], dtype=DTYPE)
        ref_point = torch.tensor(data["ref_point"], dtype=DTYPE)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_Y, train_C = obj.evaluate_raw(train_X)
        ref_point = obj.hv_ref_point(train_Y)
        start_it = 0

    # Constraint callables index the constraint model outputs (m..m+nc-1); feasible
    # when the value is <= 0, matching BoCoDe's convention.
    constraint_callables = [(lambda Z, i=m + j: Z[..., i]) for j in range(nc)]

    def feasible_hv(Y, C):
        feas = (C <= 0).all(dim=1)
        if not feas.any():
            return 0.0
        return (
            DominatedPartitioning(ref_point=ref_point.cpu(), Y=Y[feas].cpu())
            .compute_hypervolume()
            .item()
        )

    if start_it == 0:
        res.start(feasible_hv(train_Y, train_C))

    bounds_dev = obj.bounds.to(dev)
    for it in range(start_it, iters):
        X_dev, Y_dev = train_X.to(dev), train_Y.to(dev)
        model = _fit(X_dev, Y_dev, train_C.to(dev))
        # BoTorch's MOBO tutorial normalizes the Chebyshev scalarization with the GP
        # posterior mean at the training inputs, not the raw observations -- that is the
        # "N" (noisy) in qNParEGO. The constrained tutorial happens to pass the raw
        # `train_obj` instead; we follow the MOBO tutorial in BOTH ParEGO variants so the
        # two rows in the paper are doing the same thing (Rosen's call, 2026-07-13).
        # The model list is objectives-first, so the objective columns are [..., :m].
        with torch.no_grad():
            pred = model.posterior(X_dev).mean[..., :m]
        weights = sample_simplex(m, dtype=DTYPE, device=dev).squeeze()
        scalarization = get_chebyshev_scalarization(weights=weights, Y=pred)

        # Apply the scalarization only to the objective outputs (0..m-1); the
        # remaining outputs are the constraints, handled by `constraints=` below.
        def scalarized_objective(Z, X=None, s=scalarization, mm=m):
            return s(Z[..., :mm])

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_dev,
            objective=GenericMCObjective(scalarized_objective),
            constraints=constraint_callables,
            sampler=sampler,
            prune_baseline=True,
        )
        candidate, _ = optimize_acqf(
            acqf,
            bounds=bounds_dev,
            q=1,
            num_restarts=10,  # tutorial's NUM_RESTARTS
            raw_samples=512,  # tutorial's RAW_SAMPLES
            options={"batch_limit": 5, "maxiter": 200},
        )
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)
        y, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        train_C = torch.cat([train_C, c], dim=0)
        res.record(feasible_hv(train_Y, train_C))
        if checkpoint:
            res.set_history(train_X, train_Y, n_init, c=train_C)
            save_checkpoint(
                checkpoint,
                train_X,
                train_Y,
                res,
                it + 1,
                extra={
                    "C": train_C.cpu().numpy(),
                    "ref_point": ref_point.cpu().numpy(),
                },
            )
    res.set_history(train_X, train_Y, n_init, c=train_C)
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
    parser.add_argument("--iters", type=int, default=50)
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
