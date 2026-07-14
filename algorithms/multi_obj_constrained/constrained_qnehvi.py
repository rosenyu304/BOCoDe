"""Constrained qNEHVI for constrained multi-objective problems.

Models every objective and every inequality constraint with an independent GP and
maximizes q-Log Noisy Expected Hypervolume Improvement with feasibility
constraints: the hypervolume improvement is only credited for points predicted to
satisfy all constraints (``c <= 0``). Reference point is in BoCoDe's maximization
frame.

Follows BoTorch's constrained MOBO pattern: objectives and constraints go into one
``ModelListGP`` (objectives first), an ``IdentityMCMultiOutputObjective`` selects the
objective outputs, and the constraint outputs are passed as ``constraints=`` callables
(feasible when the callable returns <= 0), which feasibility-weights the sampled
hypervolume improvement.

Run::

    python -m algorithms.multi_obj_constrained.constrained_qnehvi --problem WeldedBeam --init 12 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
BoTorch constrained multi-objective tutorial: https://botorch.org/docs/tutorials/constrained_multi_objective_bo
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    """Fit GPs for the objectives followed by the constraints (one ModelListGP)."""
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
    """Constrained qNEHVI multi-objective BO over the unit cube.

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
    assert nc > 0, "constrained_qnehvi requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "constrained_qnehvi",
        type(problem).__name__,
        seed,
        acquisition_function="qLogNEHVI",
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

    # Constraints are model outputs m..m+nc-1; BoTorch convention is feasible when
    # the constraint callable returns <= 0, matching BoCoDe's c <= 0.
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
    ref_dev = ref_point.to(dev)
    for it in range(start_it, iters):
        model = _fit(train_X.to(dev), train_Y.to(dev), train_C.to(dev))
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_dev,
            X_baseline=train_X.to(dev),
            sampler=sampler,
            prune_baseline=True,
            # select the objective outputs (0..m-1) from the combined model;
            # outputs m..m+nc-1 are the constraints used by the callables.
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(m))),
            constraints=constraint_callables,
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
