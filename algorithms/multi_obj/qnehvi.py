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
from pathlib import Path

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
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """qNEHVI multi-objective BO over the unit cube.

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
    res = Result(
        "qnehvi", type(problem).__name__, seed, acquisition_function="qLogNEHVI"
    )

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
    ref_dev = ref_point.to(dev)
    for it in range(start_it, iters):
        # qLogNEHVI computes hypervolume improvement in the model's OUTPUT units. When the
        # objectives differ by orders of magnitude (RE25 spans ~1e2 and ~1e6; RE22/RE24 are
        # milder but still ~10-1000x), the large objective swamps the small one and the log
        # box-decomposition underflows -> optimize_acqf reports "gradf are NaN" and the tuple
        # is parked. Run the model + acquisition in a per-objective STANDARDIZED frame: an
        # affine, strictly-increasing, per-objective transform, so which point dominates which
        # is unchanged, but every objective is O(1). The REPORTED hypervolume (hv(), below)
        # stays in the RAW frame so it remains comparable across methods and runs. This is the
        # BoTorch MOO tutorial's own standardization; here it is also what keeps qLogNEHVI
        # finite. NB: fit on the standardized targets directly (Standardize(m=1) inside _fit
        # then acts as ~identity), and give the acquisition the ref point in the same frame.
        Y_dev = train_Y.to(dev)
        mu = Y_dev.mean(dim=0, keepdim=True)
        sigma = Y_dev.std(dim=0, keepdim=True).clamp_min(1e-9)
        model = _fit(train_X.to(dev), (Y_dev - mu) / sigma)
        ref_std = (ref_dev - mu.squeeze(0)) / sigma.squeeze(0)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_std,
            X_baseline=train_X.to(dev),
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
