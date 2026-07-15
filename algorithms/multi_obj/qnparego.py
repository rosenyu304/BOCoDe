"""qNParEGO for multi-objective problems.

ParEGO with noisy expected improvement: at each step a random augmented-Chebyshev
scalarization of the objectives is drawn, and the point maximizing q-Log Noisy
Expected Improvement on that scalarization is proposed. Cheaper than qNEHVI and a
strong multi-objective baseline.

Following BoTorch's ``optimize_qnparego_and_get_observation``, the Chebyshev
scalarization is normalized by the GP **posterior mean at the training inputs**
(not the raw observations), so the scalarization is robust to observation noise.
The weights are re-drawn from the simplex at every proposal.

Run::

    python -m algorithms.multi_obj.qnparego --problem Penicillin --init 10 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
J. Knowles. ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. IEEE TEVC 2006.
BoTorch multi-objective tutorial: https://botorch.org/docs/tutorials/multi_objective_bo
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
    """qNParEGO multi-objective BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    GP fits + acquisition optimization run on ``device`` (default cuda when available);
    the objective is evaluated on CPU. With ``checkpoint`` set, the run is resumable:
    it restores ``(X, Y, completed_iters, RNG, ref_point)`` and saves after every iter
    (the inferred ``ref_point`` is checkpointed so resume keeps the same reference).
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = MultiObjectiveProblem(problem)
    m = obj.num_objectives
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "qnparego", type(problem).__name__, seed, acquisition_function="qLogNEI"
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
    for it in range(start_it, iters):
        X_dev = train_X.to(dev)
        # Standardized objective frame (same rationale as qnehvi). The Chebyshev
        # scalarization range-normalizes by the posterior mean and qLogNEI then optimizes
        # it; on raw, badly-scaled objectives (RE22's two objectives span ~819 and ~7295)
        # both steps go numerically fragile and optimize_acqf reports "gradf are NaN" --
        # RE22 x qnparego parked on 4 of 5 seeds for exactly this. Fit (and hence scalarize,
        # since ``pred`` is now in the fitted frame) on per-objective STANDARDIZED targets:
        # an affine, strictly-increasing per-objective transform, so the Pareto ranking the
        # scalarization sees is unchanged, but every objective is O(1). Reported hypervolume
        # (hv(), on raw train_Y) is untouched, so it stays comparable across methods.
        Y_dev = train_Y.to(dev)
        mu = Y_dev.mean(dim=0, keepdim=True)
        sigma = Y_dev.std(dim=0, keepdim=True).clamp_min(1e-9)
        model = _fit(X_dev, (Y_dev - mu) / sigma)
        # Chebyshev is normalized by the posterior mean at the observed inputs (now in the
        # standardized frame), not the (noisy) observations themselves.
        with torch.no_grad():
            pred = model.posterior(X_dev).mean
        weights = sample_simplex(m, dtype=DTYPE, device=dev).squeeze()
        scalarization = get_chebyshev_scalarization(weights=weights, Y=pred)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_dev,
            objective=GenericMCObjective(scalarization),
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
