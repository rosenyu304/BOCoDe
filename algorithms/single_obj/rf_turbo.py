"""RF-TuRBO: TuRBO with a random-forest surrogate in place of the local GP.

TuRBO-1 (Eriksson et al., 2019) with **every piece of the trust-region machinery kept**
-- ``length_init = 0.8``, ``length_min = 0.5**7``, ``length_max = 1.6``, ``succtol = 3``,
``failtol = ceil(max(4/q, d/q))``, the success test ``f_next > f_best + 1e-3*|f_best|``,
the Sobol candidate set with the ``p_perturb = min(1, 20/d)`` mask, restart on collapse --
but with the local **GP replaced by the SMAC-style random-forest surrogate** used in
:mod:`algorithms.single_obj.smac_rf` (a :class:`sklearn.ensemble.RandomForestRegressor`
whose predictive mean and uncertainty are the mean and cross-tree std of the ensemble).
This is the random-forest counterpart of ``tabicl_turbo``.

The trust-region state machine is imported from :mod:`algorithms.single_obj.turbo`, so the
methods share one implementation of it by construction.

**This method is not from a paper.** It is a BoCoDe ablation ("what does TuRBO do if its
surrogate is a random forest?"). Two things TuRBO gets from a GP have no RF counterpart, and
the substitutes are **our design choices**:

1. **Thompson sampling.** TuRBO draws a realization of the GP posterior over the candidate
   set and takes its argmax (``MaxPosteriorSampling``). A random forest gives only a
   *marginal* (mean, std) per candidate and no joint posterior, so there is no correlated
   sample path to draw. We draw **one independent Gaussian sample per candidate** from its
   own ``N(mean, std)`` predictive and take the argmax (repeating without replacement for a
   batch). This is Thompson sampling under an independence assumption; it is noisier / more
   exploratory than a GP's correlated draw, and that difference is a property of the
   surrogate, not a bug -- the same choice tfm_turbo / tabicl_turbo make.
2. **Isotropic trust region** (``x_center +/- L/2``). TuRBO scales side length ``i`` by the
   GP's ``i``-th ARD lengthscale; a random forest has no lengthscales, so there is nothing to
   weight the sides by -- the same choice tfm_turbo / tabicl_turbo make.

Run::

    python -m algorithms.single_obj.rf_turbo --problem Branin --init 20 --iters 50

Sources:
D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek. Scalable Global Optimization via Local Bayesian Optimization. NeurIPS 2019. https://arxiv.org/abs/1910.01739 (trust-region logic, constants; official code https://github.com/uber-research/TuRBO)
M. Lindauer et al. SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization. JMLR 23(54), 2022. https://github.com/automl/smac3 (the random-forest surrogate)
The GP -> random-forest substitution and the independent-marginal Thompson sampling are BoCoDe's, with no paper behind them.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch.quasirandom import SobolEngine

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
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
from .smac_rf import N_TREES, _rf_predict
from .turbo import TrustRegion


def _generate_batch(
    rf: RandomForestRegressor,
    tr: TrustRegion,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """TuRBO's candidate set inside the trust region, scored by RF Thompson sampling."""
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()
    # Isotropic trust region (a cube): ``x_center +/- L/2``. TuRBO weights each side by its
    # GP's ARD lengthscale; a random forest has no lengthscales, so the region is a cube.
    half = tr.length / 2.0
    tr_lb = torch.clamp(x_center - half, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + half, 0.0, 1.0)

    n_candidates = min(5000, max(2000, 200 * dim))  # TuRBO's n_cand
    sobol = SobolEngine(dim, scramble=True, seed=seed)
    pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_candidates).to(DTYPE)

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=DTYPE) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),))] = True
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Thompson sampling on the RF predictive: one independent Gaussian draw per candidate
    # from its own N(mean, std), argmax; repeat without replacement for a batch (TuRBO's
    # MaxPosteriorSampling does the same, but from a *correlated* GP sample path).
    mean, std = _rf_predict(rf, X_cand.numpy())
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for _ in range(batch_size):
        draw = mean + std * rng.standard_normal(size=mean.shape)
        if chosen:
            draw[chosen] = -np.inf
        chosen.append(int(draw.argmax()))
    return X_cand[chosen]


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 100,
    seed: int = 0,
    batch: int = 1,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """RF-TuRBO over the unit cube. ``n_init`` defaults to the dim-scaled BoCoDe default.

    The trust-region bookkeeping is TuRBO's (restarts included: a restart's initial design
    consumes evaluations from ``iters``); only the surrogate differs (RF Thompson sampling
    in place of the GP), and the trust region is an isotropic cube because a random forest
    has no lengthscales to weight the sides by. The RF runs on CPU via sklearn; ``device``
    is accepted for signature parity but the GPU is not used.
    """
    set_seed(seed)
    resolve_device(device)  # accepted for signature parity; RF runs on CPU (sklearn)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "rf_turbo",
        type(problem).__name__,
        seed,
        acquisition_function="RF predictive Thompson sampling (isotropic trust region)",
    )

    def restart_design(restart_idx: int):
        X = initial_design(n_init, obj.dim, seed + 9973 * restart_idx)
        return X, obj(X)

    if checkpoint and Path(checkpoint).exists():
        tr_X, tr_Y, evals, data = load_checkpoint(checkpoint, res)
        hist_X = torch.tensor(data["hist_X"], dtype=DTYPE)
        hist_Y = torch.tensor(data["hist_Y"], dtype=DTYPE)
        tr = TrustRegion(dim=obj.dim, batch_size=batch)
        tr.length = float(data["tr_length"])
        tr.success_counter = int(data["tr_success"])
        tr.failure_counter = int(data["tr_failure"])
        n_restarts = int(data["n_restarts"])
        best = hist_Y.max().item()
    else:
        tr_X, tr_Y = restart_design(0)
        hist_X, hist_Y = tr_X.clone(), tr_Y.clone()
        tr = TrustRegion(dim=obj.dim, batch_size=batch)
        n_restarts = 0
        evals = 0
        best = tr_Y.max().item()
        res.start(best)

    while evals < iters:
        rf = RandomForestRegressor(
            n_estimators=N_TREES, random_state=seed + evals, n_jobs=-1
        )
        rf.fit(tr_X.numpy(), tr_Y.numpy().ravel())
        cand = _generate_batch(rf, tr, tr_X, tr_Y, batch, seed + evals)
        cand = cand.detach().to(device="cpu", dtype=DTYPE)

        y = obj(cand)
        f_best = tr_Y.max().item()  # success is judged against the TR's own incumbent
        improved = y.max().item() > f_best + 1e-3 * math.fabs(f_best)
        tr_X = torch.cat([tr_X, cand], dim=0)
        tr_Y = torch.cat([tr_Y, y], dim=0)
        hist_X = torch.cat([hist_X, cand], dim=0)
        hist_Y = torch.cat([hist_Y, y], dim=0)
        tr.update(improved)

        for j in range(cand.shape[0]):
            best = max(best, y[j].item())
            res.record(best)
            evals += 1
            if evals >= iters:
                break

        if tr.restart and evals < iters:
            n_restarts += 1
            tr_X, tr_Y = restart_design(n_restarts)
            hist_X = torch.cat([hist_X, tr_X], dim=0)
            hist_Y = torch.cat([hist_Y, tr_Y], dim=0)
            tr = TrustRegion(dim=obj.dim, batch_size=batch)
            for j in range(n_init):
                best = max(best, tr_Y[j].item())
                res.record(best)
                evals += 1
                if evals >= iters:
                    break

        if checkpoint:
            res.set_history(hist_X, hist_Y, n_init)
            save_checkpoint(
                checkpoint,
                tr_X,
                tr_Y,
                res,
                evals,
                extra=dict(
                    hist_X=hist_X.numpy(),
                    hist_Y=hist_Y.numpy(),
                    tr_length=tr.length,
                    tr_success=tr.success_counter,
                    tr_failure=tr.failure_counter,
                    n_restarts=n_restarts,
                ),
            )
    res.set_history(hist_X, hist_Y, n_init)
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
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1, help="TuRBO batch size q")
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
        batch=args.batch,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
