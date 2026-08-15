"""SMAC-style random-forest surrogate Bayesian optimization (RF-EI).

A standalone re-implementation of the random-forest surrogate used by SMAC3: fit a
:class:`sklearn.ensemble.RandomForestRegressor` on the observed ``(X, y)``, then
propose the next point with Expected Improvement (EI). Because a random forest has
no closed-form posterior, the predictive *mean* and *variance* at a candidate are
estimated empirically from the ensemble -- the mean and the spread (variance) of the
individual tree predictions. This "mean/std across the trees" surrogate is the
classic SMAC RF-EI formulation.

The RF-EI acquisition is non-differentiable, so it is NOT optimized with a
gradient-based ``optimize_acqf``. Instead a large Sobol candidate set is drawn in the
normalized ``[0, 1]^d`` domain and the argmax-EI candidate is selected.

This is a faithful, dependency-light standalone implementation (sklearn only, no
``smac`` package) chosen for reproducibility. It follows the surrogate/acquisition
design of SMAC3.

Run::

    python -m algorithms.single_obj.smac_rf --problem Branin --init 10 --iters 40

Sources:
M. Lindauer, K. Eggensperger, M. Feurer, A. Biedenkapp, D. Deng, C. Benjamins, T. Ruhkopf, R. Sass, and F. Hutter. SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization. Journal of Machine Learning Research 23(54), 2022. https://github.com/automl/smac3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, qmc
from sklearn.ensemble import RandomForestRegressor

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

N_TREES = 100
N_CANDIDATES = 4000


def _rf_predict(rf: RandomForestRegressor, X: np.ndarray):
    """Empirical predictive (mean, std) from the ensemble's per-tree predictions.

    Following the SMAC RF surrogate, the predictive mean is the mean over the trees
    and the predictive uncertainty is the spread (std) of the individual tree
    predictions at each candidate.
    """
    preds = np.stack([est.predict(X) for est in rf.estimators_], axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std


def _expected_improvement(mean: np.ndarray, std: np.ndarray, best: float):
    """Expected Improvement for maximization from the RF mean/std surrogate."""
    std = np.maximum(std, 1e-9)
    z = (mean - best) / std
    return (mean - best) * norm.cdf(z) + std * norm.pdf(z)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 40,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Continuous SMAC-style RF-EI BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    A random forest is fit on the observed data each iteration and EI (from the
    ensemble mean and cross-tree std) is maximized over a Sobol candidate set drawn
    in ``[0, 1]^d``. The random forest and candidate sampling are seeded from the run
    seed, so the run is deterministic. The RF surrogate runs on CPU via sklearn; the
    ``device`` argument is accepted (for a uniform ``optimize_problem`` signature) but
    the GPU is not used. With ``checkpoint`` set, the run is resumable: it loads
    ``(X, y, completed_iters, RNG)`` if present and saves after every iteration.
    """
    set_seed(seed)
    resolve_device(device)  # accepted for signature parity; RF runs on CPU (sklearn)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("smac_rf", type(problem).__name__, seed, acquisition_function="EI")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
        best = train_Y.max().item()
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        start_it = 0

    sobol = qmc.Sobol(d=obj.dim, seed=seed)
    for it in range(start_it, iters):
        rf = RandomForestRegressor(
            n_estimators=N_TREES, random_state=seed + it, n_jobs=-1
        )
        rf.fit(train_X.numpy(), train_Y.numpy().ravel())

        cand = sobol.random(N_CANDIDATES).astype(np.float64)
        mean, std = _rf_predict(rf, cand)
        ei = _expected_improvement(mean, std, best)
        arg = int(ei.argmax())

        candidate = torch.from_numpy(cand[arg : arg + 1]).to(DTYPE)
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(
            best,
            mean=float(mean[arg]),
            variance=float(std[arg] ** 2),
            acq_value=float(ei[arg]),
        )
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(checkpoint, train_X, train_Y, res, it + 1)
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
