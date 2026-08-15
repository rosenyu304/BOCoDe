"""Random-forest surrogate + Upper Confidence Bound (UCB) Bayesian optimization.

The SMAC-style random-forest surrogate of :mod:`algorithms.single_obj.smac_rf` (a
:class:`sklearn.ensemble.RandomForestRegressor` whose predictive mean and uncertainty are
the mean and cross-tree std of the ensemble) but with the acquisition swapped from
Expected Improvement to the **Upper Confidence Bound** ``mean + beta * std`` (Srinivas et
al., 2010). As with smac_rf the acquisition is non-differentiable, so it is maximized over
a large Sobol candidate set in the normalized ``[0, 1]^d`` domain rather than with a
gradient-based ``optimize_acqf``. The UCB coefficient is the module constant :data:`BETA`.

Run::

    python -m algorithms.single_obj.rf_ucb --problem Branin --init 10 --iters 40

Sources:
N. Srinivas, A. Krause, S. Kakade, and M. Seeger. Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. ICML 2010. https://arxiv.org/abs/0912.3995
M. Lindauer et al. SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization. JMLR 23(54), 2022. https://github.com/automl/smac3 (the random-forest surrogate)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
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
from .smac_rf import N_CANDIDATES, N_TREES, _rf_predict

BETA = 2.0  # coefficient on the RF predictive std in mean + BETA * std


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 40,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Continuous RF + UCB BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    A random forest is fit on the observed data each iteration and UCB (from the ensemble
    mean and cross-tree std) is maximized over a Sobol candidate set drawn in ``[0, 1]^d``.
    The random forest and candidate sampling are seeded from the run seed, so the run is
    deterministic. The RF surrogate runs on CPU via sklearn; ``device`` is accepted for
    signature parity but the GPU is not used. With ``checkpoint`` set, the run is resumable.
    """
    set_seed(seed)
    resolve_device(device)  # accepted for signature parity; RF runs on CPU (sklearn)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("rf_ucb", type(problem).__name__, seed, acquisition_function="UCB")

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
        ucb = mean + BETA * std
        arg = int(ucb.argmax())

        candidate = torch.from_numpy(cand[arg : arg + 1]).to(DTYPE)
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(
            best,
            mean=float(mean[arg]),
            variance=float(std[arg] ** 2),
            acq_value=float(ucb[arg]),
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
