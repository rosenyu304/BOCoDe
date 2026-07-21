"""Constrained Expected Improvement with a random-forest surrogate (RF-CEI).

The constrained-EI baseline of :mod:`algorithms.single_obj_constrained.constrained_ei` --
Expected Improvement on the objective weighted by the probability that every constraint is
satisfied (``c <= 0``) -- but with the objective and constraint GPs replaced by the
SMAC-style **random-forest surrogate** of :mod:`algorithms.single_obj.smac_rf` (one
:class:`sklearn.ensemble.RandomForestRegressor` per output; predictive mean and uncertainty
are the mean and cross-tree std of the ensemble).

Because the RF acquisition is non-differentiable it is maximized over a large Sobol
candidate set in ``[0, 1]^d`` (as in smac_rf) rather than with a gradient-based
``optimize_acqf``. Following Gelbart et al. (2014), Sec. 3.2 -- Eq. (9) -- before any
feasible point has been observed the EI factor is dropped and the acquisition maximizes the
probability of feasibility alone, ``prod_k P(c_k(x) <= 0)``; once a feasible point exists the
EI target is the best *feasible* objective and the acquisition is
``EI(x) * prod_k P(c_k(x) <= 0)``. Each ``P(c_k <= 0)`` is a Gaussian tail probability from
the RF's ``(mean, std)`` predictive for that constraint.

Run::

    python -m algorithms.single_obj_constrained.rf_cei --problem PressureVessel --init 10 --iters 50

Sources:
J. R. Gardner, M. J. Kusner, Z. Xu, K. Q. Weinberger, and J. P. Cunningham. Bayesian Optimization with Inequality Constraints. ICML 2014.
M. Gelbart, J. Snoek, and R. P. Adams. Bayesian Optimization with Unknown Constraints. UAI 2014. https://arxiv.org/abs/1403.5607
M. Lindauer et al. SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization. JMLR 23(54), 2022. https://github.com/automl/smac3 (the random-forest surrogate)
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
from ..single_obj.smac_rf import N_CANDIDATES, N_TREES, _expected_improvement, _rf_predict


def _fit_rf(X: np.ndarray, y: np.ndarray, random_state: int):
    """Fit one random forest on the rows where ``y`` is finite.

    Some benchmark outputs are non-finite on a sliver of the domain (a verbatim property of
    the official suite, not a defect; see constrained_ei's note). Those rows are withheld from
    the surrogate but still recorded in the run history at their true value.
    """
    m = np.isfinite(y)
    rf = RandomForestRegressor(n_estimators=N_TREES, random_state=random_state, n_jobs=-1)
    rf.fit(X[m], y[m])
    return rf


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """RF constrained-EI BO. ``n_init`` defaults to the dim-scaled BoCoDe default.

    A random forest is fit for the objective and one per constraint each iteration, and the
    constrained-EI acquisition (EI * probability of feasibility, or probability of feasibility
    alone before a feasible point exists) is maximized over a Sobol candidate set drawn in
    ``[0, 1]^d``. The RF surrogates run on CPU via sklearn; ``device`` is accepted for
    signature parity but the GPU is not used. With ``checkpoint`` set the run is resumable.
    """
    set_seed(seed)
    resolve_device(device)  # accepted for signature parity; RF runs on CPU (sklearn)
    obj = ProblemObjective(problem)
    assert obj.num_constraints > 0, "rf_cei requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "rf_cei", type(problem).__name__, seed, acquisition_function="constrained EI"
    )

    def best_feasible(obj_v, con_v):
        feas = (con_v <= 0).all(dim=1)
        return obj_v[feas].max().item() if feas.any() else -float("inf")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_obj, start_it, data = load_checkpoint(checkpoint, res)
        train_con = torch.tensor(data["con"], dtype=DTYPE)
        best = best_feasible(train_obj, train_con)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_obj, train_con = obj.evaluate_raw(train_X)
        best = best_feasible(train_obj, train_con)
        res.start(best)
        start_it = 0

    sobol = qmc.Sobol(d=obj.dim, seed=seed)
    for it in range(start_it, iters):
        X_np = train_X.numpy()
        obj_np = train_obj.numpy().ravel()
        con_np = train_con.numpy()

        rf_obj = _fit_rf(X_np, obj_np, seed + it)
        rf_cons = [
            _fit_rf(X_np, con_np[:, k], seed + it + 1000 * (k + 1))
            for k in range(obj.num_constraints)
        ]

        cand = sobol.random(N_CANDIDATES).astype(np.float64)
        # Probability of feasibility for every constraint: P(c_k <= 0) from the RF (mean, std).
        pof = np.ones(N_CANDIDATES)
        for rf_c in rf_cons:
            mean_c, std_c = _rf_predict(rf_c, cand)
            std_c = np.maximum(std_c, 1e-9)
            pof *= norm.cdf((0.0 - mean_c) / std_c)

        feas = (train_con <= 0).all(dim=1)
        if feas.any():
            # EI on the objective against the best feasible value, weighted by feasibility.
            mean_o, std_o = _rf_predict(rf_obj, cand)
            ei = _expected_improvement(mean_o, std_o, train_obj[feas].max().item())
            acq = ei * pof
        else:
            # No feasible point yet: maximize probability of feasibility alone (Gelbart Eq. 9).
            acq = pof
        arg = int(acq.argmax())

        candidate = torch.from_numpy(cand[arg : arg + 1]).to(DTYPE)
        o, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_obj = torch.cat([train_obj, o], dim=0)
        train_con = torch.cat([train_con, c], dim=0)
        best = max(best, best_feasible(o, c))
        res.record(best, acq_value=float(acq[arg]))
        if checkpoint:
            res.set_history(train_X, train_obj, n_init, c=train_con)
            save_checkpoint(
                checkpoint,
                train_X,
                train_obj,
                res,
                it + 1,
                extra={"con": train_con.cpu().numpy()},
            )
    res.set_history(train_X, train_obj, n_init, c=train_con)
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
