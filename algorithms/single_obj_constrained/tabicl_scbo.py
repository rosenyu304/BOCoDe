"""TabICL-SCBO: SCBO with a tabular foundation model (TabICL) as the surrogate.

SCBO (Eriksson & Poloczek, 2021) with **the whole constrained trust-region machinery kept**
-- the region centered on the best *feasible* point (or, before one exists, on the
least-violating point), ``L_init = 0.8`` / ``L_min = 2**-7`` / ``L_max = 1.6``,
``tau_s = 10`` and ``tau_f = ceil(max(4/q, d/q))``, the Sobol candidate set with the
``p_perturb = min(1, 20/d)`` mask, the constrained success rule, restart on collapse, the
**Gaussian copula** on the objective and the **bilog** warp on each constraint -- but with the
objective GP and the ``n_constraints`` constraint GPs all replaced by **frozen, pretrained
TabICL regressors** (one per output). This is the TabICL counterpart of ``tfm_scbo``.

Objective and constraints are each scored by their **own zero-shot in-context fit** --
``1 + n_constraints`` TabICL fit/predict passes per iteration, one output column each -- so
peak memory stays O(1) in the constraint count. No surrogate is gradient-trained; these
passes replace ``1 + n_constraints`` GP fits.

The trust-region state machine, the copula, the center rule and the success rule are all
imported from :mod:`algorithms.single_obj_constrained.scbo` / ``turbo``, so the methods share
one implementation of them by construction.

**This method is not from a paper** -- it is a BoCoDe ablation ("what does SCBO do if its
surrogates are a TFM?"). One thing SCBO gets from GPs has no TabICL counterpart, and the
substitute is **our design choice**:

* **Constrained Thompson sampling.** SCBO draws a realization of the objective GP's posterior
  and of every constraint GP's posterior over the candidate set, keeps the candidates whose
  constraint draws are all feasible, and takes the best objective draw among them
  (``ConstrainedMaxPosteriorSampling``); with none feasible it falls back to the smallest total
  predicted violation. We do exactly that, but each draw comes from TabICL's **marginal**
  predictive for that (candidate, output) -- TabICL has no joint posterior, so the draws are
  independent across candidates rather than a correlated sample path.

Two things that are *not* design choices, as in tfm_scbo:

* SCBO's trust region is **isotropic** in both the paper's Appendix C and the BoTorch tutorial,
  so there is nothing here to substitute for GP lengthscales.
* **bilog** ``sgn(c) * log(1 + |c|)`` is strictly increasing and fixes zero, so
  ``bilog(c) <= 0 <=> c <= 0``: the feasibility test and the total-violation tie-break are
  applied directly to the sampled bilog values, and BoCoDe's ``c <= 0 is feasible`` convention
  is preserved exactly. The Gaussian copula on the objective is strictly increasing too, so the
  argmax over objective draws is unchanged.

Needs TabICL (``pip install tabicl``). Run::

    python -m algorithms.single_obj_constrained.tabicl_scbo --problem PressureVessel --iters 50

Sources:
D. Eriksson and M. Poloczek. Scalable Constrained Bayesian Optimization. AISTATS 2021. https://arxiv.org/abs/2002.08526 (trust region, copula/bilog warps, constants)
BoTorch SCBO tutorial: https://botorch.org/docs/tutorials/scalable_constrained_bo (the tolerances, the constrained-TS structure)
J. Qu, D. Holzmüller, G. Varoquaux, M. Le Morvan. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. ICML 2025. https://github.com/soda-inria/tabicl (the surrogate)
The GP -> TabICL substitution and the independent-marginal constrained Thompson sampling are BoCoDe's, with no paper behind them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.models.transforms import Bilog
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
    save_checkpoint,
    set_seed,
)
from .._tabicl_utils import TabICLSurrogate
from .scbo import _best_index, _gaussian_copula, _is_success, _tr


def _constrained_thompson_sample(
    surrogate: TabICLSurrogate,
    train_X: torch.Tensor,
    Yo_t: torch.Tensor,
    Yc_t: torch.Tensor,
    cand: torch.Tensor,
) -> int:
    """SCBO's constrained Thompson sampling, on TabICL's predictive distribution.

    ``Yo_t`` is the copula-warped objective (n, 1) and ``Yc_t`` the bilog-warped constraints
    (n, nc); both warps are strictly increasing and bilog fixes zero, so feasibility and the
    objective ranking can be read straight off the sampled values.

    Returns the index into ``cand`` of the chosen point: the best sampled objective among the
    candidates whose sampled constraints are all ``<= 0``, or -- if no candidate is sampled
    feasible -- the one with the smallest total sampled violation (SCBO's fallback).
    """
    n_cand = cand.shape[0]
    Y_cols = torch.cat(
        [Yo_t, Yc_t], dim=1
    )  # (n_ctx, m): objective, then each constraint
    m = Y_cols.shape[1]

    # Run each output (objective, then each constraint) through TabICL as its OWN zero-shot
    # in-context regression -- m separate fit/predict passes, one column each -- rather than
    # stacking them. ICL is cheap for TabICL, so looping keeps peak memory O(1) in the
    # constraint count. The draws are independent marginals either way (this method's design
    # choice -- TabICL has no joint posterior), so one-per-column is the same estimator as a
    # stacked pass, not an approximation of it.
    draws = []
    with torch.no_grad():
        for j in range(m):
            surrogate.fit(train_X, Y_cols[:, j : j + 1])
            _, _, q = surrogate.score(cand)
            draws.append(surrogate.sample_from_quantiles(q).reshape(n_cand).to("cpu"))
    draw = torch.stack(draws, dim=1)  # (n_cand, m)

    obj_s, con_s = draw[:, 0], draw[:, 1:]
    feas = (con_s <= 0).all(dim=1)  # bilog is sign-preserving: <= 0 means feasible
    if feas.any():
        score = obj_s.clone()
        score[~feas] = -float("inf")
        return int(score.argmax())
    return int(con_s.clamp(min=0).sum(dim=1).argmin())


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 80,
    seed: int = 0,
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """TabICL-SCBO. ``n_init`` defaults to the dim-scaled BoCoDe default.

    The reported value is the best **feasible** objective so far (``-inf`` until a feasible
    point is found), as in :mod:`algorithms.single_obj_constrained.scbo`. The full evaluation
    history -- across restarts, with the constraint values -- is saved on the ``Result``.
    """
    set_seed(seed)
    obj = ProblemObjective(problem)
    nc = obj.num_constraints
    assert nc > 0, "tabicl_scbo requires a constrained problem"
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "tabicl_scbo",
        type(problem).__name__,
        seed,
        acquisition_function="TabICL predictive constrained Thompson sampling",
    )
    n_cand = min(
        5000, max(2000, 200 * dim)
    )  # TuRBO's / SCBO Appendix C's candidate count
    surrogate = TabICLSurrogate(device=device, seed=seed)
    bilog = Bilog()

    def start_region(offset: int):
        """Fresh trust region on a fresh initial design (TuRBO/SCBO restart)."""
        X = initial_design(n_init, dim, seed + offset)
        Yo, Yc = obj.evaluate_raw(X)
        i = _best_index(Yo, Yc)
        return X, Yo, Yc, _tr(dim), Yo[i].item(), Yc[i].clone()

    def best_feasible(Yo, Yc):
        feas = (Yc <= 0).all(dim=-1)
        return Yo[feas].max().item() if feas.any() else -float("inf")

    if checkpoint and Path(checkpoint).exists():
        X, Yo, start_it, data = load_checkpoint(checkpoint, res)
        Yc = torch.tensor(data["con"], dtype=DTYPE)
        hist_X = torch.tensor(data["hist_X"], dtype=DTYPE)
        hist_Yo = torch.tensor(data["hist_Yo"], dtype=DTYPE)
        hist_Yc = torch.tensor(data["hist_Yc"], dtype=DTYPE)
        tr = _tr(dim)
        tr.length = float(data["tr_length"])
        tr.success_counter = int(data["tr_success"])
        tr.failure_counter = int(data["tr_failure"])
        best_value = float(data["best_value"])
        best_con = torch.tensor(data["best_con"], dtype=DTYPE)
        n_restarts = int(data["n_restarts"])
        best = float(data["best"])
    else:
        X, Yo, Yc, tr, best_value, best_con = start_region(0)
        hist_X, hist_Yo, hist_Yc = X.clone(), Yo.clone(), Yc.clone()
        n_restarts = 0
        best = best_feasible(Yo, Yc)
        res.start(best)
        start_it = 0

    for it in range(start_it, iters):
        # SCBO's warps: Gaussian copula on the objective, bilog on every constraint.
        Yo_t = _gaussian_copula(Yo)
        Yc_t = bilog(Yc)[0]

        # Trust region around the best feasible (or least-violating) point. Isotropic --
        # SCBO does not weight the sides by lengthscales.
        x_center = X[_best_index(Yo, Yc)].clone()
        tr_lb = torch.clamp(x_center - tr.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + tr.length / 2.0, 0.0, 1.0)

        sobol = SobolEngine(dim, scramble=True, seed=seed + it)
        pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_cand).to(DTYPE)
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_cand, dim, dtype=DTYPE) <= prob_perturb
        empty = torch.where(mask.sum(dim=1) == 0)[0]
        mask[empty, torch.randint(0, dim, (len(empty),))] = True
        cand = x_center.expand(n_cand, dim).clone()
        cand[mask] = pert[mask]

        choice = _constrained_thompson_sample(surrogate, X, Yo_t, Yc_t, cand)
        x_new = cand[choice : choice + 1]

        o, c = obj.evaluate_raw(x_new)
        improved = _is_success(o.reshape(-1)[0].item(), c[0], best_value, best_con)
        if improved:  # the incumbent moves only on a success
            best_value, best_con = o.reshape(-1)[0].item(), c[0].clone()
        tr.update(improved)

        X = torch.cat([X, x_new], dim=0)
        Yo = torch.cat([Yo, o], dim=0)
        Yc = torch.cat([Yc, c], dim=0)
        hist_X = torch.cat([hist_X, x_new], dim=0)
        hist_Yo = torch.cat([hist_Yo, o], dim=0)
        hist_Yc = torch.cat([hist_Yc, c], dim=0)
        best = max(best, best_feasible(o, c))
        res.record(best)

        if tr.restart:
            # L < L_min: discard the local data and start a new trust region on a fresh
            # initial design (SCBO Sec. 3.1 step 2d). The evaluations still count.
            n_restarts += 1
            X, Yo, Yc, tr, best_value, best_con = start_region(n_restarts * 1000)
            hist_X = torch.cat([hist_X, X], dim=0)
            hist_Yo = torch.cat([hist_Yo, Yo], dim=0)
            hist_Yc = torch.cat([hist_Yc, Yc], dim=0)
            best = max(best, best_feasible(Yo, Yc))

        if checkpoint:
            res.set_history(hist_X, hist_Yo, n_init, c=hist_Yc)
            save_checkpoint(
                checkpoint,
                X,
                Yo,
                res,
                it + 1,
                extra={
                    "con": Yc.cpu().numpy(),
                    "hist_X": hist_X.cpu().numpy(),
                    "hist_Yo": hist_Yo.cpu().numpy(),
                    "hist_Yc": hist_Yc.cpu().numpy(),
                    "tr_length": tr.length,
                    "tr_success": tr.success_counter,
                    "tr_failure": tr.failure_counter,
                    "best_value": best_value,
                    "best_con": best_con.cpu().numpy(),
                    "n_restarts": n_restarts,
                    "best": best,
                },
            )
    res.set_history(hist_X, hist_Yo, n_init, c=hist_Yc)
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
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument(
        "--checkpoint", default=None, help="resumable checkpoint .npz path"
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
