"""TFM-SCBO: SCBO with a tabular foundation model (TabPFN) as the surrogate.

SCBO (Eriksson & Poloczek, 2021) with **the whole constrained trust-region machinery kept**
-- the region centered on the best *feasible* point (or, before one exists, on the
least-violating point), ``L_init = 0.8`` / ``L_min = 2**-7`` / ``L_max = 1.6``,
``tau_s = 10`` and ``tau_f = ceil(max(4/q, d/q))`` (the BoTorch tutorial's tolerances, as in
:mod:`algorithms.single_obj_constrained.scbo`), the Sobol candidate set with the
``p_perturb = min(1, 20/d)`` mask, the constrained success rule (feasible beats infeasible;
two infeasibles compared by total violation), restart on collapse, the **Gaussian copula**
on the objective and the **bilog** warp on each constraint -- but with the objective GP and
the ``n_constraints`` constraint GPs all replaced by **one frozen, pretrained TabPFN**.

Objective and constraints are scored in a **single forward pass**: the same candidate inputs
are paired with the objective and with each constraint as separate columns of TabPFN's batch
dimension (the trick :mod:`algorithms.single_obj_constrained.pfn_cei` uses). No surrogate is
trained; one TabPFN call per iteration replaces ``1 + n_constraints`` GP fits.

The trust-region state machine, the copula, the trust-region center rule and the success rule
are all imported from :mod:`algorithms.single_obj_constrained.scbo` / ``turbo``, so the two
methods share one implementation of them by construction.

**This method is not from a paper** -- it is a BoCoDe ablation ("what does SCBO do if its
surrogates are a TFM?"). One thing SCBO gets from GPs has no TabPFN counterpart, and the
substitute is **our design choice**:

* **Constrained Thompson sampling.** SCBO draws a realization of the objective GP's
  posterior and of every constraint GP's posterior over the candidate set, keeps the
  candidates whose constraint draws are all feasible, and takes the best objective draw among
  them (``ConstrainedMaxPosteriorSampling``); with no candidate predicted feasible it falls
  back to the smallest total predicted violation. We do exactly that, but each draw comes
  from TabPFN's **marginal** bar distribution for that (candidate, output) -- TabPFN has no
  joint posterior, so the draws are independent across candidates rather than a correlated
  sample path. Noisier / more exploratory than a GP's draw; a property of the surrogate.

Two things that are *not* design choices, to head off the obvious sign worries:

* SCBO's trust region is **isotropic** (``x_center +/- L/2``, no ARD-lengthscale weighting) in
  both the paper's Appendix C and the BoTorch tutorial, so unlike TFM-TuRBO there is nothing
  here to substitute for GP lengthscales.
* **bilog** ``sgn(c) * log(1 + |c|)`` is strictly increasing and fixes zero, so
  ``bilog(c) <= 0 <=> c <= 0``: the feasibility test and the total-violation tie-break are
  applied directly to the sampled bilog values, with no inverse transform, and BoCoDe's
  ``c <= 0 is feasible`` convention is preserved exactly. The Gaussian copula on the
  objective is strictly increasing too, so the argmax over objective draws is unchanged.

Needs TabPFN >= v3 (see docs/tfm_setup.md). Run::

    python -m algorithms.single_obj_constrained.tfm_scbo --problem PressureVessel --iters 50

Sources:
D. Eriksson and M. Poloczek. Scalable Constrained Bayesian Optimization. AISTATS 2021. https://arxiv.org/abs/2002.08526 (trust region, copula/bilog warps, constants)
D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, M. Poloczek. Scalable Global Optimization via Local Bayesian Optimization. NeurIPS 2019. https://arxiv.org/abs/1910.01739 (the TuRBO trust region SCBO builds on)
BoTorch SCBO tutorial: https://botorch.org/docs/tutorials/scalable_constrained_bo (the tolerances, the constrained-TS structure)
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025. https://github.com/PriorLabs/TabPFN (the surrogate)
The GP -> TabPFN substitution and the independent-marginal constrained Thompson sampling are BoCoDe's, with no paper behind them.
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
from .._tfm_utils import TabPFNSurrogate
from .scbo import _best_index, _gaussian_copula, _is_success, _tr


def _constrained_thompson_sample(
    surrogate: TabPFNSurrogate,
    train_X: torch.Tensor,
    Yo_t: torch.Tensor,
    Yc_t: torch.Tensor,
    cand: torch.Tensor,
) -> int:
    """SCBO's constrained Thompson sampling, on TabPFN's bar distribution.

    ``Yo_t`` is the copula-warped objective (n, 1) and ``Yc_t`` the bilog-warped constraints
    (n, nc); both warps are strictly increasing and bilog fixes zero, so feasibility and the
    objective ranking can be read straight off the sampled values.

    Returns the index into ``cand`` of the chosen point: the best sampled objective among the
    candidates whose sampled constraints are all ``<= 0``, or -- if no candidate is sampled
    feasible -- the one with the smallest total sampled violation (SCBO's fallback).
    """
    n_ctx, n_cand = train_X.shape[0], cand.shape[0]
    m = 1 + Yc_t.shape[1]  # objective + constraints, as TabPFN batch columns

    # Same X for every column; one target column per output.
    X_full = (
        torch.cat([train_X, cand], dim=0).unsqueeze(1).expand(-1, m, -1).contiguous()
    )
    Y_cols = torch.cat([Yo_t, Yc_t], dim=1)  # (n_ctx, m)
    Y_full = torch.cat([Y_cols, torch.zeros(n_cand, m, dtype=DTYPE)], dim=0).unsqueeze(
        -1
    )  # (n_ctx + n_cand, m, 1)

    with torch.no_grad():
        logits = surrogate.forward(X_full, Y_full, single_eval_pos=n_ctx)
        draw = surrogate.predict_sample(logits).to("cpu")  # (n_cand, m)

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
    """TFM-SCBO. ``n_init`` defaults to the dim-scaled BoCoDe default.

    The reported value is the best **feasible** objective so far (``-inf`` until a feasible
    point is found), as in :mod:`algorithms.single_obj_constrained.scbo`. The full evaluation
    history -- across restarts, with the constraint values -- is saved on the ``Result``.
    """
    set_seed(seed)
    obj = ProblemObjective(problem)
    nc = obj.num_constraints
    assert nc > 0, "tfm_scbo requires a constrained problem"
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "tfm_scbo",
        type(problem).__name__,
        seed,
        acquisition_function="TabPFN bar-distribution constrained Thompson sampling",
    )
    n_cand = min(
        5000, max(2000, 200 * dim)
    )  # TuRBO's / SCBO Appendix C's candidate count
    surrogate = TabPFNSurrogate(device=device)
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
