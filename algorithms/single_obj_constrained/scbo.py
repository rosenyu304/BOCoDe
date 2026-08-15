"""SCBO: Scalable Constrained Bayesian Optimization (Eriksson & Poloczek, 2021).

Constrained TuRBO: a trust region is maintained around the best *feasible* point (or,
before any feasible point is found, around the point of minimum total violation). The
objective and each constraint get their own GP; candidates inside the trust region are
scored by *constrained* Thompson sampling -- draw one posterior realization of the
objective and of every constraint, then take the candidate with the best objective draw
among those whose draws are feasible, falling back to the smallest total predicted
violation when no candidate is predicted feasible.

The trust-region constants follow the paper's Appendix C ("Details on SCBO") --
``L_min = 2^-7``, ``L_max = 1.6``, ``L_init = 0.8``, perturbation probability
``p_perturb = min(1, 20/d)`` -- **except the two tolerances, where we follow the BoTorch
SCBO tutorial** (Rosen's call, 2026-07-13): ``tau_s = 10`` (the paper says 3) and
``tau_f = ceil(max(4/q, d/q))`` (the paper says ``ceil(d/q)``; the two differ only for
``d < 4``). The candidate set is a scrambled Sobol sequence inside the trust region, with
each coordinate taken from Sobol with probability ``p_perturb`` and from the
trust-region center otherwise.

The paper's two output warpings (Sec. 3, "Transformations of Objective and Constraints")
are applied: a **Gaussian copula** on the objective (rank -> empirical CDF -> inverse
Gaussian CDF, which magnifies differences at the ends of the observed range where the
optima live) and the **bilog** transform ``sgn(y) log(1 + |y|)`` on each constraint
(which magnifies the range around zero, where the change of sign decides feasibility).
The paper notes these are "advantageous over a naive standardization of each function as
the latter is insensitive to the areas of interest".

Run::

    python -m algorithms.single_obj_constrained.scbo --problem PressureVessel --iters 80

Sources:
D. Eriksson and M. Poloczek. Scalable Constrained Bayesian Optimization. AISTATS 2021. https://arxiv.org/abs/2002.08526
BoTorch SCBO tutorial: https://botorch.org/docs/tutorials/scalable_constrained_bo
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Bilog
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
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
    robust_fit_mll,
    save_checkpoint,
    set_seed,
)
from ..single_obj.turbo import TrustRegion

SUCCESS_TOLERANCE = 3  # SCBO paper, Appendix C (Rosen's call, 2026-07-14)


def _tr(dim: int) -> TrustRegion:
    """SCBO's trust region: TuRBO's, with the SCBO PAPER's tolerances.

    ``failure_tolerance = ceil(max(4/q, dim/q))`` comes from ``TrustRegion`` itself and
    matches the BoTorch tutorial's ``ScboState.__post_init__``.

    ``success_tolerance = 3`` is the SCBO **paper's** value (Appendix C, and TuRBO's), NOT
    the BoTorch tutorial's 10. The tutorial and the paper disagree here; Rosen chose the
    paper (2026-07-14). A larger success tolerance makes the trust region slower to EXPAND,
    so the tutorial's 10 is the more conservative/exploitative setting -- the difference is
    real, not cosmetic, which is why it is pinned here rather than left to a default.
    """
    return TrustRegion(dim=dim, success_tolerance=SUCCESS_TOLERANCE)


def _gaussian_copula(Y: torch.Tensor) -> torch.Tensor:
    """Gaussian copula transform of the objective (SCBO Sec. 3, Fig. 1 right).

    Maps observations to quantiles through the empirical CDF, then through the inverse
    Gaussian CDF. Strictly monotone, so it preserves the ordering Thompson sampling
    needs while magnifying differences at the ends of the observed range.
    """
    n = Y.shape[0]
    ranks = torch.argsort(torch.argsort(Y, dim=0), dim=0).to(Y) + 1.0
    q = (ranks - 0.5) / n  # plotting position; keeps the inverse CDF finite
    return torch.erfinv(2.0 * q - 1.0) * math.sqrt(2.0)


def _fit_gp(X: torch.Tensor, Y: torch.Tensor, bilog: bool) -> SingleTaskGP:
    """Fit SCBO's GP: the TuRBO Matern-5/2 ARD prior with SCBO's output warping.

    Lengthscale and noise constraints are the ones the reference SCBO implementation
    uses (BoTorch tutorial ``get_fitted_model``). ``bilog=True`` fits a constraint model
    in bilog space; its posterior is untransformed back to the original scale, so
    ``c <= 0`` keeps its meaning for the feasibility test.
    """
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=X.shape[-1],
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Bilog() if bilog else None,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # A GP fit can genuinely fail on a pathological output -- e.g. a constraint that is
    # (nearly) a step function, which no smooth GP can represent, raising a ModelFittingError
    # or driving a hyperparameter outside its prior's support (a bare ValueError). Crashing the
    # whole run loses every iteration already computed, and on a cluster it looks like an
    # infrastructure failure when it is really a property of the data. robust_fit_mll falls
    # back to the model's PRIOR hyperparameters (an un-optimised but perfectly valid GP). SCBO's
    # trust region then keeps shrinking on that output, which is the correct behaviour for a
    # coordinate it cannot model.
    #
    # NB: this is a robustness guard, NOT a fix for a broken benchmark. The real cause of such an
    # output is almost always a defect in the PROBLEM (see the 0.0625 gauge-snap bug that turned
    # CEC2020_p20's constraints into 3-valued step functions). Use experiments/problem_audit.py
    # to find those; do not let this guard hide them.
    robust_fit_mll(mll, label="scbo")
    return model


def _best_index(Yo: torch.Tensor, Yc: torch.Tensor) -> int:
    """Index of the trust-region center: best feasible point, else least-violating."""
    feas = (Yc <= 0).all(dim=-1)
    if feas.any():
        score = Yo.clone().reshape(-1)
        score[~feas] = -float("inf")
        return int(score.argmax())
    return int(Yc.clamp(min=0).sum(dim=-1).argmin())


def _is_success(y_next, c_next, best_value: float, best_con: torch.Tensor) -> bool:
    """SCBO's constrained improvement rule (BoTorch tutorial ``update_state``).

    Points are compared by objective only when both are feasible; a feasible point
    always beats an infeasible one; two infeasible points are compared by total
    violation -- so *reducing* the violation counts as a success and grows the region.
    """
    if (c_next <= 0).all():
        if (best_con > 0).any():
            return True  # first feasible point beats any infeasible incumbent
        return bool(y_next > best_value + 1e-3 * math.fabs(best_value))
    return bool(c_next.clamp(min=0).sum() < best_con.clamp(min=0).sum())


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 80,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """SCBO. ``n_init`` defaults to the dim-scaled BoCoDe default.

    GP fits + Thompson sampling run on ``device`` (default cuda when available); the
    objective/constraints are evaluated on CPU. With ``checkpoint`` set the run is
    resumable: it restores the data, the trust-region state and the RNG, and saves
    after every iteration.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    assert obj.num_constraints > 0, "scbo requires a constrained problem"
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "scbo", type(problem).__name__, seed, acquisition_function="Thompson sampling"
    )
    # SCBO Appendix C: candidate set size follows TuRBO.
    n_cand = min(5000, max(2000, 200 * dim))

    def start_region(offset: int, budget: int | None = None):
        """Fresh trust region on a fresh initial design (TuRBO/SCBO restart).

        ``budget`` caps how many points may be EVALUATED, so a restart can never spend more
        objective evaluations than the run has left. Without this cap the final restart still
        overran the budget (it evaluated all n_init points before the caller could stop it).
        """
        n = n_init if budget is None else max(0, min(n_init, budget))
        X = initial_design(n_init, dim, seed + offset)[:n]
        Yo, Yc = obj.evaluate_raw(X)
        i = _best_index(Yo, Yc)
        return X, Yo, Yc, _tr(dim), Yo[i].item(), Yc[i].clone()

    def best_feasible(Yo, Yc):
        feas = (Yc <= 0).all(dim=-1)
        return Yo[feas].max().item() if feas.any() else -float("inf")

    if checkpoint and Path(checkpoint).exists():
        X, Yo, start_it, data = load_checkpoint(checkpoint, res)
        Yc = torch.tensor(data["con"], dtype=DTYPE)
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
        n_restarts = 0
        best = best_feasible(Yo, Yc)
        res.start(best)
        start_it = 0
    # X/Yo/Yc are the TRUST-REGION-LOCAL training data: SCBO Sec. 3.1 step 2d says to DISCARD
    # them on a restart, and that is correct. But the EVALUATION HISTORY must never be discarded,
    # and restart designs must be CHARGED to the budget. Keeping only the TR data caused scbo to
    # (a) spend n_restarts * n_init objective evaluations for FREE -- 100 evaluations against a
    # budget of 80 on PressureVessel, a 25% larger budget than every method it is compared with --
    # and (b) save only the LAST region's rows (33 of 100), which silently defeats the
    # total-evaluation axis in _eval_utils (it would read 33 evaluations, not 100). Mirror
    # turbo.py, which gets this right.
    hist_X, hist_Yo, hist_Yc = X.clone(), Yo.clone(), Yc.clone()
    evals = 0  # BO evaluations charged against `iters` (the initial design is separate)

    for it in range(start_it, iters):
        # SCBO applies the copula to the objective and bilog to every constraint.
        obj_model = _fit_gp(X.to(dev), _gaussian_copula(Yo).to(dev), bilog=False)
        con_model = ModelListGP(
            *[
                _fit_gp(X.to(dev), Yc[:, i : i + 1].to(dev), bilog=True)
                for i in range(obj.num_constraints)
            ]
        )

        # Trust region around the best feasible (or least-violating) point.
        x_center = X[_best_index(Yo, Yc)].clone()
        tr_lb = torch.clamp(x_center - tr.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + tr.length / 2.0, 0.0, 1.0)

        sobol = SobolEngine(dim, scramble=True, seed=seed + it)
        pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_cand).to(DTYPE)
        # Perturbation mask: take the Sobol value with probability p_perturb, the
        # center's value otherwise (SCBO Appendix C).
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_cand, dim, dtype=DTYPE) <= prob_perturb
        empty = torch.where(mask.sum(dim=1) == 0)[0]
        mask[empty, torch.randint(0, dim, (len(empty),))] = True
        cand = x_center.expand(n_cand, dim).clone()
        cand[mask] = pert[mask]

        ts = ConstrainedMaxPosteriorSampling(
            model=obj_model, constraint_model=con_model, replacement=False
        )
        with torch.no_grad():
            x_new = ts(cand.to(dev), num_samples=1).detach().to("cpu", DTYPE)

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
        evals += 1
        if evals >= iters:
            break

        if tr.restart:
            # L < L_min: DISCARD THE LOCAL DATA and start a new trust region on a fresh initial
            # design (SCBO Sec. 3.1 step 2d / Theorem 1). The local data is discarded -- the
            # EVALUATION HISTORY is not, and the fresh design is CHARGED to the budget.
            n_restarts += 1
            remaining = iters - evals
            if remaining <= 0:
                break
            X, Yo, Yc, tr, best_value, best_con = start_region(
                n_restarts * 1000, budget=remaining
            )
            hist_X = torch.cat([hist_X, X], dim=0)
            hist_Yo = torch.cat([hist_Yo, Yo], dim=0)
            hist_Yc = torch.cat([hist_Yc, Yc], dim=0)
            for j in range(X.shape[0]):
                best = max(best, best_feasible(Yo[j : j + 1], Yc[j : j + 1]))
                res.record(best)
                evals += 1
                if evals >= iters:
                    break
            if evals >= iters:
                break

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
