"""SCBO: Scalable Constrained Bayesian Optimization (Eriksson & Poloczek, 2021).

Constrained TuRBO: a trust region is maintained around the best *feasible* point
(or the least-infeasible point before any feasible one is found). The objective
and each constraint are modeled with independent GPs; candidates inside the trust
region are scored by Thompson sampling, picking the candidate with the best
posterior objective sample among those predicted feasible (falling back to the
smallest total predicted violation when none are feasible). Trust-region lengths
adapt on success/failure as in TuRBO.

Run::

    python -m algorithms.single_obj_constrained.scbo --problem PressureVessel --init 10 --iters 80

Sources:
D. Eriksson and M. Poloczek. Scalable Constrained Bayesian Optimization. AISTATS 2021. https://arxiv.org/abs/2002.08526
BoTorch SCBO tutorial: https://botorch.org/docs/tutorials/scalable_constrained_bo
"""

from __future__ import annotations

import argparse

import torch
from torch.quasirandom import SobolEngine

import bocode

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    fit_gp,
    initial_design,  # noqa: E501
    set_seed,
)
from ..single_obj.turbo import TrustRegion


def _feasible_mask(con: torch.Tensor) -> torch.Tensor:
    return (con <= 0).all(dim=1)


def _best_feasible(obj: torch.Tensor, con: torch.Tensor):
    """Return (best feasible objective, index) or (-inf, least-violating index)."""
    feas = _feasible_mask(con)
    if feas.any():
        idx = torch.where(feas)[0]
        best = idx[obj[idx].argmax()]
        return obj[best].item(), best.item()
    violation = con.clamp(min=0).sum(dim=1)
    return -float("inf"), int(violation.argmin())


def optimize_problem(problem, n_init: int = 10, iters: int = 80, seed: int = 0) -> Result:
    set_seed(seed)
    obj = ProblemObjective(problem)
    assert obj.num_constraints > 0, "scbo requires a constrained problem"
    res = Result("scbo", type(problem).__name__, seed)

    X = initial_design(n_init, obj.dim, seed)
    Yo, Yc = obj.evaluate_raw(X)
    best, _ = _best_feasible(Yo, Yc)
    for _ in range(n_init):
        res.log(best)

    tr = TrustRegion(dim=obj.dim)
    for it in range(iters):
        obj_model = fit_gp(X, Yo)
        con_models = [fit_gp(X, Yc[:, i : i + 1]) for i in range(obj.num_constraints)]

        # center on best feasible (or least-violating) point
        _, center_idx = _best_feasible(Yo, Yc)
        x_center = X[center_idx]
        lb = torch.clamp(x_center - tr.length / 2.0, 0.0, 1.0)
        ub = torch.clamp(x_center + tr.length / 2.0, 0.0, 1.0)

        n_cand = min(5000, max(2000, 200 * obj.dim))
        sobol = SobolEngine(obj.dim, scramble=True, seed=seed + it)
        cand = lb + (ub - lb) * sobol.draw(n_cand).to(DTYPE)

        # Thompson sampling: one posterior sample of objective and each constraint
        with torch.no_grad():
            obj_s = obj_model.posterior(cand).rsample().squeeze(0).squeeze(-1)
            con_s = torch.stack(
                [m.posterior(cand).rsample().squeeze(0).squeeze(-1) for m in con_models],
                dim=-1,
            )
        pred_feas = (con_s <= 0).all(dim=1)
        if pred_feas.any():
            pool = torch.where(pred_feas)[0]
            choice = pool[obj_s[pool].argmax()]
        else:
            choice = con_s.clamp(min=0).sum(dim=1).argmin()
        x_new = cand[choice : choice + 1]

        o, c = obj.evaluate_raw(x_new)
        prev_best = best
        X = torch.cat([X, x_new], dim=0)
        Yo = torch.cat([Yo, o], dim=0)
        Yc = torch.cat([Yc, c], dim=0)
        best, _ = _best_feasible(Yo, Yc)
        # "success" = strictly improved the best feasible objective
        improved = best > prev_best + 1e-9 if prev_best != -float("inf") else _feasible_mask(c).any().item()
        tr.update(bool(improved))
        res.log(best)
        if tr.restart:
            tr = TrustRegion(dim=obj.dim)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=10)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = optimize_problem(bocode.get_problem(args.problem)(), args.init, args.iters, args.seed)
    print(f"{res.algorithm} on {res.problem}: best feasible={res.best:.6g} after {len(res.best_history)} evals")


if __name__ == "__main__":
    main()
