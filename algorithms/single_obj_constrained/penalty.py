"""Penalty: the constraint-handling baseline -- optimize a penalized scalar objective.

The oldest and simplest way to handle constraints: fold them into the objective and run an
*unconstrained* optimizer. Here that optimizer is BoCoDe's stock single-objective GP loop
(``SingleTaskGP`` + ``LogExpectedImprovement`` + ``optimize_acqf``, i.e. exactly
:mod:`algorithms.single_obj.single_task_gp`), so the only thing separating this baseline from
a *constrained* method like ``constrained_ei`` / ``scbo`` is the constraint handling itself.
That is the point of the baseline: it isolates what modeling the constraints buys you.

The penalized objective is BoCoDe's existing :class:`bocode.PenalizedProblem` transform
(``bocode/transforms.py``); this file does not reinvent it. In BoCoDe's maximization frame,
with constraints feasible when ``c <= 0``::

    F(x) = obj_norm(x) - rho * viol_norm(x),
    viol(x) = sum_i max(0, c_i(x))                      (total violation, 0 iff feasible)

where ``obj`` and ``viol`` are each **min-max normalized** using a fixed 256-point Latin
hypercube drawn at construction time (so the scaling is a property of the problem, not of the
run: every algorithm and every seed sees the same ``F``).

The penalty coefficient (``--rho``)
-----------------------------------
This is a **static (non-adaptive) exterior penalty** in the taxonomy of Coello Coello's survey
(death / static / dynamic / adaptive / co-evolutionary). The survey's central practical point
is the one that matters here: a static penalty's coefficient *has to be set by the user*,
there is no value that is right for every problem, and the method is notoriously sensitive to
it -- too small and the optimum of ``F`` sits in the infeasible region, too large and the
search is driven to feasibility before it has learned anything about the objective (the
classic "minimum penalty rule": use the smallest coefficient that still keeps the penalized
optimum feasible). The same threshold behaviour is the exact-penalty theory of nonlinear
programming: an l1 penalty reproduces the constrained optimum only once its coefficient
exceeds the largest optimal Lagrange multiplier (Nocedal & Wright, Ch. 17) -- a quantity we do
not know in advance for a black box.

So there is **no principled value to read off a paper**, and BoCoDe does the next best thing,
which is what the normalized-penalty literature (Tessema & Yen; the CEC2020 real-world suite's
normalized mean violation) recommends: **normalize both terms to a common scale first, so that
the coefficient is dimensionless and ``rho = 1`` is a meaningful, scale-free default** -- at
``rho = 1`` a full-range constraint violation costs exactly a full range of the objective. That
is `PenalizedProblem`'s own default and is what we use. **It is a heuristic, not a result**;
it is exposed as ``--rho`` precisely so it can be swept, and the sweep is the honest way to
report this baseline. Two well-known alternatives we deliberately do *not* implement (they are
different constraint-handling methods, not different coefficients): Deb's parameter-free
feasibility rule (any feasible point beats any infeasible one, infeasible points ranked by
violation), and adaptive penalties that grow ``rho`` with the iteration count or with the
observed feasible fraction.

Reporting
---------
The trace is the best **feasible** raw objective so far (``-inf`` until a feasible point is
found) and the raw constraint values are saved with the history -- i.e. the same metric the
constrained methods report, so the baseline is directly comparable to them. The penalized
scalar is used *only* to fit the GP and drive the acquisition; it is never reported.

Run::

    python -m algorithms.single_obj_constrained.penalty --problem PressureVessel --iters 50
    python -m algorithms.single_obj_constrained.penalty --problem PressureVessel --rho 10

Sources:
C. A. Coello Coello. Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: a survey of the state of the art. Computer Methods in Applied Mechanics and Engineering 191(11-12):1245-1287, 2002. https://doi.org/10.1016/S0045-7825(01)00323-1
J. Nocedal and S. J. Wright. Numerical Optimization, 2nd ed., Ch. 17 (Penalty and Augmented Lagrangian Methods). Springer, 2006.
B. Tessema and G. G. Yen. An adaptive penalty formulation for constrained evolutionary optimization. IEEE Trans. Systems, Man, and Cybernetics - A 39(3):565-578, 2009. (normalized objective + normalized violation)
K. Deb. An efficient constraint handling method for genetic algorithms. Computer Methods in Applied Mechanics and Engineering 186(2-4):311-338, 2000. (the parameter-free alternative)
BoTorch getting started (the unconstrained loop this baseline runs): https://botorch.org/docs/getting_started
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from bocode.transforms import PenalizedProblem

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    fit_gp,
    gp_stats,
    initial_design,
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)

DEFAULT_RHO = (
    1.0  # dimensionless: both terms are min-max normalized. See the docstring.
)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    rho: float = DEFAULT_RHO,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Penalty baseline: SingleTaskGP + LogEI on ``PenalizedProblem(problem, weight=rho)``.

    ``n_init`` defaults to the dim-scaled BoCoDe default. The GP fit and the acquisition
    optimization run on ``device``; the problem is evaluated on CPU. Resumable via
    ``checkpoint``.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    assert obj.num_constraints > 0, "penalty requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "penalty",
        type(problem).__name__,
        seed,
        acquisition_function=f"LogEI on penalized objective (rho={rho})",
    )

    # The penalty transform's min-max normalisation is calibrated on the INITIAL DESIGN, which the
    # run has already paid for. It used to be calibrated on a private 256-point sample drawn inside
    # PenalizedProblem.__init__ -- 256 REAL objective evaluations, entirely OFF-BUDGET. A penalty
    # run given 25 iterations therefore spent 281 evaluations: an 11x larger budget than every
    # method it was compared against. (That is a comparison-invalidating bug, not a tuning choice.)
    # Calibrating on the initial design costs nothing extra and keeps every method on the same
    # n_init + iters budget. It does make the transform depend on the seed's initial design, which
    # the previous scheme deliberately avoided -- an acceptable price for a valid comparison.
    pen: PenalizedProblem | None = None

    def evaluate(X_unit: torch.Tensor):
        """Raw objective + constraints (for the metric) and the penalized scalar (for the GP).

        The base problem is evaluated **once**; ``pen.penalize`` reuses that evaluation.
        """
        values, constraints = obj.evaluate_raw(X_unit)
        return values, constraints, pen.penalize(values, constraints).to(DTYPE)

    def best_feasible(values, constraints):
        feas = (constraints <= 0).all(dim=1)
        return values[feas].max().item() if feas.any() else -float("inf")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_P, start_it, data = load_checkpoint(checkpoint, res)
        train_obj = torch.tensor(data["obj"], dtype=DTYPE)
        train_con = torch.tensor(data["con"], dtype=DTYPE)
        # rebuild the transform from the SAME initial design, so a resumed run is identical
        pen = PenalizedProblem(
            problem,
            weight=rho,
            calibration=(train_obj[:n_init], train_con[:n_init]),
        )
        best = best_feasible(train_obj, train_con)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        # evaluate the initial design ONCE (this is the budgeted n_init), then calibrate the
        # penalty transform on it -- no extra objective evaluations are spent.
        init_vals, init_cons = obj.evaluate_raw(train_X)
        pen = PenalizedProblem(problem, weight=rho, calibration=(init_vals, init_cons))
        train_obj, train_con = init_vals, init_cons
        train_P = pen.penalize(init_vals, init_cons).to(DTYPE)
        best = best_feasible(train_obj, train_con)
        res.start(best)
        start_it = 0

    bounds_dev = obj.bounds.to(dev)
    for it in range(start_it, iters):
        model = fit_gp(train_X.to(dev), train_P.to(dev))
        acqf = LogExpectedImprovement(model=model, best_f=train_P.to(dev).max())
        candidate, acq_value = optimize_acqf(
            acqf, bounds=bounds_dev, q=1, num_restarts=10, raw_samples=512
        )
        mean, var = gp_stats(model, candidate)
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)

        o, c, p = evaluate(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_obj = torch.cat([train_obj, o], dim=0)
        train_con = torch.cat([train_con, c], dim=0)
        train_P = torch.cat([train_P, p], dim=0)
        best = max(best, best_feasible(o, c))
        res.record(best, mean=mean, variance=var, acq_value=acq_value.item())

        if checkpoint:
            res.set_history(train_X, train_obj, n_init, c=train_con)
            save_checkpoint(
                checkpoint,
                train_X,
                train_P,  # the GP's targets: what a resumed run must refit on
                res,
                it + 1,
                extra={
                    "obj": train_obj.cpu().numpy(),
                    "con": train_con.cpu().numpy(),
                },
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
        "--rho",
        type=float,
        default=DEFAULT_RHO,
        help="penalty coefficient on the NORMALIZED total violation (default: 1.0, "
        "i.e. a full-range violation costs a full range of the objective). A heuristic "
        "-- sweep it; see the module docstring.",
    )
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
        rho=args.rho,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
