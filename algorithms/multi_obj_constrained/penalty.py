"""Penalty: the constraint-handling baseline for constrained multi-objective problems.

Fold the constraints into **every objective** and run an *unconstrained* multi-objective
optimizer. Here that optimizer is BoCoDe's stock qNEHVI loop (independent ``SingleTaskGP`` per
objective + ``qLogNoisyExpectedHypervolumeImprovement``, i.e. exactly
:mod:`algorithms.multi_obj.qnehvi`), so the only thing separating this baseline from
``constrained_qnehvi`` is the constraint handling itself. That is the point of the baseline.

In BoCoDe's maximization frame, with constraints feasible when ``c <= 0``, objective ``j`` is
replaced by::

    F_j(x) = obj_norm_j(x) - rho * viol_norm(x),
    viol(x) = sum_i max(0, c_i(x))                     (total violation, 0 iff feasible)

This is :class:`bocode.PenalizedProblem`'s scheme (``bocode/transforms.py``) applied to each
objective in turn -- the same normalized static exterior penalty, the same ``rho``, the same
"normalization constants come from a fixed 256-point Latin hypercube drawn once, not from this
run's data" rule, so the penalized problem is a property of the problem and not of the
seed. ``PenalizedProblem`` itself refuses multi-objective bases (it is a scalarizer's input),
which is why the recipe is applied here rather than by wrapping the problem in it.

Every feasible point has ``viol = 0`` and therefore ``F = obj_norm``, an increasing affine map
of the raw objective; the penalty only ever pushes *infeasible* points down, in every
objective at once, which is what makes their hypervolume contribution vanish. See
``single_obj_constrained/penalty.py`` for the penalty-coefficient literature (Coello Coello's
survey; Nocedal & Wright Ch. 17; Tessema & Yen's normalization; Deb's parameter-free
alternative) and for why ``rho = 1`` is the scale-free default and a **heuristic, not a result**
-- it is exposed as ``--rho`` so it can be swept.

Reporting
---------
The GPs and the acquisition see the penalized objectives; the *reported* metric is the
**feasible hypervolume of the raw objectives** against the problem's own fixed ``ref_point`` --
byte-for-byte the metric ``constrained_qnehvi`` reports, so the two are directly comparable.
The acquisition's reference point is the problem's ``ref_point`` pushed through the same
per-objective normalization (an increasing affine map, so it defines the same region).

Run::

    python -m algorithms.multi_obj_constrained.penalty --problem WeldedBeam --init 12 --iters 50

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195 (the unconstrained loop this baseline runs)
C. A. Coello Coello. Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: a survey of the state of the art. Computer Methods in Applied Mechanics and Engineering 191(11-12):1245-1287, 2002.
B. Tessema and G. G. Yen. An adaptive penalty formulation for constrained evolutionary optimization. IEEE Trans. Systems, Man, and Cybernetics - A 39(3):565-578, 2009.
Applying the penalty to each objective (rather than to a scalarization) is BoCoDe's choice; the coefficient is a heuristic, not a value from a paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

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
from ..multi_obj.qnehvi import _fit

DEFAULT_RHO = (
    1.0  # dimensionless: both terms are min-max normalized. See the docstring.
)
N_SAMPLE = 256  # PenalizedProblem's construction-time sample size
_EPS = 1e-12  # PenalizedProblem's guard against a degenerate (lo == hi) range


class _Penalty:
    """PenalizedProblem's normalized static penalty, applied to every objective.

    The normalization constants come from a fixed Latin-hypercube sample of the problem drawn
    at construction (``seed=0``), exactly as ``bocode.PenalizedProblem`` does, so they do not
    depend on the run.
    """

    def __init__(self, problem, weight: float):
        self.weight = float(weight)
        X = problem.sample(N_SAMPLE, seed=0)
        values, constraints = problem.evaluate(X)
        values = values.to(DTYPE)
        viol = constraints.to(DTYPE).clamp(min=0).sum(dim=1, keepdim=True)
        finite = torch.isfinite(values).all(dim=1)
        v = values[finite] if finite.any() else values
        self.lo, self.hi = v.min(dim=0).values, v.max(dim=0).values
        self.viol_hi = viol[torch.isfinite(viol).all(dim=1)].max()

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Min-max normalize the objectives (increasing affine, so it preserves dominance)."""
        return (values - self.lo) / (self.hi - self.lo + _EPS)

    def __call__(self, values: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        viol = constraints.clamp(min=0).sum(dim=1, keepdim=True)
        return self.normalize(values) - self.weight * (viol / (self.viol_hi + _EPS))


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    rho: float = DEFAULT_RHO,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Penalty baseline: qNEHVI on the per-objective penalized objectives.

    ``n_init`` defaults to the dim-scaled BoCoDe default. Reports the feasible hypervolume of
    the **raw** objectives against the problem's fixed ``ref_point``. Resumable via
    ``checkpoint``.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = MultiObjectiveProblem(problem)
    assert obj.num_constraints > 0, "penalty requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "penalty",
        type(problem).__name__,
        seed,
        acquisition_function=f"qLogNEHVI on penalized objectives (rho={rho})",
    )
    penalty = _Penalty(problem, rho)

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, data = load_checkpoint(checkpoint, res)
        train_C = torch.tensor(data["C"], dtype=DTYPE)
        ref_point = torch.tensor(data["ref_point"], dtype=DTYPE)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_Y, train_C = obj.evaluate_raw(train_X)
        ref_point = obj.hv_ref_point(train_Y)
        start_it = 0

    def feasible_hv(Y, C):
        """The reported metric: hypervolume of the FEASIBLE raw objectives (as in
        ``constrained_qnehvi``), so this baseline is comparable to the constrained methods."""
        feas = (C <= 0).all(dim=1)
        if not feas.any():
            return 0.0
        return (
            DominatedPartitioning(ref_point=ref_point.cpu(), Y=Y[feas].cpu())
            .compute_hypervolume()
            .item()
        )

    if start_it == 0:
        res.start(feasible_hv(train_Y, train_C))

    bounds_dev = obj.bounds.to(dev)
    # The acquisition lives in penalized space, so its reference point is the problem's own
    # ref_point pushed through the same normalization (increasing affine: same region).
    ref_pen = penalty.normalize(ref_point.reshape(1, -1)).reshape(-1).to(dev)
    for it in range(start_it, iters):
        train_P = penalty(train_Y, train_C)
        model = _fit(train_X.to(dev), train_P.to(dev))
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_pen,
            X_baseline=train_X.to(dev),
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
        y, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        train_C = torch.cat([train_C, c], dim=0)
        res.record(feasible_hv(train_Y, train_C))
        if checkpoint:
            res.set_history(train_X, train_Y, n_init, c=train_C)
            save_checkpoint(
                checkpoint,
                train_X,
                train_Y,  # the RAW objectives; the penalized targets are recomputed on resume
                res,
                it + 1,
                extra={
                    "C": train_C.cpu().numpy(),
                    "ref_point": ref_point.cpu().numpy(),
                },
            )
    res.set_history(train_X, train_Y, n_init, c=train_C)
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
        help="penalty coefficient on the NORMALIZED total violation (default: 1.0). "
        "A heuristic -- sweep it; see the module docstring.",
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
