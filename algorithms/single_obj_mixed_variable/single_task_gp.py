"""Mixed-variable SingleTaskGP Bayesian optimization (optimize_acqf_mixed).

The basic BoTorch mixed-variable baseline: fit a ``SingleTaskGP`` and optimize
LogExpectedImprovement with ``optimize_acqf_mixed``, which enumerates the discrete
configurations (integer / categorical dimensions) and runs continuous acquisition
optimization for each. The discrete grid is read from the problem's
``variable_types`` (integer ranges and explicit allowed-value lists), mapped into
the unit cube. When the number of discrete configurations is large (the product of
the per-dimension cardinalities exceeds ``MAX_DISCRETE_CONFIGS``), the method falls
back to continuous-relaxation acquisition optimization followed by snapping the
proposal to the nearest valid value (which the harness also enforces at evaluation).

Constraints (if the problem has any) are folded into a penalized objective, as in
the other single-objective baselines.

Run::

    python -m algorithms.single_obj_mixed_variable.single_task_gp --problem PressureVessel --init 10 --iters 40
    python -m algorithms.single_obj_mixed_variable.single_task_gp --problem SpeedReducer --discrete --iters 40

Sources:
BoTorch optimize_acqf_mixed: https://github.com/meta-pytorch/botorch/blob/main/botorch/optim/optimize.py
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse
import itertools
import math

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf, optimize_acqf_mixed

from .._bo_utils import (
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    fit_gp,
    gp_stats,
    initial_design,
    make_problem,
    set_seed,
)

#: Above this many discrete configurations, fall back to relaxation + snapping
#: (optimize_acqf_mixed enumerates every configuration, so it is only practical
#: for a modest discrete grid).
MAX_DISCRETE_CONFIGS = 256


def _discrete_grids(problem) -> dict[int, list[float]]:
    """Per-dimension allowed values in unit-cube coordinates, for non-continuous dims."""
    types = problem.resolved_variable_types()
    bounds = problem.bounds
    grids: dict[int, list[float]] = {}
    for i, t in enumerate(types):
        # bounds may be stored as (lower, upper) or (upper, lower); the unit-cube
        # map (v - b0) / (b1 - b0) inverts scale() either way, but the integer
        # enumeration needs the true low/high.
        b0, b1 = float(bounds[i][0]), float(bounds[i][1])
        span = (b1 - b0) or 1.0
        if t == "continuous":
            continue
        if t == "integer":
            values = range(math.ceil(min(b0, b1)), math.floor(max(b0, b1)) + 1)
        else:  # explicit list of allowed values
            values = t
        grids[i] = [min(max((float(v) - b0) / span, 0.0), 1.0) for v in values]
    return grids


def _fixed_features_list(grids: dict[int, list[float]]):
    """Cartesian product of the discrete grids as a fixed_features_list, or None."""
    if not grids:
        return None
    dims = sorted(grids)
    total = math.prod(len(grids[d]) for d in dims)
    if total > MAX_DISCRETE_CONFIGS:
        return None
    return [
        dict(zip(dims, combo, strict=True))
        for combo in itertools.product(*(grids[d] for d in dims))
    ]


def _snap(X_unit: torch.Tensor, grids: dict[int, list[float]]) -> torch.Tensor:
    """Snap the discrete dimensions of ``X_unit`` to their nearest allowed values."""
    X = X_unit.clone()
    for d, vals in grids.items():
        if not vals:
            continue
        v = torch.tensor(vals, dtype=X.dtype)
        X[:, d] = v[(X[:, d].unsqueeze(-1) - v).abs().argmin(dim=-1)]
    return X


def optimize_problem(
    problem, n_init: int = 10, iters: int = 40, seed: int = 0
) -> Result:
    """Mixed-variable SingleTaskGP + LogEI BO over the unit cube."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    res = Result(
        "single_task_gp_mixed",
        type(problem).__name__,
        seed,
        acquisition_function="LogEI",
    )

    grids = _discrete_grids(problem)
    fixed_features = _fixed_features_list(grids)

    train_X = _snap(initial_design(n_init, obj.dim, seed), grids)
    train_Y = obj(train_X)
    best = train_Y.max().item()
    res.start(best)

    for _ in range(iters):
        model = fit_gp(train_X, train_Y)
        acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
        if fixed_features is not None:
            candidate, acq_value = optimize_acqf_mixed(
                acqf,
                bounds=obj.bounds,
                q=1,
                num_restarts=4,
                raw_samples=128,
                fixed_features_list=fixed_features,
            )
        else:
            # discrete grid too large to enumerate: relax, then snap to valid values
            candidate, acq_value = optimize_acqf(
                acqf, bounds=obj.bounds, q=1, num_restarts=10, raw_samples=512
            )
            candidate = _snap(candidate, grids)
        mean, var = gp_stats(model, candidate)
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(best, mean=mean, variance=var, acq_value=acq_value.item())
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args), args.init, args.iters, args.seed
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
