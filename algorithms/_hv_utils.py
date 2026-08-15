"""Fast exact hypervolume traces.

The multi-objective baselines used to build their HV trace by calling BoTorch's
``DominatedPartitioning(ref_point, Y[:i]).compute_hypervolume()`` once per
iteration — recomputing the whole box decomposition from scratch, ``n`` times.
Exact HV is #P-hard in the number of objectives, so that cost explodes with ``m``.
Measured on one 1,020-point run (single HV call):

    m=3 (RE33)                    30 ms
    m=4 (BotorchCarSideImpact)    74 ms
    m=6 (RE61)                   698 ms

so RE61 cost ~340 s *per seed* — ~700x a single-objective problem, and it starved
tables T3/T4 out of the campaign entirely.

Two fixes, both exact (no approximation):

1. **pymoo's compiled-C WFG hypervolume** instead of BoTorch's box decomposition.
   Numerically identical (agreement ~1e-16) and much faster:
   ``m=3: 163x``, ``m=4: 224x``, ``m=6: 9.5x``.

2. **Only recompute when the Pareto front actually changes.** The HV trace is a
   step function: a newly drawn point that is dominated cannot change it. Random
   search draws mostly dominated points, so most iterations become an O(m·|front|)
   dominance test instead of a full HV evaluation.

Note on hardware: the GPU is *not* an option here. ``torch.cuda.is_available()`` is
False in the ``bocode`` env (torch is built for CUDA 13.0; the driver is 12.6), and
even with a working driver an exact HV is a branch-heavy recursive decomposition,
not a dense tensor kernel — it does not map onto CUDA. Compiled C is the right tool.

BoCoDe MAXIMIZES; pymoo MINIMIZES — every entry point here negates internally.
"""

from __future__ import annotations

import numpy as np
import torch


def hypervolume(
    Y: torch.Tensor | np.ndarray, ref_point: torch.Tensor | np.ndarray
) -> float:
    """Exact hypervolume of ``Y`` w.r.t. ``ref_point`` (both in the MAXIMIZATION frame)."""
    from pymoo.indicators.hv import HV

    Y = np.asarray(Y.detach().cpu() if torch.is_tensor(Y) else Y, dtype=float)
    r = np.asarray(
        ref_point.detach().cpu() if torch.is_tensor(ref_point) else ref_point,
        dtype=float,
    ).ravel()
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    return float(HV(ref_point=-r)(-Y))


def hypervolume_trace(
    Y: torch.Tensor | np.ndarray,
    ref_point: torch.Tensor | np.ndarray,
    feasible: torch.Tensor | np.ndarray | None = None,
) -> list[float]:
    """Cumulative exact HV after each of the ``n`` evaluations (maximization frame).

    Equivalent to calling :func:`hypervolume` on ``Y[:i]`` for every ``i``, but skips
    the recomputation whenever the new point is dominated (the HV cannot have changed).

    ``feasible``: optional boolean mask. Infeasible points never enter the front, so
    the trace is the *feasible* hypervolume and stays 0 until the first feasible point
    is found — the correct convention for constrained multi-objective problems.
    """
    from pymoo.indicators.hv import HV

    Y = np.asarray(Y.detach().cpu() if torch.is_tensor(Y) else Y, dtype=float)
    r = np.asarray(
        ref_point.detach().cpu() if torch.is_tensor(ref_point) else ref_point,
        dtype=float,
    ).ravel()
    if feasible is None:
        mask = np.ones(len(Y), dtype=bool)
    else:
        mask = (
            np.asarray(
                feasible.detach().cpu() if torch.is_tensor(feasible) else feasible
            )
            .astype(bool)
            .ravel()
        )
    ind = HV(ref_point=-r)

    trace: list[float] = []
    front: list[np.ndarray] = []
    last = 0.0
    for p, ok in zip(Y, mask, strict=True):
        if not ok:
            trace.append(last)  # infeasible: cannot contribute
            continue
        # dominated by the incumbent front => the HV is unchanged, skip the decomposition
        if any(np.all(q >= p) and np.any(q > p) for q in front):
            trace.append(last)
            continue
        front = [q for q in front if not (np.all(p >= q) and np.any(p > q))]
        front.append(p)
        last = float(ind(-np.asarray(front)))
        trace.append(last)
    return trace
