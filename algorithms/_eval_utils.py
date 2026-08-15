"""Comparing methods fairly: the evaluation axis is TOTAL FUNCTION EVALUATIONS.

THE BUG THIS EXISTS TO PREVENT
------------------------------
Different methods consume their budget differently:

    single_task_gp:  n_init = 20 initial samples, then 1000 BO iterations
                     -> its trace has 1001 entries, and trace index i corresponds to
                        evaluation number (n_init + i)
    random_search:   1020 samples, no separate initial design (n_init stored as 0)
                     -> its trace has 1020 entries, and trace index i corresponds to
                        evaluation number (i + 1)

Plotting both against the TRACE INDEX therefore places the GP's iteration 0 -- which has
already spent 20 function evaluations -- at the same x as random search's iteration 0,
which has spent 1. That systematically FLATTERS the GP at the left of every plot, and it
silently changes the headline "best after N evaluations" number.

"1000 evaluations" means **1000 BO iterations EXCLUDING the initial design** (Rosen,
2026-07-14). So every method is given the same TOTAL budget of ``n_init + 1000`` function
evaluations, and every comparison is made on the total-evaluation axis.

Use :func:`best_at_evaluations` / :func:`aligned_traces` for ANY plot or table. Never index
a raw trace directly.
"""

from __future__ import annotations

import numpy as np


def n_init_of(npz, dim_default: int | None = None) -> int:
    """Initial-design size actually used by the run that produced ``npz``."""
    ni = int(np.asarray(npz["n_init"]).item()) if "n_init" in npz else 0
    return ni


def eval_axis(npz) -> np.ndarray:
    """The number of function evaluations consumed at each entry of the trace.

    For a method with an initial design (``n_init > 0``) the trace starts *after* that
    design is spent, so entry 0 already cost ``n_init`` evaluations.
    For a pure-sampling method (``n_init == 0``) entry ``i`` costs ``i + 1`` evaluations.
    """
    tr = np.asarray(npz["per_iteration_value"], dtype=float)
    ni = n_init_of(npz)
    if ni > 0:
        return ni + np.arange(len(tr))  # entry 0 == n_init evaluations spent
    return np.arange(1, len(tr) + 1)  # entry 0 == 1 evaluation spent


def best_at_evaluations(npz, budget: int) -> float:
    """Best-so-far value once exactly ``budget`` TOTAL function evaluations are spent.

    ``budget`` is the total (i.e. ``n_init + n_bo_iterations``). Returns NaN if the run
    never reached that many evaluations, so a truncated run can never masquerade as a
    complete one.
    """
    tr = np.asarray(npz["per_iteration_value"], dtype=float)
    ax = eval_axis(npz)
    ok = ax <= budget
    if not ok.any() or ax.max() < budget:
        return float("nan")
    return float(tr[ok][-1])


def aligned_traces(npzs: dict, budget: int) -> dict:
    """Interpolate every method's trace onto a COMMON total-evaluation grid ``1..budget``.

    ``npzs`` maps method name -> loaded npz. Returns method -> (grid, best_so_far) where
    the grid is identical across methods, so the curves are directly comparable.

    A method's curve is NaN before its initial design is spent — it genuinely has no
    "best-so-far" at evaluation 5 if it needs 20 evaluations before it reports anything.
    Showing NaN (a gap) is honest; back-filling it would invent data.
    """
    grid = np.arange(1, budget + 1)
    out = {}
    for name, d in npzs.items():
        tr = np.asarray(d["per_iteration_value"], dtype=float)
        ax = eval_axis(d)
        y = np.full(grid.shape, np.nan, dtype=float)
        # step function: best-so-far is constant between recorded evaluations
        j = 0
        cur = np.nan
        for k, g in enumerate(grid):
            while j < len(ax) and ax[j] <= g:
                cur = tr[j]
                j += 1
            y[k] = cur
        out[name] = (grid, y)
    return out


def check_budget_parity(npzs: dict) -> list[str]:
    """Return a list of complaints if the methods did NOT get the same total budget."""
    totals = {}
    for name, d in npzs.items():
        X = np.asarray(d["X"]) if "X" in d else None
        totals[name] = int(X.shape[0]) if X is not None else int(eval_axis(d).max())
    bad = []
    if len(set(totals.values())) > 1:
        bad.append(
            "UNEQUAL TOTAL BUDGET across methods — they are not comparable: "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        )
    return bad
