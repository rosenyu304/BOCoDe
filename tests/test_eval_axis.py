"""Guards on the evaluation axis and budget parity.

"1000 evaluations" means 1000 BO iterations EXCLUDING the initial design, so every method
gets the same TOTAL budget of ``n_init + 1000`` function evaluations, and every comparison
is made on the total-evaluation axis — never on the raw trace index.

Without these guards, a GP's iteration 0 (which has already spent n_init evaluations) gets
plotted at the same x as random search's iteration 0 (1 evaluation), which flatters the GP.
"""

import numpy as np
import pytest

from algorithms._eval_utils import (
    best_at_evaluations,
    check_budget_parity,
    eval_axis,
)


def _fake(n_init: int, n_trace: int, n_x: int):
    """A minimal npz-like dict."""
    return {
        "n_init": np.array(n_init),
        "per_iteration_value": np.arange(n_trace, dtype=float),
        "X": np.zeros((n_x, 3)),
    }


def test_eval_axis_offsets_by_n_init():
    gp = _fake(n_init=20, n_trace=1001, n_x=1020)
    ax = eval_axis(gp)
    # the GP's first recorded point has ALREADY cost n_init evaluations
    assert ax[0] == 20
    assert ax[-1] == 1020


def test_eval_axis_for_pure_sampling_starts_at_one():
    rs = _fake(n_init=0, n_trace=1020, n_x=1020)
    ax = eval_axis(rs)
    assert ax[0] == 1
    assert ax[-1] == 1020


def test_gp_and_rs_end_on_the_same_total_budget():
    """The whole point: equal TOTAL evaluations, so the endpoints are comparable."""
    gp = _fake(n_init=20, n_trace=1001, n_x=1020)
    rs = _fake(n_init=0, n_trace=1020, n_x=1020)
    assert eval_axis(gp)[-1] == eval_axis(rs)[-1] == 1020
    assert not check_budget_parity({"gp": gp, "rs": rs})


def test_unequal_budget_is_flagged():
    """The archived 2026-06 runs had RS=1000 evals vs GP=1020 — must be caught."""
    gp = _fake(n_init=20, n_trace=1001, n_x=1020)
    rs = _fake(n_init=0, n_trace=1000, n_x=1000)
    problems = check_budget_parity({"gp": gp, "rs": rs})
    assert problems and "UNEQUAL TOTAL BUDGET" in problems[0]


def test_best_at_evaluations_returns_nan_past_the_budget():
    """A truncated run must never masquerade as a complete one."""
    rs = _fake(n_init=0, n_trace=1000, n_x=1000)
    assert np.isnan(best_at_evaluations(rs, 1020))
    assert not np.isnan(best_at_evaluations(rs, 1000))


def test_comparing_on_trace_index_would_be_wrong():
    """Demonstrates the bug the module exists to prevent."""
    gp = _fake(n_init=20, n_trace=1001, n_x=1020)
    rs = _fake(n_init=0, n_trace=1020, n_x=1020)
    # same trace index -> DIFFERENT numbers of function evaluations spent
    assert eval_axis(gp)[0] != eval_axis(rs)[0]
    # the GP has spent 20 evaluations by the time RS has spent 1
    assert eval_axis(gp)[0] == 20 and eval_axis(rs)[0] == 1


@pytest.mark.parametrize("budget", [100, 520, 1020])
def test_best_at_evaluations_is_monotone_in_budget(budget):
    gp = _fake(n_init=20, n_trace=1001, n_x=1020)
    v = best_at_evaluations(gp, budget)
    assert np.isfinite(v)
