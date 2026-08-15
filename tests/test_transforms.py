"""Tests for the problem-transform wrappers (penalty / scalarize / relax)."""

import torch

import bocode


def test_penalized_drops_constraints():
    base = bocode.get_problem("PressureVessel")()  # constrained, mixed
    p = bocode.PenalizedProblem(base)
    assert p.num_objectives == 1
    assert p.num_constraints == 0
    assert p.is_mixed_variable == base.is_mixed_variable  # variable types preserved
    values, constraints = p.evaluate(p.sample(4))
    assert values.shape == (4, 1)
    assert constraints.shape == (4, 0)
    assert torch.isfinite(values).all()


def test_scalarized_reduces_to_single_objective():
    mo = next(
        n
        for n in bocode.list_problems()
        if (bocode.get_metadata(n).get("num_objectives") or 0) >= 2
    )
    base = bocode.get_problem(mo)()
    s = bocode.ScalarizedProblem(base)
    assert s.num_objectives == 1
    assert s.num_constraints == base.num_constraints
    values, _ = s.evaluate(s.sample(4))
    assert values.shape == (4, 1)


def test_continuous_relaxation_drops_mixed():
    base = bocode.get_problem("PressureVessel")()
    r = bocode.ContinuousRelaxation(base)
    assert r.is_mixed_variable is False
    assert r.num_constraints == base.num_constraints
    values, _ = r.evaluate(r.sample(4))
    assert values.shape[0] == 4
