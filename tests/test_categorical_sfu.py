"""Tests for the high-cardinality categorical SFU problems.

AckleyCat / RastriginCat / SchwefelCat (20 nominal vars, K=4) and Ackley53
(50 binary nominal + 3 continuous). The critical property under test is
*non-degeneracy*: the categorical levels sit on a one-sided grid ``[0, ub]``, not
the symmetric grid ``[-ub, ub]``, because Ackley and Rastrigin are even in every
coordinate and a symmetric grid would make levels ``i`` and ``K-1-i`` score
identically (for the binary Ackley53 grid, it would make all 50 categorical
variables completely inert).
"""

import math

import pytest
import torch

import bocode

CATEGORICAL_SFU = ["AckleyCat", "RastriginCat", "SchwefelCat", "Ackley53"]


@pytest.mark.parametrize("name", CATEGORICAL_SFU)
def test_registered_and_evaluates(name):
    p = bocode.get_problem(name)()

    bounds = p.torch_bounds.to(torch.float64)
    X = bounds[:, 0] + torch.rand(16, p.dim, dtype=torch.float64) * (
        bounds[:, 1] - bounds[:, 0]
    )
    values, constraints = p.evaluate(X)

    assert values.shape == (16, 1)
    assert constraints.shape == (16, 0)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("name", CATEGORICAL_SFU)
def test_sampling_lands_on_the_grid(name):
    p = bocode.get_problem(name)()
    assert p.is_mixed_variable

    s = p.sample(12, seed=0)
    assert s.shape == (12, p.dim)
    assert torch.equal(s, p.enforce_variable_types(s))

    values, _ = p.evaluate(s)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("name", CATEGORICAL_SFU)
def test_optimum_is_attainable_at_x_opt(name):
    """f(x_opt) must equal the declared optimum (in the minimization sense)."""
    p = bocode.get_problem(name)()
    values, _ = p.evaluate(torch.tensor(p.x_opt, dtype=torch.float64))
    f_min = -float(values)  # evaluate() returns -f because BoCoDe maximizes
    assert f_min == pytest.approx(p.optimum[0], abs=1e-6)


@pytest.mark.parametrize("name", CATEGORICAL_SFU)
def test_x_opt_is_the_best_attainable_value(name):
    """No on-grid point beats the declared optimum."""
    p = bocode.get_problem(name)()
    values, _ = p.evaluate(p.sample(4000, seed=3))
    best_f = -float(values.max())
    assert best_f >= p.optimum[0] - 1e-6


@pytest.mark.parametrize("name", CATEGORICAL_SFU)
def test_categories_are_not_degenerate(name):
    """Sweeping one variable through its levels must give K *distinct* objectives.

    This is what the one-sided grid buys. On the symmetric grid an even landscape
    (Ackley, Rastrigin) scores levels ``i`` and ``K-1-i`` identically, so this would
    yield only ceil(K/2) distinct values.
    """
    p = bocode.get_problem(name)()
    levels = p.resolved_variable_types()[0]
    assert not isinstance(levels, str)  # dim 0 is categorical in all four problems

    base = torch.tensor(p.x_opt, dtype=torch.float64)
    seen = set()
    for level in levels:
        x = base.clone()
        x[0, 0] = level
        values, _ = p.evaluate(x)
        seen.add(round(-float(values), 6))

    assert len(seen) == len(levels)


def test_ackley_cat_levels_are_one_sided():
    p = bocode.AckleyCat()
    assert p.LEVELS == pytest.approx([0.0, 32.768 / 3, 2 * 32.768 / 3, 32.768])
    assert p.resolved_variable_types() == [p.LEVELS] * 20
    assert p.optimum == [0.0]  # Ackley's f=0 at x=0 sits on level 0


def test_rastrigin_cat_levels_are_one_sided():
    p = bocode.RastriginCat()
    assert p.LEVELS == pytest.approx([0.0, 5.12 / 3, 2 * 5.12 / 3, 5.12])
    assert p.optimum == [0.0]  # Rastrigin's f=0 at x=0 sits on level 0


def test_schwefel_cat_grid_optimum_is_off_the_continuous_optimum():
    """Schwefel's continuous minimiser (420.9687) is not on the grid; f* > 0."""
    p = bocode.SchwefelCat()
    assert p.LEVELS == pytest.approx([0.0, 500 / 3, 1000 / 3, 500.0])
    assert p.optimum[0] > 0.0
    # separable, so the exact grid optimum is 20 copies of the best single level
    best_level = max(p.LEVELS, key=lambda v: v * math.sin(math.sqrt(abs(v))))
    assert best_level == pytest.approx(500 / 3)
    assert p.x_opt == [[best_level] * 20]


def test_ackley53_composition_and_symmetric_grid_would_be_inert():
    """50 binary categorical + 3 continuous; the symmetric grid would kill all 50."""
    p = bocode.Ackley53()
    types = p.resolved_variable_types()
    assert p.dim == 53
    assert sum(1 for t in types if not isinstance(t, str)) == 50
    assert sum(1 for t in types if t == "continuous") == 3
    assert types[0] == pytest.approx([0.0, 32.768])

    # The 3 continuous dims are live and span the full symmetric box.
    assert p.bounds[50] == (-32.768, 32.768)
    x = torch.zeros(2, 53, dtype=torch.float64)
    x[1, 50:] = 1.0
    values, _ = p.evaluate(x)
    assert not math.isclose(float(values[0]), float(values[1]))

    # Counterfactual: on MCBO's symmetric binary grid the two levels are identical,
    # so every one of the 2^50 categorical configurations would score the same.
    def ackley(v: torch.Tensor) -> float:
        n = v.shape[1]
        return float(
            -20 * torch.exp(-0.2 * torch.sqrt((v**2).sum(1) / n))
            - torch.exp(torch.cos(2 * math.pi * v).sum(1) / n)
            + 20
            + math.e
        )

    lo = torch.zeros(1, 53, dtype=torch.float64)
    hi = torch.zeros(1, 53, dtype=torch.float64)
    lo[0, :50], hi[0, :50] = -32.768, 32.768  # symmetric grid
    assert ackley(lo) == pytest.approx(ackley(hi))  # inert: the bug we avoid

    lo[0, :50], hi[0, :50] = 0.0, 32.768  # one-sided grid (what we ship)
    assert ackley(lo) != pytest.approx(ackley(hi))  # active
    assert ackley(lo) == pytest.approx(0.0, abs=1e-9)  # and level 0 reaches f=0
