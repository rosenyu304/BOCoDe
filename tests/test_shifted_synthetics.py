"""Guards for the SHIFTED synthetic test functions.

Six of the centered synthetics (Ackley, Griewank, Rastrigin, DixonPrice, Levy, Powell)
have their global optimum exactly at the center of the search box, so a center-biased
initial design -- or a linear embedding whose origin maps to the box center -- gets the
optimum for free (observed live: BAxUS hit f = 0.60 on 10-D Ackley after 3 iterations
while random search was at 19.7). The ``<Name>Shifted`` variants translate the input,
``f_shifted(x) = f(x - delta)``, so the optimum lands at the 2/3 point of each
even-indexed dimension and the 1/3 point of each odd-indexed dimension.

These tests check that the shift is a *pure input offset*:

* the optimal VALUE is unchanged (``evaluate(x_opt) == -optimum``, maximization frame);
* the shifted optimum is strictly inside the bounds, and is neither the box center nor a
  corner;
* the box center is no longer optimal for the six center-biased problems;
* random search cannot beat the known optimum (i.e. translating the box did not expose a
  better region of the underlying formula).
"""

import pytest
import torch

import bocode

# Shifted problem name -> the centered problem it is derived from.
SHIFTED = {
    "AckleyShifted": "Ackley",
    "GriewankShifted": "Griewank",
    "RastriginShifted": "Rastrigin",
    "DixonPriceShifted": "DixonPrice",
    "LevyShifted": "Levy",
    "PowellShifted": "Powell",
    "RosenbrockShifted": "Rosenbrock",
    "StyblinskiTangShifted": "StyblinskiTang",
    "HartmannShifted": "Hartmann",
    "BraninShifted": "Branin",
    "SixHumpCamelShifted": "SixHumpCamel",
}

# The six whose CENTERED version hands the optimum to a center-biased method.
CENTER_BIASED = ["Ackley", "Griewank", "Rastrigin", "DixonPrice", "Levy", "Powell"]

N_SAMPLES = 2048  # random-search budget for the "cannot beat the optimum" check


def _center(problem):
    bounds = problem.torch_bounds.to(torch.double)
    return ((bounds[:, 0] + bounds[:, 1]) / 2).unsqueeze(0)


def _f(problem, X):
    """Objective value (maximization frame) at a single point."""
    values, _ = problem.evaluate(X)
    return values.item()


@pytest.mark.parametrize("name", SHIFTED)
def test_shifted_optimum_has_the_same_value(name):
    """``f_shifted(x* + delta)`` equals the centered optimum: a pure input offset."""
    problem = bocode.get_problem(name)()
    centered = bocode.get_problem(SHIFTED[name])()

    x_star_centered = centered._fn.optimizers[0].to(torch.double).unsqueeze(0)
    f_centered = _f(centered, x_star_centered)
    f_shifted = _f(problem, problem.x_opt.unsqueeze(0))

    scale = max(1.0, abs(f_centered))
    assert f_shifted == pytest.approx(f_centered, abs=1e-3 * scale)
    # BoCoDe convention: ``optimum`` is f* in the MINIMIZATION frame.
    assert f_shifted == pytest.approx(-problem.optimum, abs=1e-3 * scale)


@pytest.mark.parametrize("name", SHIFTED)
def test_shifted_optimum_is_strictly_inside_the_bounds(name):
    """The optimum lands at the 2/3 (even dims) / 1/3 (odd dims) point, not on an edge."""
    problem = bocode.get_problem(name)()
    bounds = problem.torch_bounds.to(torch.double)
    lo, hi = bounds[:, 0], bounds[:, 1]
    x_opt = problem.x_opt

    assert bool(((x_opt > lo) & (x_opt < hi)).all()), (
        "shifted optimum is outside/on bounds"
    )

    fractions = (x_opt - lo) / (hi - lo)
    expected = torch.tensor(
        [2 / 3 if j % 2 == 0 else 1 / 3 for j in range(problem.dim)], dtype=torch.double
    )
    assert torch.allclose(fractions, expected, atol=1e-9)
    # Neither the center nor a corner.
    assert not torch.allclose(fractions, torch.full_like(fractions, 0.5), atol=1e-6)
    assert not bool(((fractions < 1e-6) | (fractions > 1 - 1e-6)).all()), (
        "shifted optimum is a corner"
    )


@pytest.mark.parametrize("name", CENTER_BIASED)
def test_center_no_longer_gives_the_optimum_away(name):
    """The box center is optimal for the centered problem but not for the shifted one."""
    centered = bocode.get_problem(name)()
    shifted = bocode.get_problem(f"{name}Shifted")()

    f_star = -shifted.optimum  # maximization frame
    f_center_centered = _f(centered, _center(centered))
    f_center_shifted = _f(shifted, _center(shifted))

    scale = max(1.0, abs(f_star))
    # Centered: the box center IS (essentially) the answer for Ackley/Griewank/Rastrigin,
    # and beats thousands of random samples for DixonPrice/Levy/Powell.
    assert f_center_centered > f_center_shifted
    # Shifted: the center is strictly, meaningfully worse than the optimum.
    assert f_center_shifted < f_star - 1e-2 * scale


@pytest.mark.parametrize("name", SHIFTED)
def test_random_search_cannot_beat_the_known_optimum(name):
    """No random point may score above ``-optimum`` (the shift must not expose a better region)."""
    problem = bocode.get_problem(name)()
    X = problem.sample(N_SAMPLES, seed=0)
    values, _ = problem.evaluate(X)

    f_star = -problem.optimum
    scale = max(1.0, abs(f_star))
    assert values.max().item() <= f_star + 1e-3 * scale
