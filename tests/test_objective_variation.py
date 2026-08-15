"""Guards that a problem's objective actually VARIES over its search space.

A benchmark whose objective is (effectively) CONSTANT is not a benchmark: every point
is as good as every other, so the optimizer's result is noise and any comparison between
algorithms on it is meaningless.

This file exists because of ``QPowerModel``. Its vendored NEORL ``eval`` had been reduced
to ``float(unorm.sum())`` -- the sum of a *normalized* power distribution, which is 1.0
by construction. Over 256 random points the objective ranged over [0.999436, 1.002288]:
a spread of 2.9e-3 around a value that is mathematically pinned to 1. Three seeds of
results were produced against it before anyone noticed.

A NOTE ON WHY THERE IS NO SINGLE MAGIC THRESHOLD. It is tempting to assert "relative
spread > eps" for every problem. That does not work: the broken QPowerModel's relative
spread (2.9e-3) sits *inside* the range spanned by perfectly legitimate objectives --
the HPO-B classification-accuracy problems have relative spreads of 6e-3 to 5e-2 (an
accuracy near 0.9 that moves by a few points). Any threshold that flagged the old
QPowerModel would also flag real problems. So the guards here are:

1. ``test_objective_is_not_constant`` -- a hard, false-positive-free check that the
   objective is not *literally* constant. Cheap, and it already found two degenerate
   problems (see ``_KNOWN_CONSTANT``).
2. ``test_qpower_eval_returns_the_quadrant_power_vector`` and friends -- STRUCTURAL
   regression tests pinning the specific semantics that were lost. This is the class of
   test that actually catches a "the objective got collapsed to a constant" bug, because
   it knows what the objective is *supposed* to be.
"""

import numpy as np
import pytest
import torch

import bocode

# Random-sample size. Large enough to expose a flat objective, small enough to stay cheap.
N_SAMPLES = 64

# Problems that are too slow or too side-effectful to sample in a unit test.
_SKIP = {
    "MOPTA08Car": "shells out to a native binary per evaluation",
    "Mazda": "spawns a subprocess binary per evaluation",
    "Mazda_SCA": "spawns a subprocess binary per evaluation",
    "RobotPush": "opens a Box2D/pygame world",
    "SVM": "loads an 81 MB dataset and fits an SVR per evaluation",
}

# Problems whose objective IS literally constant. These are real defects, found by
# ``test_objective_is_not_constant`` -- they are xfailed (not skipped) so they stay
# visible until someone fixes or drops them.
_KNOWN_CONSTANT = {
    "InvertedPendulumProblem": (
        "degenerate by construction: Gymnasium's InvertedPendulum pays +1 per upright "
        "timestep, and this is a SINGLE-step episode starting from reset (where the "
        "pole is always upright), so the reward is exactly 1.0 for every one of its 1-D "
        "actions. The objective cannot vary; the problem is not optimizable."
    ),
    "LassoRCV1": (
        "objective is exactly -0.026292777777777783 at all 128 sampled points across "
        "its 47236 dimensions -- the weighted Lasso appears to collapse to the same "
        "(probably all-zero) predictor everywhere in the sampled box."
    ),
}

# Problems that raise before an objective can be observed. Also real defects.
_KNOWN_BROKEN = {
    "Truss120D": (
        "calls super().scale(X) unconditionally in _evaluate_implementation, so it "
        "expects unit-cube input while the harness hands it already-scaled X -> "
        "RangeException."
    ),
    "Truss200D": (
        "calls super().scale(X) unconditionally in _evaluate_implementation, so it "
        "expects unit-cube input while the harness hands it already-scaled X -> "
        "RangeException."
    ),
}


def _sampleable_problems():
    names = []
    for name in bocode.list_problems():
        if name in _SKIP:
            continue
        try:
            bocode.get_problem(name)()
        except ImportError:  # optional extra not installed
            continue
        except Exception:  # noqa: BLE001 - construction failures are another test's job
            continue
        names.append(name)
    return names


@pytest.mark.parametrize("name", _sampleable_problems())
def test_objective_is_not_constant(name):
    """Every objective column must take more than one value over a random sample.

    A constant objective means the problem carries no signal at all: the optimizer is
    choosing between points that are all exactly equally good.
    """
    if name in _KNOWN_CONSTANT:
        pytest.xfail(f"{name}: {_KNOWN_CONSTANT[name]}")
    if name in _KNOWN_BROKEN:
        pytest.xfail(f"{name}: {_KNOWN_BROKEN[name]}")

    problem = bocode.get_problem(name)()
    X = problem.sample(N_SAMPLES, seed=0)
    values, _ = problem.evaluate(X)
    values = values.to(torch.double)

    for j in range(values.shape[1]):
        column = values[:, j]
        column = column[torch.isfinite(column)]
        assert column.numel() > 0, f"{name}: objective {j} has no finite value"
        spread = float(column.max() - column.min())
        assert spread > 0.0, (
            f"{name}: objective {j} is CONSTANT at {float(column[0])} over "
            f"{N_SAMPLES} random points. A constant objective is not optimizable -- "
            f"the objective has probably been collapsed (e.g. reduced to the sum of a "
            f"normalized vector, which is 1 by construction), or the search space is "
            f"not actually reaching the evaluation."
        )


# --------------------------------------------------------------------------------------
# Structural regression tests for the two NEORL microreactor objectives. These pin the
# semantics that were lost, which a generic variation check cannot do (see module docs).
# --------------------------------------------------------------------------------------


def test_qpower_eval_returns_the_quadrant_power_vector():
    """The vendored NEORL model must return the 4 quadrant power FRACTIONS, not a scalar.

    Upstream is ``return unorm / unorm.sum()``. It was once ``float(unorm.sum())``, i.e.
    the sum of a normalized distribution == 1.0, which is what destroyed the objective.
    Summing the returned vector is therefore exactly the degenerate quantity: if this
    ever returns a scalar again, the ``len == 4`` assert below fires.
    """
    pytest.importorskip("onnxruntime")
    from bocode.opt_problems._vendor.neorl_lib.qpower_model import QPowerModel

    powers = QPowerModel().eval(np.zeros(8, dtype=np.float32))

    assert np.ndim(powers) == 1 and len(powers) == 4, (
        f"QPowerModel.eval must return the 4 quadrant power fractions, got {powers!r}. "
        f"Reducing it to a scalar (.sum()) makes the objective identically 1.0."
    )
    assert powers.sum() == pytest.approx(1.0, abs=1e-6), (
        "the quadrant powers are a normalized distribution and must sum to 1"
    )


def test_qpower_objective_is_the_negated_power_imbalance():
    """QPowerModel maximizes ``-sum_q |P_q - 1/4|`` (NEORL ex11's minimized ``hatfp``)."""
    problem = pytest.importorskip("bocode").get_problem("QPowerModel")()
    from bocode.opt_problems._vendor.neorl_lib.qpower_model import QPowerModel as _Model

    X = problem.sample(N_SAMPLES, seed=0)
    values, _ = problem.evaluate(X)
    values = values.to(torch.double).flatten()

    # It is a distance-to-target, so it is non-positive with a best possible value of 0.
    assert (values <= 0).all(), "the negated power imbalance must be <= 0"

    # It must genuinely vary -- the whole point of the fix. Before the fix the relative
    # spread was 2.9e-3; it is now ~2.5 (spread is larger than the mean magnitude).
    spread = float(values.max() - values.min())
    assert spread > 1e-2, (
        f"QPowerModel's objective spread is {spread}, which is far too flat. NEORL "
        f"normalizes this objective by fp_max = 0.0345, so a healthy random sample "
        f"spans a good fraction of [0, 0.0345]."
    )

    # And it must equal the hand-computed NEORL objective at a concrete point.
    model = _Model()
    x0 = X[0].detach().cpu().numpy()
    expected = -np.abs(np.asarray(model.eval(x0), dtype=float) - 0.25).sum()
    assert float(values[0]) == pytest.approx(expected, abs=1e-6)


def test_reactivity_objective_is_the_negated_distance_to_target():
    """ReactivityModel maximizes ``-|rho(x) - 0.03308|`` (NEORL ex11's ``hatfc``).

    It used to return the raw signed reactivity worth, which has no well-defined
    optimization direction: BO simply drove the reactivity as high as it would go.
    """
    from bocode.opt_problems._vendor.neorl_lib.reactivity_model import (
        ReactivityModel as _Model,
    )

    problem = bocode.get_problem("ReactivityModel")()
    X = problem.sample(N_SAMPLES, seed=0)
    values, _ = problem.evaluate(X)
    values = values.to(torch.double).flatten()

    assert (values <= 0).all(), "the negated distance to the target must be <= 0"
    assert float(values.max() - values.min()) > 1e-3, "objective is too flat"

    model = _Model("wtd")
    x0 = X[0].detach().cpu().numpy()
    expected = -abs(float(model.eval(x0)) - 0.03308)
    assert float(values[0]) == pytest.approx(expected, abs=1e-6)
