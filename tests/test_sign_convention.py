"""Guards for BoCoDe's sign conventions.

Two conventions are load-bearing for every algorithm in ``bocode.algorithms``:

* **BoCoDe maximizes.** ``_evaluate_implementation`` must return objective values in
  the MAXIMIZATION frame. Almost every source problem in the literature minimizes a
  cost/weight/stress/error, so those must return the *negated* objective.
* **A constraint is satisfied when it is ``<= 0``.**

The ``optimum`` attribute, by contrast, stores the literature ``f*`` in the ORIGINAL
(minimization) sense. So for a problem with a known ``optimum``:

* ``evaluate(x_opt)`` must equal ``-optimum``; and
* no feasible point may score better than ``-optimum``.

These tests exist because both conventions have been silently violated in the past
(whole problem families returning minimization-frame values, and BoTorch constraint
*slack* -- feasible when ``>= 0`` -- passed through unnegated).
"""

import pytest
import torch

import bocode

# ``sample()`` size for the random-search bound check. Large enough that the bound is
# actually exercised on the easy problems, small enough to stay cheap.
N_SAMPLES = 256

# Problems excluded from ``test_random_search_cannot_beat_negated_optimum``, with the
# reason. These are NOT sign violations.
_BOUND_CHECK_EXCLUDED = {
    # Each evaluation shells out to the MOPTA08 binary; far too slow for a unit test.
    "MOPTA08Car": "slow (subprocess per evaluation)",
    # Continuous relaxation of a mixed problem: its ``optimum`` is the optimum of the
    # 12-profile *catalog* version (see EulerBeamMixed), but this class relaxes the
    # moment of inertia to [0, 1], so the box admits strictly better points.
    "EulerBernoulliBeamBending": "continuous relaxation; f* is the catalog optimum",
    # Their equality tolerance is deliberately relaxed to 1e12 (see CEC2020_p1_20.py,
    # commit "relax equality tolerance to 1e12"), which switches the equality
    # constraints off so the campaign has a feasibility signal on a near-measure-zero
    # feasible set. The stored optima are the correct CEC2020 f* -- verified that every
    # point beating -optimum is equality-INFEASIBLE at the standard 1e-4 tolerance
    # (residuals O(100)) -- so this bound check is not meaningful under the 1e12 tol.
    "CEC2020_p5": "equality tolerance relaxed to 1e12; feasibility filter not meaningful",
    "CEC2020_p7": "equality tolerance relaxed to 1e12; feasibility filter not meaningful",
    "CEC2020_p9": "equality tolerance relaxed to 1e12; feasibility filter not meaningful",
}


def _single_objective_problems_with_optimum():
    """Names of registry problems with one objective and a scalar ``optimum``."""
    names = []
    for name in bocode.list_problems():
        if name in _BOUND_CHECK_EXCLUDED:
            continue
        try:
            problem = bocode.get_problem(name)()
        except ImportError:  # optional extra not installed
            continue
        if problem.num_objectives != 1 or problem.optimum is None:
            continue
        names.append(name)
    return names


@pytest.mark.parametrize("name", _single_objective_problems_with_optimum())
def test_random_search_cannot_beat_negated_optimum(name):
    """No FEASIBLE random point may score above ``-optimum``.

    This is the empirical trap for an inverted objective: if a minimization problem
    forgets to negate, ``evaluate`` returns a large positive cost and immediately
    blows past ``-optimum``.
    """
    problem = bocode.get_problem(name)()
    optimum = float(torch.as_tensor(problem.optimum).flatten()[0])

    X = problem.sample(N_SAMPLES, seed=0)
    values, constraints = problem.evaluate(X)
    values = values.to(torch.double)

    if problem.num_constraints:
        feasible = (constraints.to(torch.double) <= 0).all(dim=1)
        if not feasible.any():
            pytest.skip(f"{name}: no feasible point in {N_SAMPLES} samples")
        values = values[feasible]

    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        pytest.skip(f"{name}: no finite objective value")

    best = float(values.max())
    bound = -optimum
    # Tolerate float noise in the published f*, relative to its own magnitude.
    tol = 1e-4 * max(abs(bound), 1.0)
    assert best <= bound + tol, (
        f"{name}: random search reached {best} but -optimum is {bound}. "
        f"Either the objective is returned in the minimization frame (missing "
        f"negation) or `optimum` is not the literature f* in the minimization sense."
    )


def _problems_with_x_opt():
    names = []
    for name in bocode.list_problems():
        try:
            problem = bocode.get_problem(name)()
        except ImportError:
            continue
        if problem.x_opt is None or problem.optimum is None:
            continue
        if problem.num_objectives != 1:
            continue
        names.append(name)
    return names


@pytest.mark.parametrize("name", _problems_with_x_opt())
def test_value_at_x_opt_is_negated_optimum(name):
    """``evaluate(x_opt)`` is in the maximization frame, so it must equal ``-optimum``."""
    problem = bocode.get_problem(name)()
    x_opt = torch.as_tensor(problem.x_opt, dtype=torch.double)
    expected = -torch.as_tensor(problem.optimum, dtype=torch.double).flatten()

    values, _ = problem.evaluate(x_opt)
    actual = values.to(torch.double).flatten()

    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3), (
        f"{name}: evaluate(x_opt) = {actual.tolist()} but -optimum = "
        f"{expected.tolist()}. `optimum` must store the literature f* in the "
        f"ORIGINAL (minimization) sense while evaluate() returns the maximization frame."
    )


# Constrained problems whose feasible region is a non-trivial fraction of the box.
# A 0% feasible rate here means a constraint block was passed through with the wrong
# sign -- this is exactly how the DiscBrake constraint-slack bug was caught (BoTorch
# reports slack that is feasible when >= 0; BoCoDe needs <= 0).
_MUST_HAVE_FEASIBLE_POINTS = [
    "DiscBrake",
    "CarSideImpact",
    "PressureVessel",
    "ThreeTruss",
    "Truss10D",
    "ReinforcedConcreteBeam",
    "WaterResources",
    "Sellar",
    "SatelliteDesign",
    "KeaneBumpFunction",
]


@pytest.mark.parametrize("name", _MUST_HAVE_FEASIBLE_POINTS)
def test_constrained_problem_has_feasible_points(name):
    """A random sample must contain feasible points (constraint is satisfied when <= 0)."""
    problem = bocode.get_problem(name)()
    assert problem.num_constraints > 0

    X = problem.sample(N_SAMPLES, seed=0)
    _, constraints = problem.evaluate(X)
    feasible = (constraints.to(torch.double) <= 0).all(dim=1)

    assert feasible.any(), (
        f"{name}: 0 / {N_SAMPLES} random points are feasible. A constraint block is "
        f"likely inverted (BoCoDe's convention is: feasible when the value is <= 0)."
    )


# Problems whose source objective is a cost / weight / volume / error to be MINIMIZED.
# In the maximization frame every returned value must therefore be <= 0.
_MINIMIZED_COST_PROBLEMS = [
    "CantileverBeam",
    "SteppedCantileverBeam",
    "CompressionSpring",
    "PressureVessel",
    "SpeedReducer",
    "ThreeTruss",
    "Truss10D",
    "HeatExchanger",
    "WeldedBeamSO",
    "HelicalSpring",
    "ReinforcedConcreteBeam",
    "MiniAeroWing",
    "Wing",
    "Car",
    "GearTrain",
    "MOPTA08Car",
    # Borehole minimizes the water flow rate (see its docstring: the direction is not
    # fixed by the canonical source, and the BO literature minimizes it).
    "Borehole",
    # The two NEORL microreactor objectives are distances to a target -- NEORL minimizes
    # |rho - rho_tgt| and sum_q |P_q - 1/4| -- so negated they are non-positive.
    "QPowerModel",
    "ReactivityModel",
]


@pytest.mark.parametrize("name", _MINIMIZED_COST_PROBLEMS)
def test_cost_problem_returns_non_positive_values(name):
    """A cost/weight/error problem must return non-positive (negated) values."""
    problem = bocode.get_problem(name)()
    # MOPTA08 is slow: a couple of points is enough to see the sign.
    n = 4 if name == "MOPTA08Car" else 64

    X = problem.sample(n, seed=0)
    values, _ = problem.evaluate(X)
    values = values.to(torch.double)
    values = values[torch.isfinite(values)]

    assert (values <= 0).all(), (
        f"{name}: minimizes a cost in its source, so every maximization-frame value "
        f"must be <= 0, but got a max of {float(values.max())}. Missing negation."
    )


class _StubEnv:
    """Minimal gym-like env that always hands back a fixed, known reward."""

    REWARD = 7.0

    def reset(self, seed=None):
        # The rollouts seed every reset (MuJoCo perturbs the initial state, so an unseeded
        # reset makes the objective non-deterministic). A real gym env accepts ``seed``;
        # this stub must too, or it no longer stands in for one.
        self._n = (
            0  # a real env restarts the episode here; without this, only the first
        )
        # row of a batch gets a full episode and the rest terminate immediately.
        return None, {}

    # The problems roll out a full EPISODE under the constant action (they used to take a
    # single env.step(), which is why InvertedPendulum's objective was constant at 1.0).
    # So the stub must terminate, and the expected value is the EPISODE RETURN.
    STEPS = 5

    def __init__(self):
        self._n = 0

    def step(self, action):
        self._n += 1
        terminated = self._n >= self.STEPS
        return None, self.REWARD, terminated, False, {}

    @property
    def episode_return(self):
        return self.REWARD * self.STEPS


@pytest.mark.parametrize(
    "name",
    ["HalfCheetahProblem", "PusherProblem", "ReacherProblem", "AntProblem"],
)
def test_mujoco_returns_the_reward_not_its_negation(name):
    """The MuJoCo problems must return the gym EPISODE RETURN as-is, not its negation.

    Gymnasium's MuJoCo rewards are *maximized*, which is already BoCoDe's convention,
    so they must not be negated. The reward is signed (Reacher's and Pusher's are
    negative), so a plain sign check would not catch an inversion; instead the env is
    stubbed with a known reward and the returned value is pinned to the episode return.
    A negated reward would point every algorithm at the WORST control action.

    The expected value is REWARD * STEPS, not REWARD: these problems roll out a full
    episode under the constant action. They used to take a single ``env.step()``, which is
    what made InvertedPendulum's objective constant at 1.0 (+1 per timestep alive).
    """
    pytest.importorskip("gymnasium")

    problem = bocode.get_problem(name)()
    problem.env = _StubEnv()

    values, _ = problem.evaluate(torch.zeros(3, problem.dim, dtype=torch.double))

    expected = _StubEnv.REWARD * _StubEnv.STEPS
    assert torch.allclose(
        values.to(torch.double),
        torch.full_like(values.to(torch.double), expected),
    ), (
        f"{name}: must return the gym episode return as-is (got "
        f"{values.flatten().tolist()}, expected {expected}). MuJoCo rewards are already a maximization "
        f"target, so negating them inverts the search."
    )
