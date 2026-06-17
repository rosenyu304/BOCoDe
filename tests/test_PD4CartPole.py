import pytest
import torch

try:
    from bocode import PD4CartPole
except ImportError as exc:  # optional-dependency extra not installed
    pytest.skip(f"requires an optional dependency: {exc}", allow_module_level=True)


def test_pd4cartpole_evaluate():
    problem = PD4CartPole()
    dim = problem.dim
    assert dim == 4
    assert len(problem.bounds) == dim

    rand_test_points = 2  # simulations are not cheap; keep this small
    X = torch.rand((rand_test_points, dim))
    X = problem.scale(X)

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives)
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (rand_test_points, problem.num_constraints)
    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"
