import pytest
import torch

from bocode.Synthetics import Ackley


@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_ackley_evaluate(dim):
    problem = Ackley(dim=dim)

    rand_test_points = 3  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim)) * 15 - 5

    gx, fx = problem._evaluate_implementation(X)

    assert gx.shape == (rand_test_points, problem.num_constraints), (
        f"Unexpected gx shape: {gx.shape}"
    )
    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    # Check constraints by repeating calculations
    sum_constraints = torch.sum(X, dim=1)
    norm_constraints = torch.norm(X, p=2, dim=1) - 5

    assert torch.allclose(gx[:, 0], sum_constraints, atol=1e-5), (
        "gx[:, 0] does not match expected sum constraint"
    )
    assert torch.allclose(gx[:, 1], norm_constraints, atol=1e-5), (
        "gx[:, 1] does not match expected norm constraint"
    )

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )

    # TODO: Add test points to ensure that fx is calculated correctly

    print(f"Test passed for dim={dim}")
