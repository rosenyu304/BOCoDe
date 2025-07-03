import torch

from bocode.Synthetics import Goldstein, Goldstein_Discrete


def test_goldstein_evaluate():
    problem = Goldstein()

    dim = 2

    rand_test_points = 10  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    X[:, 0] = X[:, 0] * 4 - 2

    _, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )

    # TODO: Add test points to ensure that fx is calculated correctly


def test_goldstein_discrete_evaluate():
    problem = Goldstein_Discrete()

    dim = 2

    rand_test_points = 10  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    X[:, 0] = X[:, 0] * 4 - 2

    _, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, 1), f"Unexpected fx shape: {fx.shape}"

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )

    # TODO: Add test points to ensure that fx is calculated correctly
