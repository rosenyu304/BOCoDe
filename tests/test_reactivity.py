import torch

from bocode.NEORL import ReactivityModel


def test_reactivity_evaluate():
    problem = ReactivityModel()

    dim = problem.dim

    rand_test_points = 10  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    X = problem.scale(X)

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
