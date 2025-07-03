import pytest
import torch

from bocode.Engineering.BayesianCHT import (
    NonLinearConstraintProblemA3,
    NonLinearConstraintProblemA4,
    NonLinearConstraintProblemA7,
    NonLinearConstraintProblemA8,
    NonLinearConstraintProblemB3,
    NonLinearConstraintProblemB4,
    NonLinearConstraintProblemB7,
    NonLinearConstraintProblemB8,
)


def general_test(func, dim, num_obj):
    problem = func(dim, num_objectives=num_obj)

    rand_test_points = 10  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    assert gx.shape == (rand_test_points, problem.num_constraints), (
        f"Unexpected gx shape: {gx.shape}"
    )

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(
            torch.Tensor(problem.x_opt), scaling=False
        )[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )


@pytest.mark.parametrize(
    "func",
    [
        NonLinearConstraintProblemA3,
        NonLinearConstraintProblemA4,
        NonLinearConstraintProblemA7,
        NonLinearConstraintProblemA8,
        NonLinearConstraintProblemB3,
        NonLinearConstraintProblemB4,
        NonLinearConstraintProblemB7,
        NonLinearConstraintProblemB8,
    ],
)
@pytest.mark.parametrize("dim", [2, 3, 5, 10])
@pytest.mark.parametrize("num_objectives", [1, 2, 3, 4])
def test_botorch_funcs(func, dim, num_objectives):
    general_test(func, dim, num_objectives)
