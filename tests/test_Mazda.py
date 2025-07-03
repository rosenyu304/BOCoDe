import sys

import pytest
import torch

from bocode.Engineering import Mazda, Mazda_SCA


# Repeat 200 times
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="This test cannot run on macOS due to precompiled executables.",
)
@pytest.mark.parametrize("run", range(5))
def test_mazdaSCA_evaluate(run):
    problem = Mazda_SCA()

    dim = 148

    rand_test_points = 100  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    assert gx.shape == (rand_test_points, problem.num_constraints), (
        f"Unexpected gx shape: {gx.shape}"
    )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="This test cannot run on macOS due to precompiled executables.",
)
@pytest.mark.parametrize("run", range(5))
def test_mazda_evaluate(run):
    problem = Mazda()

    dim = 222

    rand_test_points = 100  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    assert gx.shape == (rand_test_points, problem.num_constraints), (
        f"Unexpected gx shape: {gx.shape}"
    )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )
