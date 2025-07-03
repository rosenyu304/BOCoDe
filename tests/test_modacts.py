import pytest
import torch

from bocode.MODAct import (
    CS1,
    CS2,
    CS3,
    CS4,
    CT1,
    CT2,
    CT3,
    CT4,
    CTS1,
    CTS2,
    CTS3,
    CTS4,
    CTSE1,
    CTSE2,
    CTSE3,
    CTSE4,
    CTSEI1,
    CTSEI2,
    CTSEI3,
    CTSEI4,
)


def general_test(func, dim=None):
    if dim:
        problem = func(dim)
    else:
        problem = func()
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


@pytest.mark.parametrize(
    "func",
    [
        CS1,
        CT1,
        CTS1,
        CTSE1,
        CTSEI1,
        CS2,
        CT2,
        CTS2,
        CTSE2,
        CTSEI2,
        CS3,
        CT3,
        CTS3,
        CTSE3,
        CTSEI3,
        CS4,
        CT4,
        CTS4,
        CTSE4,
        CTSEI4,
    ],
)
def test_cs(func):
    general_test(func)
