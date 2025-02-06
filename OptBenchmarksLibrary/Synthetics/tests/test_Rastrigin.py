import torch
import pytest
from .. import Rastrigin
import math

@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_rastrigin_evaluate(dim):
    problem = Rastrigin(dim)

    rand_test_points = 10 # Number of random points to test
    
    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim)) * 10.24 - 5.12

    _, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, 1), f"Unexpected fx shape: {fx.shape}"

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    # TODO: Add test points to ensure that fx is calculated correctly

    print(f"Test passed")
