import torch
import pytest
from .. import Bukin

def test_bukin_evaluate():
    problem = Bukin()

    rand_test_points = 10 # Number of random points to test
    
    # Generate random points within constraints
    X = torch.rand((rand_test_points, 2))
    X[:, 0] = X[:, 0] * 10 - 15
    X[:, 1] = X[:, 1] * 6 - 3

    _, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, ), f"Unexpected fx shape: {fx.shape}"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    # TODO: Add test points to ensure that fx is calculated correctly

    print(f"Test passed")
