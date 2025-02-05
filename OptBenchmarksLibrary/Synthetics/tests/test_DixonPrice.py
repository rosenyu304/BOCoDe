import torch
import pytest
from .. import DixonPrice

def test_dixon_evaluate():
    problem = DixonPrice()

    rand_test_points = 10 # Number of random points to test
    
    # Generate random points within constraints
    X = torch.rand((rand_test_points, 2)) * 20 - 10

    _, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, 1), f"Unexpected fx shape: {fx.shape}"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    # TODO: Add test points to ensure that fx is calculated correctly

    print(f"Test passed")
