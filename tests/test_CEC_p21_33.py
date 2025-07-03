import pytest
import torch

from bocode import CEC

benchmark_classes = [
    CEC.CEC2020_p21,
    CEC.CEC2020_p22,
    CEC.CEC2020_p23,
    CEC.CEC2020_p24,
    CEC.CEC2020_p25,
    CEC.CEC2020_p26,
    CEC.CEC2020_p27,
    CEC.CEC2020_p28,
    CEC.CEC2020_p29,
    CEC.CEC2020_p30,
    CEC.CEC2020_p31,
    CEC.CEC2020_p32,
    CEC.CEC2020_p33,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC21_33_evaluate(benchmark):
    problem = benchmark()

    dim = problem.dim

    rand_test_points = 5  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    gx, ex, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (rand_test_points, problem.num_constraints), (
            f"Unexpected gx shape: {gx.shape}"
        )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    if problem.num_constraints == 0 and (ex is None or ex.numel() == 0):
        assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(
            torch.Tensor(problem.x_opt), scaling=False
        )[2].float()
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )
