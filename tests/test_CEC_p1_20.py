import pytest
import torch

from bocode import CEC

benchmark_classes = [
    CEC.CEC2020_p1,
    CEC.CEC2020_p2,
    CEC.CEC2020_p3,
    CEC.CEC2020_p4,
    CEC.CEC2020_p5,
    CEC.CEC2020_p6,
    CEC.CEC2020_p7,
    CEC.CEC2020_p8,
    CEC.CEC2020_p9,
    CEC.CEC2020_p10,
    CEC.CEC2020_p11,
    CEC.CEC2020_p12,
    CEC.CEC2020_p13,
    CEC.CEC2020_p14,
    CEC.CEC2020_p15,
    CEC.CEC2020_p16,
    CEC.CEC2020_p17,
    CEC.CEC2020_p18,
    CEC.CEC2020_p19,
    CEC.CEC2020_p20,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC1_20_evaluate(benchmark):
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
