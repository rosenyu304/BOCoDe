import pytest
import torch

from bocode import CEC

benchmark_classes = [
    CEC.CEC2020_p40,
    CEC.CEC2020_p41,
    CEC.CEC2020_p42,
    CEC.CEC2020_p43,
    CEC.CEC2020_p44,
    CEC.CEC2020_p45,
    CEC.CEC2020_p46,
    CEC.CEC2020_p47,
    CEC.CEC2020_p48,
    CEC.CEC2020_p49,
    CEC.CEC2020_p50,
    CEC.CEC2020_p51,
    CEC.CEC2020_p52,
    CEC.CEC2020_p53,
    CEC.CEC2020_p54,
    CEC.CEC2020_p55,
    CEC.CEC2020_p56,
    CEC.CEC2020_p57,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC40_57_evaluate(benchmark):
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
