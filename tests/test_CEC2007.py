import pytest
import torch

import bocode.CEC.CEC2007 as CEC2007
from bocode.exceptions import FunctionDefinitionAssertionError

benchmark_classes = [
    CEC2007.CEC2007_OKA2,
    CEC2007.CEC2007_S_ZDT1,
    CEC2007.CEC2007_S_ZDT2,
    CEC2007.CEC2007_S_ZDT4,
    CEC2007.CEC2007_S_ZDT6,
]
benchmark_classes_with_dim = [CEC2007.CEC2007_SYMPART, CEC2007.CEC2007_R_ZDT4]
benchmark_classes_with_dim_and_objectives = [
    CEC2007.CEC2007_S_DTLZ2,
    CEC2007.CEC2007_S_DTLZ3,
    CEC2007.CEC2007_R_DTLZ2,
    CEC2007.CEC2007_WFG1,
    CEC2007.CEC2007_WFG8,
    CEC2007.CEC2007_WFG9,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC2007_evaluate(benchmark):
    problem = benchmark()

    dim = problem.dim

    rand_test_points = 5  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (rand_test_points, problem.num_constraints), (
            f"Unexpected gx shape: {gx.shape}"
        )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    if problem.num_constraints == 0:
        assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[
            1
        ].float()
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )


@pytest.mark.parametrize("benchmark", benchmark_classes_with_dim)
@pytest.mark.parametrize("dim", [2, 3, 4, 10, 20, 30, 40, 50])
def test_CEC2007_evaluate_with_dim(benchmark, dim):
    try:
        problem = benchmark(dim)
    except FunctionDefinitionAssertionError:
        return

    rand_test_points = 5  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (rand_test_points, problem.num_constraints), (
            f"Unexpected gx shape: {gx.shape}"
        )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    if problem.num_constraints == 0:
        assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[
            1
        ].float()
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )


@pytest.mark.parametrize("benchmark", benchmark_classes_with_dim_and_objectives)
@pytest.mark.parametrize("dim", [2, 3, 4, 8, 10, 30, 50])
@pytest.mark.parametrize("num_objectives", [2, 3, 4, 5])
def test_CEC2007_evaluate_with_dim_and_objectives(benchmark, dim, num_objectives):
    try:
        problem = benchmark(dim=dim, num_objectives=num_objectives)
    except FunctionDefinitionAssertionError:
        return

    rand_test_points = 5  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    X = problem.scale(X)

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (rand_test_points, problem.num_constraints), (
            f"Unexpected gx shape: {gx.shape}"
        )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    if problem.num_constraints == 0:
        assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[
            1
        ].float()
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )
