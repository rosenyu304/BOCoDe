import pytest
import torch

from bocode import DTLZ
from bocode.exceptions import FunctionDefinitionAssertionError

benchmark_classes = [
    DTLZ.DTLZ1,
    DTLZ.DTLZ2,
    DTLZ.DTLZ3,
    DTLZ.DTLZ4,
    DTLZ.DTLZ5,
    DTLZ.DTLZ6,
    DTLZ.DTLZ7,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
@pytest.mark.parametrize("dim", [4, 8, 10, 20])
@pytest.mark.parametrize("num_objectives", [2, 3, 5, 8])
def test_WFG_evaluate(benchmark, dim, num_objectives):
    try:
        problem = benchmark(dim=dim, num_objectives=num_objectives)
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
