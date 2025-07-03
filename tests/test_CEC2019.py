import pytest
import torch

import bocode.CEC.CEC2019 as CEC2019

benchmark_classes = [
    getattr(CEC2019, cls_name)
    for cls_name in dir(CEC2019)
    if cls_name.startswith("CEC2019_p") and callable(getattr(CEC2019, cls_name))
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC2019_evaluate(benchmark):
    problem = benchmark()

    dim = problem.dim

    if isinstance(problem.available_dimensions, int):
        assert dim == problem.available_dimensions, (
            f"Dimension {dim} does not match available dimensions variable: {problem.available_dimensions}"
        )

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
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=0.05), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )
