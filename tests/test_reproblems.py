import pytest
import torch

from bocode.Engineering import reproblems

re_classes = [
    reproblems.RE21,
    reproblems.RE22,
    reproblems.RE23,
    reproblems.RE24,
    reproblems.RE25,
    reproblems.RE31,
    reproblems.RE32,
    reproblems.RE33,
    reproblems.RE34,
    reproblems.RE35,
    reproblems.RE36,
    reproblems.RE37,
    reproblems.RE41,
    reproblems.RE42,
    reproblems.RE61,
    reproblems.RE91,
]

cre_classes = [
    reproblems.CRE21,
    reproblems.CRE22,
    reproblems.CRE23,
    reproblems.CRE24,
    reproblems.CRE25,
    reproblems.CRE31,
    reproblems.CRE32,
    reproblems.CRE51,
]


@pytest.mark.parametrize("benchmark", re_classes)
def test_RE_evaluate(benchmark):
    problem = benchmark()
    dim = problem.dim
    rand_test_points = 5

    # Use scaling=True to map [0,1] inputs to problem bounds
    X = torch.rand((rand_test_points, dim))
    gx, fx = problem._evaluate_implementation(X, scaling=True)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    assert gx.shape == (rand_test_points, problem.num_constraints), (
        f"Unexpected gx shape: {gx.shape}"
    )
    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"
    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"


@pytest.mark.parametrize("benchmark", cre_classes)
def test_CRE_evaluate(benchmark):
    problem = benchmark()
    dim = problem.dim
    rand_test_points = 5

    # Use scaling=True to map [0,1] inputs to problem bounds
    X = torch.rand((rand_test_points, dim))
    gx, fx = problem._evaluate_implementation(X, scaling=True)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    assert gx.shape == (rand_test_points, problem.num_constraints), (
        f"Unexpected gx shape: {gx.shape}"
    )
    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"
    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"
    assert torch.isfinite(gx).all(), "gx contains NaN or Inf values"
