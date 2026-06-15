import pytest
import torch

from bocode.opt_problems import cec2020_rw as CEC2020_RW_Constrained

benchmark_classes = [
    CEC2020_RW_Constrained.CEC2020_p21,
    CEC2020_RW_Constrained.CEC2020_p22,
    CEC2020_RW_Constrained.CEC2020_p23,
    CEC2020_RW_Constrained.CEC2020_p24,
    CEC2020_RW_Constrained.CEC2020_p25,
    CEC2020_RW_Constrained.CEC2020_p26,
    CEC2020_RW_Constrained.CEC2020_p27,
    CEC2020_RW_Constrained.CEC2020_p28,
    CEC2020_RW_Constrained.CEC2020_p29,
    CEC2020_RW_Constrained.CEC2020_p30,
    CEC2020_RW_Constrained.CEC2020_p31,
    CEC2020_RW_Constrained.CEC2020_p32,
    CEC2020_RW_Constrained.CEC2020_p33,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC21_33_evaluate(benchmark):
    problem = benchmark()

    dim = problem.dim

    rand_test_points = 5  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    # evaluate() concatenates equality + inequality into one constraint tensor,
    # so its width equals num_constraints for every problem.
    values, constraints = problem.evaluate(X)

    assert values.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected objective shape: {values.shape}"
    )
    assert constraints.shape == (rand_test_points, problem.num_constraints), (
        f"Unexpected constraint shape: {constraints.shape}"
    )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"
    assert torch.isfinite(values).all(), "objective contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem.evaluate(torch.Tensor(problem.x_opt))[0].float()
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )
