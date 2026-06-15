import pytest
import torch

from bocode.opt_problems import cec2020_rw as CEC2020_RW_Constrained

benchmark_classes = [
    CEC2020_RW_Constrained.CEC2020_p40,
    CEC2020_RW_Constrained.CEC2020_p41,
    CEC2020_RW_Constrained.CEC2020_p42,
    CEC2020_RW_Constrained.CEC2020_p43,
    CEC2020_RW_Constrained.CEC2020_p44,
    CEC2020_RW_Constrained.CEC2020_p45,
    CEC2020_RW_Constrained.CEC2020_p46,
    CEC2020_RW_Constrained.CEC2020_p47,
    CEC2020_RW_Constrained.CEC2020_p48,
    CEC2020_RW_Constrained.CEC2020_p49,
    CEC2020_RW_Constrained.CEC2020_p50,
    CEC2020_RW_Constrained.CEC2020_p51,
    CEC2020_RW_Constrained.CEC2020_p52,
    CEC2020_RW_Constrained.CEC2020_p53,
    CEC2020_RW_Constrained.CEC2020_p54,
    CEC2020_RW_Constrained.CEC2020_p55,
    CEC2020_RW_Constrained.CEC2020_p56,
    CEC2020_RW_Constrained.CEC2020_p57,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC40_57_evaluate(benchmark):
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
