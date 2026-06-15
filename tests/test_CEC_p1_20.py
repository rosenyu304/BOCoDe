import pytest
import torch

from bocode.opt_problems import cec2020_rw as CEC2020_RW_Constrained

benchmark_classes = [
    CEC2020_RW_Constrained.CEC2020_p1,
    CEC2020_RW_Constrained.CEC2020_p2,
    CEC2020_RW_Constrained.CEC2020_p3,
    CEC2020_RW_Constrained.CEC2020_p4,
    CEC2020_RW_Constrained.CEC2020_p5,
    CEC2020_RW_Constrained.CEC2020_p6,
    CEC2020_RW_Constrained.CEC2020_p7,
    CEC2020_RW_Constrained.CEC2020_p8,
    CEC2020_RW_Constrained.CEC2020_p9,
    CEC2020_RW_Constrained.CEC2020_p10,
    CEC2020_RW_Constrained.CEC2020_p11,
    CEC2020_RW_Constrained.CEC2020_p12,
    CEC2020_RW_Constrained.CEC2020_p13,
    CEC2020_RW_Constrained.CEC2020_p14,
    CEC2020_RW_Constrained.CEC2020_p15,
    CEC2020_RW_Constrained.CEC2020_p16,
    CEC2020_RW_Constrained.CEC2020_p17,
    CEC2020_RW_Constrained.CEC2020_p18,
    CEC2020_RW_Constrained.CEC2020_p19,
    CEC2020_RW_Constrained.CEC2020_p20,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC1_20_evaluate(benchmark):
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
