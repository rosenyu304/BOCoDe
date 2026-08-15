import pytest
import torch

try:
    from bocode import (
        AntPolicySearchProblem,
        AntProblem,
        HalfCheetahPolicySearchProblem,
        HalfCheetahProblem,
        HopperPolicySearchProblem,
        HopperProblem,
        HumanoidProblem,
        HumanoidStandupProblem,
        InvertedDoublePendulumProblem,
        InvertedPendulumProblem,
        PusherProblem,
        ReacherProblem,
        SwimmerPolicySearchProblem,
        SwimmerProblem,
        Walker2DPolicySearchProblem,
        Walker2DProblem,
    )
except ImportError as exc:  # optional-dependency extra not installed
    pytest.skip(f"requires an optional dependency: {exc}", allow_module_level=True)

benchmark_classes = [
    AntProblem,
    HalfCheetahProblem,
    HopperProblem,
    HumanoidProblem,
    HumanoidStandupProblem,
    InvertedDoublePendulumProblem,
    InvertedPendulumProblem,
    PusherProblem,
    ReacherProblem,
    SwimmerProblem,
    Walker2DProblem,
    SwimmerPolicySearchProblem,
    AntPolicySearchProblem,
    HalfCheetahPolicySearchProblem,
    HopperPolicySearchProblem,
    Walker2DPolicySearchProblem,
]


@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_mujoco_evaluate(benchmark):
    problem = benchmark()

    dim = problem.dim

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
