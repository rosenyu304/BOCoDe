import random

import cocoex
import pytest
import torch

from bocode.BBOB import (
    BBOB,
    BBOB_Biobj,
    BBOB_BiobjMixInt,
    BBOB_Boxed,
    BBOB_Constrained,
    BBOB_LargeScale,
    BBOB_MixInt,
    BBOB_Noisy,
)

suiteClasses = [
    BBOB,
    BBOB_Biobj,
    BBOB_BiobjMixInt,
    BBOB_Boxed,
    BBOB_Constrained,
    BBOB_LargeScale,
    BBOB_MixInt,
    BBOB_Noisy,
]
suiteNames = [
    "bbob",
    "bbob-biobj",
    "bbob-biobj-mixint",
    "bbob-boxed",
    "bbob-constrained",
    "bbob-largescale",
    "bbob-mixint",
    "bbob-noisy",
]


# Generate test cases for each suite
def generate_test_cases():
    test_cases = []
    for suite, name in zip(suiteClasses, suiteNames):
        suite_instance = cocoex.Suite(name, "", "")
        for problem in suite_instance:
            test_cases.append(
                (suite, problem.id_function, problem.id_instance, problem.dimension)
            )
    # Reduce by 1/600 randomly
    test_cases = random.sample(test_cases, len(test_cases) // 600)
    return test_cases


# Total 2072 items
@pytest.mark.parametrize(
    "suite, function_number, instance_number, dimension", generate_test_cases()
)
def test_BBOB_evaluate(suite, function_number, instance_number, dimension):
    try:
        problem = suite(
            dim=dimension,
            function_number=function_number,
            instance_number=instance_number,
        )
    except cocoex.exceptions.NoSuchProblemException:
        print(
            f"Skipping {suite.__name__}({dimension}, {function_number}, {instance_number}) due to NoSuchProblemException"
        )
        return

    dim = problem.dim

    rand_test_points = 3  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    with torch.no_grad():
        gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (
        rand_test_points,
        problem.num_objectives,
    ), f"Unexpected fx shape: {fx.shape}"
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (
            rand_test_points,
            problem.num_constraints,
        ), f"Unexpected gx shape: {gx.shape}"

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )

    print(f"{suite.__name__}({dimension}, {function_number}, {instance_number}) passed")
