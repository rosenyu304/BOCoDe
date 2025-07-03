import pytest
import torch

from bocode.BoTorch.botorch_MultiFidelity import (
    AugmentedBranin,
    AugmentedHartmann,
    AugmentedRosenbrock,
)
from bocode.BoTorch.botorch_MultiObj import (
    BNH,
    C2DTLZ2,
    CONSTR,
    DH1,
    DH2,
    DH3,
    DH4,
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ7,
    GMM,
    MW7,
    OSY,
    SRN,
    ZDT1,
    ZDT2,
    ZDT3,
    BraninCurrin,
    CarSideImpact,
    ConstrainedBraninCurrin,
    DiscBrake,
    Penicillin,
    ToyRobust,
    VehicleSafety,
    WeldedBeam,
)

from bocode.BoTorch.botorch_MultiFidelityMultiObj import MOMFBraninCurrin, MOMFPark1
from bocode.BoTorch.botorch_SensitivityAnalysis import Gsobol, Ishigami, Morris


def general_test(func, dim=None):
    if dim:
        problem = func(dim)
    else:
        problem = func()
        dim = problem.dim

    rand_test_points = 10  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))
    X = problem.scale(X)

    _, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(
            torch.Tensor(problem.x_opt), scaling=False
        )[1]
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=1e-4), (
            f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"
        )


@pytest.mark.parametrize(
    "func", [AugmentedBranin, AugmentedHartmann, AugmentedRosenbrock]
)
def test_botorch_funcs(func):
    general_test(func)


@pytest.mark.parametrize(
    "func",
    [
        BraninCurrin,
        DH1,
        DH2,
        DH3,
        DH4,
        DTLZ1,
        DTLZ2,
        DTLZ3,
        DTLZ4,
        DTLZ5,
        DTLZ7,
        GMM,
        Penicillin,
        ToyRobust,
        VehicleSafety,
        ZDT1,
        ZDT2,
        ZDT3,
        CarSideImpact,
        BNH,
        CONSTR,
        ConstrainedBraninCurrin,
        C2DTLZ2,
        DiscBrake,
        MW7,
        OSY,
        SRN,
        WeldedBeam,
    ],
)
def test_multiobj_funcs(func):
    general_test(func)


@pytest.mark.parametrize("func", [MOMFBraninCurrin, MOMFPark1])
def test_multifidelitymultiobj_funcs(func):
    general_test(func)


@pytest.mark.parametrize("func", [Ishigami, Gsobol, Morris])
def test_sensitivity_funcs(func):
    general_test(func)
