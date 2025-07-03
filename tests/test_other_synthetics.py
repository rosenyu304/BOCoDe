import torch

from bocode.Synthetics import SVM
from bocode.Synthetics.More_Synthetics import (
    Beale,
    ConstrainedGramacy,
    ConstrainedHartmann,
    ConstrainedHartmannSmooth,
    Cosine8,
    DropWave,
    EggHolder,
    Hartmann3D,
    Hartmann6D,
    HolderTable,
    PressureVessel,
    Shekelm5,
    Shekelm7,
    Shekelm10,
    SixHumpCamel,
    SpeedReducer,
    TensionCompressionString,
    ThreeHumpCamel,
    WeldedBeamSO,
)


def general_test(func, dim=None):
    if dim:
        problem = func(dim)
    else:
        problem = func()
        dim = problem.dim

    rand_test_points = 5  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    _, fx = problem._evaluate_implementation(X, scaling=True)

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


def test_beale():
    general_test(Beale)


def test_cosine8():
    general_test(Cosine8)


def test_dropwave():
    general_test(DropWave)


def test_eggholder():
    general_test(EggHolder)


def test_hartmann3d():
    general_test(Hartmann3D)


def test_hartmann6d():
    general_test(Hartmann6D)


def test_holdertable():
    general_test(HolderTable)


def test_shekelm5():
    general_test(Shekelm5)


def test_shekelm7():
    general_test(Shekelm7)


def test_shekelm10():
    general_test(Shekelm10)


def test_sixhumpcamel():
    general_test(SixHumpCamel)


def test_threehumpcamel():
    general_test(ThreeHumpCamel)


def test_constrainedgramacy():
    general_test(ConstrainedGramacy)


def test_constrainedhartmann():
    general_test(ConstrainedHartmann)


def test_constrainedhartmannsmooth():
    general_test(ConstrainedHartmannSmooth)


def test_pressurevessel():
    general_test(PressureVessel)


def test_weldedbeam():
    general_test(WeldedBeamSO)


def test_tensioncompressionstring():
    general_test(TensionCompressionString)


def test_speedreducer():
    general_test(SpeedReducer)


def test_SVM():
    general_test(SVM)
