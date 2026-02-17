import torch

from bocode.Synthetics import (
    Beale,
    ConstrainedGramacy,
    ConstrainedHartmann,
    ConstrainedHartmannSmooth,
    Cosine8,
    CrossInTray,
    DropWave,
    EggHolder,
    GramacyLee,
    Hartmann3D,
    Hartmann6D,
    HolderTable,
    Langermann,
    LevyN13,
    SchafferN2,
    SchafferN4,
    Schwefel,
    Shekelm5,
    Shekelm7,
    Shekelm10,
    Shubert,
    SixHumpCamel,
    ThreeHumpCamel,
)


def general_test(func, dim=None, has_scaling=True):
    if dim:
        problem = func(dim)
    else:
        problem = func()
        dim = problem.dim

    rand_test_points = 5  # Number of random points to test

    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    if has_scaling:
        _, fx = problem._evaluate_implementation(X, scaling=True)
    else:
        _, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"

    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"


def general_test_constrained(func, dim=None, has_scaling=True):
    if dim:
        problem = func(dim)
    else:
        problem = func()
        dim = problem.dim

    rand_test_points = 5

    X = torch.rand((rand_test_points, dim))

    if has_scaling:
        gx, fx = problem._evaluate_implementation(X, scaling=True)
    else:
        gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), (
        f"Unexpected fx shape: {fx.shape}"
    )
    assert gx is not None
    assert gx.shape[0] == rand_test_points
    assert gx.shape[1] == problem.num_constraints

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"
    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"


def test_beale():
    general_test(Beale)


def test_cosine8():
    general_test(Cosine8)


def test_crossintray():
    # CrossInTray uses torch.exp(~95) which overflows float32 (max ~3.4e38).
    # Use float64 to avoid overflow.
    problem = CrossInTray()
    X = torch.rand((5, problem.dim), dtype=torch.float64)
    X = problem.scale(X)
    _, fx = problem._evaluate_implementation(X)
    assert fx.shape == (5, problem.num_objectives)
    assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"


def test_dropwave():
    general_test(DropWave)


def test_eggholder():
    general_test(EggHolder)


def test_gramacylee():
    general_test(GramacyLee, has_scaling=False)


def test_hartmann3d():
    general_test(Hartmann3D)


def test_hartmann6d():
    general_test(Hartmann6D)


def test_holdertable():
    general_test(HolderTable)


def test_langermann():
    general_test(Langermann, has_scaling=False)


def test_levyn13():
    general_test(LevyN13, has_scaling=False)


def test_schaffern2():
    general_test(SchafferN2, has_scaling=False)


def test_schaffern4():
    general_test(SchafferN4, has_scaling=False)


def test_schwefel():
    general_test(Schwefel, has_scaling=False)


def test_shekelm5():
    general_test(Shekelm5)


def test_shekelm7():
    general_test(Shekelm7)


def test_shekelm10():
    general_test(Shekelm10)


def test_shubert():
    general_test(Shubert, has_scaling=False)


def test_sixhumpcamel():
    general_test(SixHumpCamel)


def test_threehumpcamel():
    general_test(ThreeHumpCamel)


def test_constrainedgramacy():
    general_test_constrained(ConstrainedGramacy)


def test_constrainedhartmann():
    general_test_constrained(ConstrainedHartmann)


def test_constrainedhartmannsmooth():
    general_test_constrained(ConstrainedHartmannSmooth)
