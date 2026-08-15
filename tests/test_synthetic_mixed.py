"""Tests for the mixed continuous + categorical/integer synthetic problems."""

import pytest
import torch

import bocode

# (name, dim, num_constraints)
PROBLEMS = [
    ("BraninLVGP", 2, 0),
    ("GoldsteinLVGP", 2, 0),
    ("HartmannCat", 6, 0),
    ("CoCaBOFunc2C", 4, 0),
    ("CoCaBOFunc3C", 5, 0),
    ("MixedAckley", 5, 0),
    ("BraninCategorical", 3, 0),
    ("StyblinskiTangMixed", 5, 0),
    ("WeldedBeamCategorical", 5, 5),
    ("StyblinskiTangCat", 10, 0),
    ("GoldsteinMixed", 2, 0),
    ("ShekelMixed", 4, 0),
    ("Ackley5Mixed", 5, 0),
    ("Rosenbrock5Mixed", 5, 0),
]

NAMES = [p[0] for p in PROBLEMS]

# Reference optima in the ORIGINAL (minimization) sense. ``evaluate`` maximizes, so
# the best achievable value of ``-values`` must not be below these.
REFERENCE_OPTIMA = {
    "BraninLVGP": 2.79118,
    "GoldsteinLVGP": 3.0,
    "HartmannCat": -3.32237,
    "MixedAckley": 0.0,
    "BraninCategorical": 0.397887,
    "StyblinskiTangMixed": -39.16616 * 4,
    "WeldedBeamCategorical": 1.7249,
    "StyblinskiTangCat": -367.1875,
    "GoldsteinMixed": 3.0,
    "ShekelMixed": -10.536363,
    "Ackley5Mixed": 0.0,
    "Rosenbrock5Mixed": 0.0,
}


@pytest.mark.parametrize("name, dim, num_constraints", PROBLEMS)
def test_instantiation(name, dim, num_constraints):
    p = bocode.get_problem(name)()
    assert p.dim == dim
    assert p.num_objectives == 1
    assert p.num_constraints == num_constraints
    assert p.is_constrained == (num_constraints > 0)
    assert len(p.bounds) == dim


@pytest.mark.parametrize("name, dim, num_constraints", PROBLEMS)
def test_evaluate_shapes_and_finiteness(name, dim, num_constraints):
    p = bocode.get_problem(name)()
    X = p.sample(16, seed=0)
    values, constraints = p.evaluate(X)

    assert values.shape == (16, 1)
    assert constraints.shape == (16, num_constraints)
    assert torch.is_floating_point(values)
    assert torch.isfinite(values).all()
    assert torch.isfinite(constraints).all()


@pytest.mark.parametrize("name", NAMES)
def test_evaluate_is_deterministic(name):
    p = bocode.get_problem(name)()
    X = p.sample(8, seed=1)
    v1, c1 = p.evaluate(X)
    v2, c2 = p.evaluate(X)
    assert torch.equal(v1, v2)
    assert torch.equal(c1, c2)


@pytest.mark.parametrize("name, dim, num_constraints", PROBLEMS)
def test_variable_types_and_bounds_are_consistent(name, dim, num_constraints):
    p = bocode.get_problem(name)()
    vtypes = p.resolved_variable_types()
    assert len(vtypes) == dim
    assert p.is_mixed_variable  # every problem here has a non-continuous dimension

    bounds = p.torch_bounds
    for j, t in enumerate(vtypes):
        lo, hi = float(bounds[j][0]), float(bounds[j][1])
        assert lo < hi
        if isinstance(t, list):
            # a categorical/discrete level list: every level must lie in the bounds
            assert len(t) >= 2
            assert all(lo <= v <= hi for v in t), (
                f"{name} dim {j}: levels outside bounds"
            )


@pytest.mark.parametrize("name", NAMES)
def test_sample_respects_variable_types(name):
    p = bocode.get_problem(name)()
    X = p.sample(32, seed=2)
    assert X.shape == (32, p.dim)
    # sample() already snaps; enforcing again must be a no-op
    assert torch.equal(X, p.enforce_variable_types(X))

    bounds = p.torch_bounds.to(X)
    assert (X >= bounds[:, 0] - 1e-9).all()
    assert (X <= bounds[:, 1] + 1e-9).all()

    for j, t in enumerate(p.resolved_variable_types()):
        if t == "integer":
            assert torch.equal(X[:, j], X[:, j].round())
        elif isinstance(t, list):
            allowed = torch.tensor(t, dtype=X.dtype)
            assert torch.isin(X[:, j], allowed).all()


@pytest.mark.parametrize("name", sorted(REFERENCE_OPTIMA))
def test_random_search_does_not_beat_reference_optimum(name):
    """A wrong port usually shows up as a value *better* than the published f*."""
    p = bocode.get_problem(name)()
    ref = REFERENCE_OPTIMA[name]
    assert p.optimum[0] == pytest.approx(ref, rel=1e-4)

    g = torch.Generator().manual_seed(0)
    b = p.torch_bounds.to(torch.float64)
    X = b[:, 0] + torch.rand(4096, p.dim, generator=g, dtype=torch.float64) * (
        b[:, 1] - b[:, 0]
    )
    X = p.enforce_variable_types(X)
    values, constraints = p.evaluate(X)

    f_min = -values[:, 0]  # back to the minimization sense
    if p.num_constraints:
        feasible = (constraints <= 0).all(dim=1)
        if not feasible.any():
            pytest.skip(f"{name}: no feasible point in the random sample")
        f_min = f_min[feasible]

    # tolerance absorbs the float32 cast in BenchmarkProblem.evaluate
    assert f_min.min().item() >= ref - 1e-3


@pytest.mark.parametrize("name", NAMES)
def test_registered_with_metadata(name):
    assert name in bocode.PROBLEM_REGISTRY
    meta = bocode.get_metadata(name)
    assert meta["name"] == name
    assert meta["module"].startswith("bocode.opt_problems.synthetic_mixed.")
    assert meta["input_type"] in ("mixed", "discrete")
    assert meta["source"], f"{name}: empty Sources block"


def test_styblinski_tang_cat_is_purely_discrete():
    p = bocode.StyblinskiTangCat()
    assert bocode.get_metadata("StyblinskiTangCat")["input_type"] == "discrete"
    assert all(isinstance(t, list) for t in p.resolved_variable_types())


def test_welded_beam_optimum_is_constraint_active():
    """The known steel-arm optimum sits on the shear/bending/buckling boundary."""
    p = bocode.WeldedBeamCategorical()
    X = torch.tensor(
        [[0.205730, 3.470489, 9.036624, 0.205729, 0.0]], dtype=torch.float64
    )
    values, constraints = p.evaluate(X)
    assert -values[0, 0].item() == pytest.approx(1.7249, abs=1e-3)
    # g1 (shear), g2 (bending) and g5 (buckling) are active at the optimum
    for j in (0, 1, 4):
        assert abs(constraints[0, j].item()) < 1e-3
