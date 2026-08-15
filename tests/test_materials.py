import pytest
import torch

import bocode

MATERIALS = ["AgNP", "CrossedBarrel", "P3HT", "Perovskite", "AutoAM", "HOIP"]
# Problems whose underlying quantity should be minimized (loss-like).
MINIMIZE = {"AgNP", "Perovskite", "HOIP"}


def _averaged_dataset(cls):
    """The replicate-averaged dataset the candidate pool is built from."""
    import pandas as pd

    from bocode.opt_problems.materials._dataset_problem import _DATA_DIR

    df = pd.read_csv(_DATA_DIR / cls.csv_name, encoding="utf-8-sig")
    features = cls.feature_columns or [
        c for c in df.columns if c != cls.objective_column
    ]
    return (
        df,
        features,
        df.groupby(features, as_index=False, sort=False)[cls.objective_column].mean(),
    )


@pytest.mark.parametrize("name", MATERIALS)
def test_materials_shapes(name):
    p = bocode.get_problem(name)()
    cand = p.candidates
    assert cand.shape[1] == p.dim
    assert p.values.shape[0] == cand.shape[0]

    values, constraints = p.evaluate(cand[:8])
    assert values.shape == (8, 1)
    assert constraints.shape == (8, 0)


@pytest.mark.parametrize("name", MATERIALS)
def test_candidate_pool_is_one_row_per_unique_input(name):
    """Replicate rows are averaged, so every candidate has a distinct feature vector."""
    cls = bocode.get_problem(name)
    p = cls()
    _, _, averaged = _averaged_dataset(cls)

    assert p.candidates.shape[0] == len(averaged)
    assert torch.unique(p.candidates, dim=0).shape[0] == p.candidates.shape[0]


@pytest.mark.parametrize("name", MATERIALS)
def test_materials_lookup_matches_dataset(name):
    """Evaluating a measured candidate is a no-op: it returns that row's own value.

    Every candidate is a unique feature vector (replicates are averaged), so an
    exact-match query must resolve to exactly that row rather than to some other
    row that happens to be closer under a badly-scaled distance.
    """
    p = bocode.get_problem(name)()
    values, _ = p.evaluate(p.candidates)
    assert torch.allclose(values.flatten(), p.values, atol=1e-9)


@pytest.mark.parametrize("name", MATERIALS)
def test_lookup_is_invariant_to_feature_units(name):
    """The nearest-neighbour lookup must not depend on the units a feature is in.

    Distances are taken in min-max normalized feature space, so rescaling a feature
    (e.g. recording a flow rate in mL/min instead of uL/min) cannot change which
    measured experiment is nearest. A raw-unit ``cdist`` fails this: it is dominated
    by whichever feature carries the largest numeric range.
    """
    p = bocode.get_problem(name)()
    lo, hi = p.candidates.min(dim=0).values, p.candidates.max(dim=0).values
    queries = lo + torch.rand(64, p.dim, dtype=torch.float64) * (hi - lo)
    expected, _ = p.evaluate(queries)

    for j in range(p.dim):
        for factor in (1e-3, 1e3):
            scaled = p.__class__()
            scaled._X = scaled._X.clone()
            scaled._X[:, j] *= factor
            scaled._lo = scaled._X.min(dim=0).values
            scaled._span = (scaled._X.max(dim=0).values - scaled._lo).clamp(min=1e-12)
            scaled._Xn = (scaled._X - scaled._lo) / scaled._span

            q = queries.clone()
            q[:, j] *= factor
            got, _ = scaled.evaluate(q)
            assert torch.allclose(got, expected), f"{name}: dim {j} scaled by {factor}"


@pytest.mark.parametrize("name", MATERIALS)
def test_materials_maximization_sign(name):
    """argmax(values) must be the best candidate (min loss / max target).

    The reference is the *replicate-averaged* objective, since that is what the
    candidate pool stores (one row per unique input vector).
    """
    cls = bocode.get_problem(name)
    p = cls()
    _, _, averaged = _averaged_dataset(cls)
    raw = averaged[cls.objective_column].to_numpy(dtype=float)

    best_idx = int(torch.argmax(p.values))
    if name in MINIMIZE:
        assert raw[best_idx] == pytest.approx(raw.min())
    else:
        assert raw[best_idx] == pytest.approx(raw.max())


def test_crossed_barrel_is_a_discrete_four_factor_grid():
    """CrossedBarrel is a 4-factor discrete design (n=4, theta=9, r=11, t=3 levels)."""
    p = bocode.CrossedBarrel()
    assert p.is_mixed_variable
    levels = p.resolved_variable_types()
    assert [len(v) for v in levels] == [4, 9, 11, 3]
    assert levels[0] == [6.0, 8.0, 10.0, 12.0]
    assert levels[3] == [0.7, 1.05, 1.4]
    # 600 of the 4*9*11*3 = 1188 factor combinations were actually fabricated.
    assert p.candidates.shape[0] == 600

    s = p.sample(8, seed=0)
    assert torch.equal(s, p.enforce_variable_types(s))  # sampling lands on the grid


def test_crossed_barrel_averages_its_triplicates():
    """Each fabricated design was measured 3x; the objective is the triplicate mean."""
    import pandas as pd

    from bocode.opt_problems.materials._dataset_problem import _DATA_DIR

    df = pd.read_csv(_DATA_DIR / "Crossed barrel_dataset.csv", encoding="utf-8-sig")
    assert len(df) == 1800  # 600 designs x 3 replicates

    p = bocode.CrossedBarrel()
    row = p.candidates[0]
    same = (df[["n", "theta", "r", "t"]].to_numpy() == row.numpy()).all(axis=1)
    assert same.sum() == 3
    assert float(p.values[0]) == pytest.approx(df.loc[same, "toughness"].mean())
