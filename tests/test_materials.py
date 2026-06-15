import pytest
import torch

import bocode

MATERIALS = ["AgNP", "CrossedBarrel", "P3HT", "Perovskite", "AutoAM"]
# Problems whose underlying quantity should be minimized (loss-like).
MINIMIZE = {"AgNP", "Perovskite"}


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
def test_materials_lookup_matches_dataset(name):
    """Evaluating a candidate returns a stored value of an identical-feature row.

    Some datasets contain repeated experiments (identical features, different
    measured values), so the nearest-neighbour lookup may return any matching
    row's value rather than the queried index — but it must be one of them.
    """
    p = bocode.get_problem(name)()
    idx = torch.tensor([0, 5, 17])
    values, _ = p.evaluate(p.candidates[idx])
    for q, v in zip(idx.tolist(), values.flatten(), strict=True):
        same = (p.candidates == p.candidates[q]).all(dim=1)
        assert torch.isclose(p.values[same], v.to(p.values.dtype)).any()


@pytest.mark.parametrize("name", MATERIALS)
def test_materials_maximization_sign(name):
    """argmax(values) must be the best candidate (min loss / max target)."""
    import pandas as pd

    from bocode.opt_problems.materials._dataset_problem import _DATA_DIR

    cls = bocode.get_problem(name)
    p = cls()
    df = pd.read_csv(_DATA_DIR / cls.csv_name, encoding="utf-8-sig")
    raw = df[cls.objective_column].to_numpy(dtype=float)

    best_idx = int(torch.argmax(p.values))
    if name in MINIMIZE:
        assert raw[best_idx] == pytest.approx(raw.min())
    else:
        assert raw[best_idx] == pytest.approx(raw.max())
