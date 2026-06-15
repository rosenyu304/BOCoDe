import torch

import bocode


def test_satellite_design_evaluate():
    p = bocode.SatelliteDesign()
    assert p.dim == 4
    assert p.num_constraints == 3
    assert len(p.bounds) == 4

    X = p.sample(8, seed=0)
    values, constraints = p.evaluate(X)
    assert values.shape == (8, 1)
    assert constraints.shape == (8, 3)
    assert torch.isfinite(values).all()
    assert torch.isfinite(constraints).all()
    # objective is negated mass; masses should be positive and physically plausible
    masses = -values.flatten()
    assert (masses > 0).all()
    assert (masses < 1e4).all()
