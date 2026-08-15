"""The firefly-paper engineering problems expose mixed-integer and continuous variants."""

import pytest
import torch

import bocode

FIREFLY = ["GearTrain", "PressureVessel", "SpeedReducer", "Car"]


@pytest.mark.parametrize("name", FIREFLY)
def test_mixed_and_continuous_variants(name):
    cls = bocode.get_problem(name)

    mixed = cls()  # default = original firefly mixed-variable formulation
    cont = cls(is_discrete=False)  # continuous relaxation

    assert mixed.is_mixed_variable
    assert not cont.is_mixed_variable

    # both variants sample and evaluate to finite objectives
    for p in (mixed, cont):
        X = p.sample(5, seed=0)
        assert X.shape == (5, p.dim)
        values, _ = p.evaluate(X)
        assert torch.isfinite(values).all()

    # the mixed variant's sampled design respects the declared discrete structure
    Xm = mixed.sample(5, seed=1)
    assert torch.equal(Xm, mixed.enforce_variable_types(Xm))
