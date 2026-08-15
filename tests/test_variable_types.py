"""Tests for per-variable types, type-aware sampling, and enforcement."""

import torch

import bocode


def test_continuous_problem_is_not_mixed():
    p = bocode.CompressionSpring()
    assert not p.is_mixed_variable
    assert p.resolved_variable_types() == ["continuous"] * p.dim


def test_geartrain_integer_sampling_and_enforcement():
    p = bocode.GearTrain()  # discrete by default
    assert p.is_mixed_variable
    s = p.sample(8, seed=0)
    assert s.shape == (8, 4)
    assert torch.equal(s, s.round())  # all integer
    # values within [12, 60]
    assert (s >= 12).all() and (s <= 60).all()


def test_geartrain_continuous_variant():
    p = bocode.GearTrain(is_discrete=False)
    assert not p.is_mixed_variable
    s = p.sample(8, seed=0)
    assert (s != s.round()).any()  # not all integer


def test_car_categorical_snap():
    p = bocode.Car()
    X = torch.rand(5, 11)
    Xe = p.enforce_variable_types(X)
    allowed = torch.tensor([0.192, 0.345])
    for j in (7, 8):
        assert torch.isin(Xe[:, j], allowed).all()


def test_pressure_vessel_gauge_grid():
    p = bocode.PressureVessel()
    s = p.sample(6, seed=1)
    # first two dims snap to multiples of 0.0625
    grid = s[:, :2] / 0.0625
    assert torch.allclose(grid, grid.round(), atol=1e-6)


def test_harness_projects_mixed_candidates():
    from algorithms._bo_utils import _scale_clamped

    Xb = _scale_clamped(bocode.GearTrain(), torch.rand(4, 4))
    assert torch.equal(Xb, Xb.round())


def test_continuous_sample_is_within_bounds():
    p = bocode.CantileverBeam()
    s = p.sample(10, seed=0)
    b = p.torch_bounds.to(s)
    lo = torch.minimum(b[:, 0], b[:, 1])
    hi = torch.maximum(b[:, 0], b[:, 1])
    assert (s >= lo - 1e-9).all() and (s <= hi + 1e-9).all()
