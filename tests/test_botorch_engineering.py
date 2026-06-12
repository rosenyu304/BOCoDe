import torch

from bocode import (
    BotorchCarSideImpact,
    DiscBrake,
    Penicillin,
    VehicleSafety,
    WeldedBeam,
)


def _general_test(cls):
    problem = cls()
    dim = problem.dim
    X = torch.rand((3, dim))
    X = problem.scale(X)
    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape[0] == 3
    assert fx.shape[1] == problem.num_objectives
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (3, problem.num_constraints)


def test_penicillin():
    _general_test(Penicillin)


def test_vehicle_safety():
    _general_test(VehicleSafety)


def test_botorch_car_side_impact():
    _general_test(BotorchCarSideImpact)


def test_disc_brake():
    _general_test(DiscBrake)


def test_welded_beam():
    _general_test(WeldedBeam)
