"""MiniAeroWing — minimize cruise drag of a simple wing (geometric-program model).

The classic geometric-programming wing-design model: choose aspect ratio ``A``,
wing area ``S``, and cruise speed ``V`` to minimize total drag, trading parasite
drag (grows with area and speed), induced drag (falls with aspect ratio and speed),
and structural wing weight (grows with aspect ratio, which in turn raises the lift
needed). The total weight ``W`` is a coupling variable — wing weight depends on
``W`` and ``W`` depends on wing weight — so each evaluation solves that fixed point
in closed form before forming the drag. Adapted from the ``minimdo`` ``miniaero``
application (Hoburg & Abbeel geometric-programming aircraft model).

Disciplines:
    Re = rho*V/mu * sqrt(S/A);   Cf = 0.074 / Re^0.2
    W_w = rho_S*S + 8.71e-5 * N/t * A^1.5 * S^0.5 * sqrt(W0*W)   (solved with W = W0 + W_w)
    C_L = 2*W / (rho*V^2*S)
    C_D = CDA/S + k*Cf*Swet/S + C_L^2 / (pi*A*e)
Objective (minimize):  D = 0.5*rho*V^2*C_D*S

Known optimum D* ~ 242.3 N at (A, S, V) ~ (18.2, 5.3, 49.2).

Sources:
W. Hoburg and P. Abbeel. Geometric programming for aircraft design optimization. AIAA Journal 52(11):2414-2426, 2014.
P. Norheim, minimdo (miniaero application). https://github.com/norheim/minimdo
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType

_RHO = 1.23  # air density [kg/m^3]
_MU = 1.78e-5  # dynamic viscosity [kg/(m s)]
_SWET_S = 2.05  # wetted-area ratio
_K = 1.2  # form factor
_T = 0.12  # airfoil thickness ratio
_E = 0.96  # Oswald efficiency
_N = 2.5  # ultimate load factor
_CDA = 0.0306  # non-wing parasite drag area [m^2]
_W0 = 4940.0  # fixed weight (fuselage + payload) [N]
_RHO_S = 45.42  # wing areal weight [N/m^2]


class MiniAeroWing(BenchmarkProblem):
    """Minimize simple-wing cruise drag (3 continuous vars, weight fixed-point).

    Variables: aspect ratio ``A`` [1, 40], wing area ``S`` [1, 40] m^2, cruise
    speed ``V`` [15, 100] m/s. Box-constrained (no inequality constraints).
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            bounds=[(1.0, 40.0), (1.0, 40.0), (15.0, 100.0)],
            optimum=[242.27],  # D* [N] at (A, S, V) ~ (18.2, 5.3, 49.2)
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        A, S, V = x[:, 0], x[:, 1], x[:, 2]

        # total weight: W = W0 + W_w with W_w depending on sqrt(W) -> quadratic in sqrt(W)
        c = 8.71e-5 * _N / _T * A**1.5 * S**0.5 * np.sqrt(_W0)
        s = (c + np.sqrt(c**2 + 4 * (_W0 + _RHO_S * S))) / 2
        W = s**2

        CL = 2 * W / (_RHO * V**2 * S)
        Re = _RHO * V / _MU * np.sqrt(S / A)
        Cf = 0.074 / Re**0.2
        CD = _CDA / S + _K * Cf * _SWET_S + CL**2 / (np.pi * A * _E)
        D = 0.5 * _RHO * V**2 * CD * S

        fx = torch.tensor(-D, dtype=torch.float64).reshape(-1, 1)  # maximize -drag
        return None, fx
