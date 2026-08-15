"""Helical compression spring — mixed-variable design (Firefly Case III).

Minimize the wire volume of a helical compression spring. The richest of the
Firefly mixed-variable problems: it has one continuous, one integer, and one
discrete (gauge-table) variable simultaneously. Three design variables in the
order ``(D, d, n)``: mean coil diameter ``D`` (continuous), wire diameter ``d``
(one of 42 standard wire gauges), and number of active coils ``n`` (integer).
With ``is_discrete=False`` the gauge/integer restrictions are relaxed to a
continuous box (``d`` continuous in the gauge range, ``n`` continuous).

Objective (minimize):  ``f = (pi^2 / 4) * d^2 * D * (n + 2)``  (wire volume).
Eight constraints (feasible <= 0): shear stress (Wahl factor), free length,
minimum wire diameter, outer-diameter limit, coil ratio ``D/d >= 3``, deflection
under preload, a (degenerate) deflection-balance identity, and the
preload-to-max-load deflection.

Reference optimum (Gandomi, Yang & Alavi 2011, Table 8/9): ``f* ~ 2.6586`` at
``d = 0.283, D = 1.223049, n = 9``.

Sources:
A. H. Gandomi, X.-S. Yang, A. H. Alavi. Mixed variable structural optimization using Firefly Algorithm. Computers & Structures 89(23-24):2325-2336, 2011.
E. Sandgren. Nonlinear integer and discrete programming in mechanical design optimization. Journal of Mechanical Design 112(2):223-229, 1990.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

# Constants (inch / pound / psi).
_PMAX = 1000.0  # maximum working load [lb]
_S = 189000.0  # maximum shear stress [psi]
_G = 11.5e6  # shear modulus [psi]
_LFREE = 14.0  # maximum free length [in]
_DMIN = 0.2  # minimum wire diameter [in]
_DMAX = 3.0  # maximum outside diameter [in]
_PLOAD = 300.0  # preload compression force [lb]
_DELTA_PM = 6.0  # maximum deflection under preload [in]
_DELTA_W = 1.25  # deflection from preload to maximum load [in]

# Table 6: the 42 allowed standard wire diameters [in].
_WIRE_GAUGES = [
    0.009,
    0.0095,
    0.0104,
    0.0118,
    0.0128,
    0.0132,
    0.014,
    0.015,
    0.0162,
    0.0173,
    0.018,
    0.020,
    0.023,
    0.025,
    0.028,
    0.032,
    0.035,
    0.041,
    0.047,
    0.054,
    0.063,
    0.072,
    0.080,
    0.092,
    0.105,
    0.120,
    0.135,
    0.148,
    0.162,
    0.177,
    0.192,
    0.207,
    0.225,
    0.244,
    0.263,
    0.283,
    0.307,
    0.331,
    0.362,
    0.394,
    0.4375,
    0.500,
]


class HelicalSpring(BenchmarkProblem):
    """Minimize helical-spring wire volume (3 vars, 8 constraints; mixed by default).

    Variables (order): ``D`` mean coil diameter [0.6, 3.0] in (continuous),
    ``d`` wire diameter [0.009, 0.5] in (42 gauges if ``is_discrete``), ``n`` active
    coils [1, 70] (integer if ``is_discrete``).
    """

    available_dimensions = 3
    num_objectives = 1
    num_constraints = 8

    def __init__(self, is_discrete: bool = True) -> None:
        bounds = [(0.6, 3.0), (0.009, 0.5), (1.0, 70.0)]
        if is_discrete:
            self.variable_types = ["continuous", list(_WIRE_GAUGES), "integer"]
        else:
            self.variable_types = None
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=8,
            bounds=bounds,
            optimum=[2.6586],  # d=0.283, D=1.223049, n=9
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        D, d, n = x[:, 0], x[:, 1], x[:, 2]

        vol = (np.pi**2 / 4.0) * d**2 * D * (n + 2.0)

        Si = D / d  # coil ratio C
        Cf = (4.0 * Si - 1.0) / (4.0 * Si - 4.0) + 0.615 / Si  # Wahl factor
        K = _G * d**4 / (8.0 * n * D**3)  # spring rate

        g1 = 8.0 * Cf * _PMAX * D / (np.pi * d**3) - _S
        g2 = _PMAX / K + 1.05 * (n + 2.0) * d - _LFREE  # free length - L_max
        g3 = _DMIN - d
        g4 = D - _DMAX
        g5 = 3.0 - Si
        g6 = _PLOAD / K - _DELTA_PM
        # paper's g7 reduces algebraically to (F_preload - P_load)/K, and since the
        # preload force equals P_load it is identically zero (Table 9 lists g7 = 0).
        g7 = np.zeros_like(D)
        g8 = _DELTA_W - (_PMAX - _PLOAD) / K

        gx = torch.tensor(
            np.stack([g1, g2, g3, g4, g5, g6, g7, g8], axis=1), dtype=torch.float64
        )
        fx = torch.tensor(-vol, dtype=torch.float64).reshape(-1, 1)  # maximize -volume
        return gx, fx
