"""Welded beam — single-objective design (continuous or mixed-variable).

The classic single-objective welded-beam design: minimize the fabrication cost of a
beam welded to a rigid support, subject to shear stress, bending stress, buckling,
deflection and geometric constraints. This complements the multi-objective
``WeldedBeam`` already in BoCoDe (a different, 2-objective formulation).

Four design variables: weld thickness ``h``, weld length ``l``, beam height ``t``,
beam thickness ``b``. With ``is_discrete=True`` (the Firefly mixed-variable
formulation) ``h`` and ``l`` are restricted to integer multiples of 0.0065 in;
``t`` and ``b`` stay continuous. With ``is_discrete=False`` all four are continuous.

Objective (minimize): ``f = 1.10471 h^2 l + 0.04811 t b (14 + l)``.
Constraints (feasible <= 0): shear stress, bending stress, ``h <= b``, a cost cap,
``h >= 0.125``, end deflection, and buckling load.

Reference optima: continuous ``f* ~ 1.7249``; mixed (h,l on the 0.0065 grid)
``f* ~ 1.7312`` (Gandomi, Yang & Alavi, 2011). BoTorch provides the continuous
single-objective version as ``WeldedBeamSO``.

Sources:
C. A. Coello Coello. Use of a self-adaptive penalty approach for engineering optimization problems. Computers in Industry 41(2):113-127, 2000.
A. H. Gandomi, X.-S. Yang, A. H. Alavi. Mixed variable structural optimization using Firefly Algorithm. Computers & Structures 89(23-24):2325-2336, 2011.
BoTorch synthetic test functions (WeldedBeamSO): https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType

# Constants (psi, inches, pounds).
_P = 6000.0
_L = 14.0
_E = 30e6
_G = 12e6
_TAU_MAX = 13600.0
_SIGMA_MAX = 30000.0
_DELTA_MAX = 0.25
_STEP = 0.0065  # mixed-variable grid for h, l


def _grid(lo: float, hi: float, step: float) -> list[float]:
    """Integer multiples of ``step`` lying within ``[lo, hi]``."""
    k0 = int(np.ceil(lo / step))
    k1 = int(np.floor(hi / step))
    return [round(step * k, 10) for k in range(k0, k1 + 1)]


class WeldedBeamSO(BenchmarkProblem):
    """Minimize welded-beam fabrication cost (4 vars, 7 constraints).

    Variables: ``h`` [0.125, 5], ``l`` [0.1, 10], ``t`` [0.1, 10], ``b`` [0.1, 5].
    ``is_discrete=True`` snaps ``h, l`` to multiples of 0.0065 (Firefly mixed form).
    """

    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 7

    def __init__(self, is_discrete: bool = False) -> None:
        bounds = [(0.125, 5.0), (0.1, 10.0), (0.1, 10.0), (0.1, 5.0)]
        if is_discrete:
            self.variable_types = [
                _grid(*bounds[0], _STEP),
                _grid(*bounds[1], _STEP),
                "continuous",
                "continuous",
            ]
            self.input_type = DataType.MIXED
        else:
            self.variable_types = None
        super().__init__(dim=4, num_objectives=1, num_constraints=7, bounds=bounds)

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        h, l, t, b = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)

        M = _P * (_L + l / 2.0)
        R = np.sqrt(l**2 / 4.0 + ((h + t) / 2.0) ** 2)
        J = 2.0 * (np.sqrt(2.0) * h * l * (l**2 / 12.0 + ((h + t) / 2.0) ** 2))
        tau_p = _P / (np.sqrt(2.0) * h * l)
        tau_pp = M * R / J
        tau = np.sqrt(tau_p**2 + 2.0 * tau_p * tau_pp * (l / (2.0 * R)) + tau_pp**2)
        sigma = 6.0 * _P * _L / (b * t**2)
        delta = 4.0 * _P * _L**3 / (_E * t**3 * b)
        Pc = (4.013 * _E * np.sqrt(t**2 * b**6 / 36.0) / _L**2) * (
            1.0 - (t / (2.0 * _L)) * np.sqrt(_E / (4.0 * _G))
        )

        g1 = tau - _TAU_MAX
        g2 = sigma - _SIGMA_MAX
        g3 = h - b
        g4 = 0.10471 * h**2 + 0.04811 * t * b * (14.0 + l) - 5.0
        g5 = 0.125 - h
        g6 = delta - _DELTA_MAX
        g7 = _P - Pc

        gx = torch.tensor(
            np.stack([g1, g2, g3, g4, g5, g6, g7], axis=1), dtype=torch.float64
        )
        fx = torch.tensor(-cost, dtype=torch.float64).reshape(-1, 1)  # maximize -cost
        return gx, fx
