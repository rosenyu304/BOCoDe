"""Wing weight — light-aircraft wing weight estimation (10-D), continuous or mixed.

The classic 10-dimensional wing-weight function (estimates the weight of a light
aircraft wing). The mixed-variable form (``is_discrete=True``, from the GP+ project,
Bostanabad Research Group) turns the wing area ``Sw`` and the thickness-to-chord
ratio ``tc`` into categorical variables chosen from discrete levels.

Variables (order): ``Sw`` [150,200], ``Wfw`` [220,300], ``A`` [6,10], ``Gamma``
[-10,10] deg (sweep), ``q`` [16,45], ``lambda`` [0.5,1], ``tc`` [0.08,0.18], ``Nz``
[2.5,6], ``Wdg`` [1700,2500], ``Wp`` [0.025,0.08]. Objective (minimize): wing weight.
Unconstrained.

Sources:
A. Yousefpour, Z. Zanjani Foumani, M. Shishehbor, C. Mora, R. Bostanabad. GP+: a Python library for kernel-based learning via Gaussian processes. Advances in Engineering Software, 2024. https://github.com/Bostanabad-Research-Group/GP-Plus
Wing weight function: https://www.sfu.ca/~ssurjano/wingweight.html
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType

_BOUNDS = [
    (150.0, 200.0),  # Sw
    (220.0, 300.0),  # Wfw
    (6.0, 10.0),  # A
    (-10.0, 10.0),  # Gamma [deg]
    (16.0, 45.0),  # q
    (0.5, 1.0),  # lambda
    (0.08, 0.18),  # tc
    (2.5, 6.0),  # Nz
    (1700.0, 2500.0),  # Wdg
    (0.025, 0.08),  # Wp
]
_SW_LEVELS = list(np.linspace(150.0, 200.0, 5))
_TC_LEVELS = list(np.linspace(0.08, 0.18, 3))


class Wing(BenchmarkProblem):
    """Minimize wing weight (10 vars). ``is_discrete`` -> mixed (Sw, tc categorical)."""

    available_dimensions = 10
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, is_discrete: bool = False) -> None:
        if is_discrete:
            self.variable_types = ["continuous"] * 10
            self.variable_types[0] = _SW_LEVELS
            self.variable_types[6] = _TC_LEVELS
            self.input_type = DataType.MIXED
        else:
            self.variable_types = None
        super().__init__(dim=10, num_objectives=1, num_constraints=0, bounds=_BOUNDS)

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        Sw, Wfw, A, Gamma, q, lam, tc, Nz, Wdg, Wp = (x[:, i] for i in range(10))
        g = np.cos(np.deg2rad(Gamma))

        weight = (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / g**2) ** 0.6
            * q**0.006
            * lam**0.04
            * (100.0 * tc / g) ** -0.3
            * (Nz * Wdg) ** 0.49
            + Sw * Wp
        )
        fx = torch.tensor(-weight, dtype=torch.float64).reshape(
            -1, 1
        )  # maximize -weight
        return None, fx
