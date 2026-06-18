"""Column buckling — Euler critical load of a column, mixed-variable.

The Euler critical buckling load of a slender column, a mixed continuous/categorical
engineering problem from the GP+ project (Bostanabad Research Group). The column
length is continuous; the material modulus, the effective-length factor, and the
section moment of inertia are categorical (a small catalog of standard choices).

Variables (order): ``L`` length [0.5, 1.5] (continuous), ``E`` Young's modulus
(categorical: 73.1, 200.0 GPa), ``K`` effective-length factor (categorical: 0.5,
0.7, 1.0, 2.0), ``I`` moment of inertia (categorical: 9.49, 12.1, 29.5).

Objective (maximize): critical load ``P = pi*E*I / (L*K)^2``. Unconstrained.

Sources:
A. Yousefpour, Z. Zanjani Foumani, M. Shishehbor, C. Mora, R. Bostanabad. GP+: a Python library for kernel-based learning via Gaussian processes. Advances in Engineering Software, 2024. https://github.com/Bostanabad-Research-Group/GP-Plus
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem, DataType

_E_LEVELS = [73.1, 200.0]
_K_LEVELS = [0.5, 0.7, 1.0, 2.0]
_I_LEVELS = [9.49, 12.1, 29.5]


class ColumnBuckling(BenchmarkProblem):
    """Maximize a column's Euler buckling load (4 vars; L continuous, E/K/I categorical).

    ``is_discrete=False`` relaxes E, K, I to continuous ranges.
    """

    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, is_discrete: bool = True) -> None:
        bounds = [
            (0.5, 1.5),
            (min(_E_LEVELS), max(_E_LEVELS)),
            (min(_K_LEVELS), max(_K_LEVELS)),
            (min(_I_LEVELS), max(_I_LEVELS)),
        ]
        if is_discrete:
            self.variable_types = ["continuous", _E_LEVELS, _K_LEVELS, _I_LEVELS]
            self.input_type = DataType.MIXED
        else:
            self.variable_types = None
        super().__init__(dim=4, num_objectives=1, num_constraints=0, bounds=bounds)

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        L, E, K, I = x[:, 0], x[:, 1], x[:, 2], x[:, 3]  # noqa: E741

        P = torch.pi * E * I / (L * K) ** 2
        fx = torch.tensor(P, dtype=torch.float64).reshape(-1, 1)  # maximize load
        return None, fx
