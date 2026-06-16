"""Sellar problem — the canonical coupled multidisciplinary (MDA) test problem.

Two analysis disciplines exchange coupling variables ``y1`` and ``y2`` through a
feedback loop, so each design evaluation runs an inner fixed-point solve to
multidisciplinary consistency before the objective and constraints are formed.
This is the standard Sellar benchmark used throughout the MDO literature; the
``minimdo`` ``sellar_opt`` application reproduces it.

Disciplines (solved to consistency):
    y1 = z1^2 + z2 + x - 0.2*y2
    y2 = sqrt(y1) + z1 + z2

Objective (minimize):  f = x^2 + z2 + y1 + exp(-y2)
Constraints (<= 0):    g1 = 3.16 - y1,   g2 = y2 - 24

Known optimum f* ~ 3.1834 at (z1, z2, x) ~ (1.978, 0, 0).

Sources:
R. S. Sellar, S. M. Batill, J. E. Renaud. Response surface based, concurrent subspace optimization for multidisciplinary system design. AIAA 96-0714, 1996.
P. Norheim, minimdo (sellar_opt application). https://github.com/norheim/minimdo
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType


def _solve_coupling(z1, z2, x, iters: int = 200):
    """Fixed-point solve of the two-discipline Sellar coupling."""
    y1 = np.ones_like(z1)
    y2 = np.ones_like(z1)
    for _ in range(iters):
        y1 = z1**2 + z2 + x - 0.2 * y2
        y2 = np.sqrt(np.clip(y1, 0.0, None)) + z1 + z2
    return y1, y2


class Sellar(BenchmarkProblem):
    """Minimize the Sellar objective (3 continuous vars, 2 constraints, inner MDA).

    Variables: ``z1`` shared design [-10, 10], ``z2`` shared design [0, 10],
    ``x`` local design [0, 10]. Constraints (feasible <= 0): ``y1 >= 3.16`` and
    ``y2 <= 24`` on the converged coupling variables.
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 2

    def __init__(self) -> None:
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=2,
            bounds=[(-10.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            optimum=[3.18339],  # f* at (z1, z2, x) ~ (1.978, 0, 0)
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        z1, z2, xl = x[:, 0], x[:, 1], x[:, 2]

        y1, y2 = _solve_coupling(z1, z2, xl)

        f = xl**2 + z2 + y1 + np.exp(-y2)
        g1 = 3.16 - y1
        g2 = y2 - 24.0

        gx = torch.tensor(np.stack([g1, g2], axis=1), dtype=torch.float64)
        fx = torch.tensor(-f, dtype=torch.float64).reshape(-1, 1)  # maximize -f
        return gx, fx
