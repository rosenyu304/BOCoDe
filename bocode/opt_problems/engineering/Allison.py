"""Allison problem — a three-discipline coupled MDA test problem.

Three analysis disciplines exchange coupling variables ``y1, y2, y3``. Given the
design variables the couplings are linear, so each evaluation solves the 3x3
linear system to multidisciplinary consistency before forming the objective.
From J. T. Allison's thesis work on coupled-system optimization, reproduced in
the ``minimdo`` ``thesis_allison`` application.

Disciplines (solved to consistency):
    y1 = 0.1*x1*y2 + 0.8*x1*y3 + 2
    y2 = x2*y3 + 2.5
    y3 = 0.1*x3*y1 + 0.8*x3*y2 + 3

Objective (minimize):
    f = (y1 - 0.2)^2 + (1.3*x1)^2 + (1.5*x2)^2 + (1.2*x3)^2

Known optimum f* ~ 0.5698 at x ~ (-0.507, 0.047, 0.179).

Sources:
J. T. Allison. Complex system optimization: a review of analytical target cascading, collaborative optimization, and other formulations. M.S. thesis, University of Michigan, 2004.
P. Norheim, minimdo (thesis_allison application). https://github.com/norheim/minimdo
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem


def _solve_coupling(x1, x2, x3):
    """Solve the linear 3x3 coupling system A(x) y = b for each row."""
    n = x1.shape[0]
    A = np.zeros((n, 3, 3))
    A[:, 0, 0] = 1.0
    A[:, 0, 1] = -0.1 * x1
    A[:, 0, 2] = -0.8 * x1
    A[:, 1, 1] = 1.0
    A[:, 1, 2] = -x2
    A[:, 2, 0] = -0.1 * x3
    A[:, 2, 1] = -0.8 * x3
    A[:, 2, 2] = 1.0
    # b as an explicit (n, 3, 1) stack so np.linalg.solve treats it as a batch of
    # right-hand-side vectors on both numpy 1.x and numpy 2.x (2.0 changed how a
    # 2-D b is broadcast against a stacked a).
    b = np.tile(np.array([2.0, 2.5, 3.0]), (n, 1))[:, :, None]
    y = np.linalg.solve(A, b)[:, :, 0]
    return y[:, 0], y[:, 1], y[:, 2]


class Allison(BenchmarkProblem):
    """Minimize the Allison coupled-system objective (3 continuous vars, inner MDA).

    Variables ``x1, x2, x3`` each in [-1, 1]; unconstrained apart from the box.
    """

    available_dimensions = 3
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            optimum=[0.5698],  # f* at x ~ (-0.507, 0.047, 0.179)
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        y1, _, _ = _solve_coupling(x1, x2, x3)

        f = (y1 - 0.2) ** 2 + (1.3 * x1) ** 2 + (1.5 * x2) ** 2 + (1.2 * x3) ** 2
        fx = torch.tensor(-f, dtype=torch.float64).reshape(-1, 1)  # maximize -f
        return None, fx
