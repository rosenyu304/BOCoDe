"""Hartmann-6 with two dimensions recast as qualitative factors (4 continuous + 2 categorical).

The standard 6-D Hartmann function; dimensions 4 and 5 are discretised into small
level sets presented as unordered categoricals (the same quantitative-to-qualitative
recipe as the LVGP-paper Branin/Goldstein benchmarks). The level sets contain the
coordinates of the continuous optimum (``x4 ~ 0.3117 -> 0.312``,
``x5 ~ 0.6573 -> 0.657``), so the discrete-constrained minimum essentially equals
the continuous one and the categorical choice is consequential.

Objective (minimize), the standard Hartmann-6 form::

    f(x) = - sum_{i=1..4} alpha_i exp( - sum_{j=1..6} A_ij (x_j - P_ij)^2 )

Variables: ``x0..x3`` continuous in ``[0, 1]``; ``x4`` categorical in
``{0.35, 0.257, 0.477, 0.312, 0.657}`` (K=5); ``x5`` categorical in
``{0.150, 0.657, 0.512, 0.741}`` (K=4).

Reference optimum ``f* = -3.32237`` (minimization sense). ``evaluate()`` returns
``-f`` because BoCoDe maximizes.

Sources:
L. C. W. Dixon, G. P. Szego. The global optimisation problem: an introduction. Towards Global Optimisation 2:1-15, 1978 (the Hartmann-6 function).
Y. Zhang, D. W. Apley, W. Chen. Bayesian Optimization for Materials Design with Mixed Quantitative and Qualitative Variables. Scientific Reports 10:4924, 2020 (the quantitative-to-qualitative construction).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem

_ALPHA = [1.0, 1.2, 3.0, 3.2]
_A = [
    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
]
_P = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
]


class HartmannCat(BenchmarkProblem):
    """Hartmann-6 with x4, x5 recast as unordered categoricals (6-D, mixed)."""

    available_dimensions = 6
    num_objectives = 1
    num_constraints = 0

    #: qualitative levels of x4 (K=5) and x5 (K=4), given as their physical values
    LEVELS_X4 = [0.35, 0.257, 0.477, 0.312, 0.657]
    LEVELS_X5 = [0.150, 0.657, 0.512, 0.741]

    def __init__(self) -> None:
        self.variable_types = ["continuous"] * 4 + [
            list(self.LEVELS_X4),
            list(self.LEVELS_X5),
        ]
        super().__init__(
            dim=6,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * 6,
            x_opt=[[0.20169, 0.150011, 0.476874, 0.275332, 0.312, 0.657]],
            optimum=[-3.32237],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        alpha = torch.tensor(_ALPHA, dtype=torch.float64)
        A = torch.tensor(_A, dtype=torch.float64)
        P = torch.tensor(_P, dtype=torch.float64)

        # inner[n, i] = sum_j A_ij (x_nj - P_ij)^2
        inner = (A * (x.unsqueeze(1) - P) ** 2).sum(dim=2)
        f = -(alpha * torch.exp(-inner)).sum(dim=1)

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
