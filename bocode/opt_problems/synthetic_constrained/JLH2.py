"""JLH2: 2-D single-objective, 1 inequality constraint (negated to maximize).

Reference:
    Jetton C, Li C, Hoyle C (2023) Constrained Bayesian optimization methods using
    regression and classification Gaussian processes as constraints. In: International
    Design Engineering Technical Conferences and Computers and Information in
    Engineering Conference, ASME, p V03BT03A033.
    https://github.com/rosenyu304/BOEngineeringBenchmark
"""

import torch

from ...base import BenchmarkProblem


class JLH2(BenchmarkProblem):
    """JLH2 (2-D) with one elliptical constraint. Feasible when ``g(x) <= 0``.

    Minimization form: f(x) = cos(2*x1)*cos(x2) + sin(x1), negated here to maximize.
    Constraint: g(x) = (x1+5)^2/4 + x2^2/100 - 2.5 <= 0.
    Domain: x1 in [-5, 0], x2 in [-5, 5].
    """

    num_objectives = 1
    available_dimensions = 2
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=1,
            bounds=[(-5.0, 0.0), (-5.0, 5.0)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        x1, x2 = X[:, 0], X[:, 1]
        fx = -(torch.cos(2 * x1) * torch.cos(x2) + torch.sin(x1))
        gx = ((x1 + 5) ** 2) / 4 + (x2**2) / 100 - 2.5
        return gx.reshape(-1, 1), fx.reshape(-1, 1)
