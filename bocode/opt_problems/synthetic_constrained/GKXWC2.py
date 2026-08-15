"""GKXWC2: 2-D single-objective, 1 inequality constraint (negated to maximize).

Reference:
    Gardner JR, Kusner MJ, Xu ZE, et al (2014) Bayesian optimization with inequality
    constraints. In: ICML, pp 937-945.
    https://github.com/rosenyu304/BOEngineeringBenchmark
"""

import torch

from ...base import BenchmarkProblem


class GKXWC2(BenchmarkProblem):
    """GKXWC2 (2-D) with one trigonometric constraint. Feasible when ``g(x) <= 0``.

    Minimization form: f(x) = sin(x1) + x2, negated here to maximize.
    Constraint: g(x) = sin(x1)*sin(x2) + 0.95 <= 0. Domain [0, 6]^2.
    """

    num_objectives = 1
    available_dimensions = 2
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=1,
            bounds=[(0.0, 6.0), (0.0, 6.0)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        x1, x2 = X[:, 0], X[:, 1]
        fx = -(torch.sin(x1) + x2)
        gx = torch.sin(x1) * torch.sin(x2) + 0.95
        return gx.reshape(-1, 1), fx.reshape(-1, 1)
