"""JLH1: 2-D single-objective, 1 inequality constraint (negated to maximize).

Reference:
    Jetton C, Li C, Hoyle C (2023) Constrained Bayesian optimization methods using
    regression and classification Gaussian processes as constraints. In: International
    Design Engineering Technical Conferences and Computers and Information in
    Engineering Conference, ASME, p V03BT03A033.
    https://github.com/rosenyu304/BOEngineeringBenchmark
"""

from ...base import BenchmarkProblem


class JLH1(BenchmarkProblem):
    """JLH1 (2-D) with one linear constraint. Feasible when ``g(x) <= 0``.

    Minimization form: f(x) = (x1-0.5)^2 + (x2-0.5)^2, negated here to maximize.
    Constraint: g(x) = x1 + x2 - 0.75 <= 0. Domain [0, 1]^2.
    """

    num_objectives = 1
    available_dimensions = 2
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=1,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        x1, x2 = X[:, 0], X[:, 1]
        fx = -((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2)
        gx = x1 + x2 - 0.75
        return gx.reshape(-1, 1), fx.reshape(-1, 1)
