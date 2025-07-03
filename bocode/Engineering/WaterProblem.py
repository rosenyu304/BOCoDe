import torch

from ..base import BenchmarkProblem, DataType


class WaterProblem(BenchmarkProblem):
    """
    H. Jain and K. Deb. An evolutionary many-objective optimization algorithm
    using reference-point based nondominated sorting approach,
    part II: Handling constraints and extending to an adaptive approach,
    IEEE Transactions on Evolutionary Computation 18(4):602–622, 2014
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 5
    num_constraints = 7

    # 3D objective, 7 constraints, X = 3-by-dim

    tags = {"multi_objective", "constrained", "continuous", "3D"}

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=5,
            num_constraints=7,
            bounds=[(0.01, 0.45), (0.01, 0.10), (0.01, 0.10)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = torch.zeros((n, self.num_objectives))
        # negate for maximization
        fx[:, 0] = -(106780.37 * (x2 + x3) + 61704.67)
        fx[:, 1] = -(3000.0 * x1)
        fx[:, 2] = -(30570 * 0.02289 * x2 / (0.06 * 2289.0) ** 0.65)
        fx[:, 3] = -(250.0 * 2289.0 * torch.e ** (-39.75 * x2 + 9.9 * x3 + 2.74))
        fx[:, 4] = -(25.0 * ((1.39 / (x1 * x2)) + 4940.0 * x3 - 80.0))

        gx = torch.zeros((n, self.num_constraints))
        gx[:, 0] = 0.00139 / (x1 * x2) + 4.94 * x3 - 0.08 - 1
        gx[:, 1] = 0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986 - 1
        gx[:, 2] = 12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02 - 50000
        gx[:, 3] = 2.098 / (x1 * x2) + 8046.33 * x3 - 696.71 - 16000
        gx[:, 4] = 2.138 / (x1 * x2) + 7883.39 * x3 - 705.04 - 10000
        gx[:, 5] = 0.417 * (x1 * x2) + 1721.26 * x3 - 136.54 - 2000
        gx[:, 6] = 0.164 / (x1 * x2) + 631.13 * x3 - 54.48 - 550

        return gx, fx
