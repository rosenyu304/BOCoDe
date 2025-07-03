import torch

from ..base import BenchmarkProblem, DataType


class TwoBarTruss(BenchmarkProblem):
    """
    S. S. Rao. Game theory approach for multiobjective structural optimization.
    Computers and Structures 26(1):119–127, 1987
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 5

    # 2D objective, 5 constraints, X = 2-by-dim

    tags = {"multi_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(
            dim=2, num_objectives=2, num_constraints=5, bounds=[(0, 1)] * 2
        )

    def _evaluate_implementation(self, X):
        X = super().scale(X)

        n = X.size(0)

        rho = 0.283
        h = 100
        P = 10000
        sigma_0 = 20000
        E = 30 * 10**6

        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_lower_bound = 0.1
        x2_lower_bound = 1.0

        fx = torch.zeros((n, self.num_objectives))
        # negate for maximization
        fx[:, 0] = -(2 * rho * h * x2 * (1 + x1**2) ** 0.5)
        fx[:, 1] = -(P * h * (1 + x1**2) ** 1.5 * (1 + x1**4) ** 0.5) / (
            2 * 2**0.5 * E * x1**2 * x2
        )

        gx = torch.zeros((n, self.num_constraints))
        gx[:, 0] = (P * (1 + x1) * (1 + x1**2) ** 0.5) / (
            2 * 2**0.5 * x1 * x2
        ) - sigma_0
        gx[:, 1] = (P * (x1 - 1) * (1 + x1**2) ** 0.5) / (
            2 * 2**0.5 * x1 * x2
        ) - sigma_0
        gx[:, 2] = x1_lower_bound - x1
        gx[:, 3] = x2_lower_bound - x2

        return gx, fx
