import torch

from ..base import BenchmarkProblem, DataType


class CarSideImpact(BenchmarkProblem):
    """
    H. Jain and K. Deb. An evolutionary many-objective optimization algorithm
    using reference-point based nondominated sorting approach,
    part II: Handling constraints and extending to an adaptive approach,
    IEEE Transactions on Evolutionary Computation 18(4):602–622, 2014
    """

    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 10

    # 7D objective, 10 constraints, X = 7-by-dim

    tags = {"multi_objective", "constrained", "continuous", "7D"}

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=3,
            num_constraints=10,
            bounds=[
                (0.5, 1.5),
                (0.45, 1.35),
                (0.5, 1.5),
                (0.5, 1.5),
                (0.875, 2.625),
                (0.4, 1.2),
                (0.4, 1.2),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        F = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        V_MBP = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        V_FD = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6

        fx = torch.zeros((n, self.num_objectives))
        # negate for maximization
        fx[:, 0] = -(
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.78 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )
        fx[:, 1] = -F
        fx[:, 2] = -(0.5 * (V_MBP + V_FD))

        gx = torch.zeros((n, self.num_constraints))
        gx[:, 0] = 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3 - 1
        gx[:, 1] = (
            0.261
            - 0.0159 * x1 * x2
            - 0.06486 * x1
            - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.0154464 * x6
            - 0.32
        )
        gx[:, 2] = (
            0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2**2
            - 0.32
        )
        gx[:, 3] = (
            0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2**2 - 0.32
        )
        gx[:, 4] = 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7 - 32
        gx[:, 5] = (
            33.86
            + 2.95 * x3
            - 5.057 * x1 * x2
            - 3.795 * x2
            - 3.4431 * x7
            + 1.45728
            - 32
        )
        gx[:, 6] = 46.36 - 9.9 * x2 - 4.4505 * x1 - 32
        gx[:, 7] = F - 4
        gx[:, 8] = V_MBP - 9.9
        gx[:, 9] = V_FD - 15.7

        return gx, fx
