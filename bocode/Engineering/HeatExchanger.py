import torch

from ..base import BenchmarkProblem, DataType


class HeatExchanger(BenchmarkProblem):
    """
    Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for
    global engineering optimization. Engineering computations 29(5):464–483
    """

    available_dimensions = 8
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 6

    # 8D objective, 6 constraints, X = n-by-8

    tags = {"single_objective", "constrained", "8D"}

    def __init__(self):
        super().__init__(
            dim=8,
            num_objectives=1,
            num_constraints=6,
            bounds=[
                (100, 100000),
                (100, 100000),
                (100, 100000),
                (10, 1000),
                (10, 1000),
                (10, 1000),
                (10, 1000),
                (10, 1000),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        gx = torch.zeros((n, self.num_constraints))

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        x8 = X[:, 7]

        test_function = -(x1 + x2 + x3)

        fx = test_function.reshape(n, self.num_objectives)

        gx[:, 0] = 0.0025 * (x4 + x6) - 1
        gx[:, 1] = 0.0025 * (x5 + x7 - x4) - 1
        gx[:, 2] = 0.01 * (x8 - x5) - 1
        gx[:, 3] = 833.33252 * x4 + 100 * x1 - x1 * x6 - 83333.333
        gx[:, 4] = 1250 * x5 + x2 * x4 - x2 * x7 - 125 * x4
        gx[:, 5] = x3 * x5 - 2500 * x5 - x3 * x8 + 125 * 10000

        return gx, fx
