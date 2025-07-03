import torch

from ..base import BenchmarkProblem, DataType


class CantileverBeam(BenchmarkProblem):
    """
    Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for
    global engineering optimization. Engineering computations 29(5):464–483
    """

    available_dimensions = 10
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 11

    # 10D objective, 11 constraints, X = n-by-10

    tags = {"single_objective", "constrained", "10D"}

    def __init__(self):
        super().__init__(
            dim=10,
            num_objectives=1,
            num_constraints=11,
            bounds=[
                (1, 5),
                (1, 5),
                (1, 5),
                (1, 5),
                (1, 5),
                (30, 65),
                (30, 65),
                (30, 65),
                (30, 65),
                (30, 65),
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
        x9 = X[:, 8]
        x10 = X[:, 9]

        P = 50000
        E = 2 * 107
        L = 100

        test_function = -(
            x1 * x6 * L + x2 * x7 * L + x3 * x8 * L + x4 * x9 * L + x5 * x10 * L
        )

        fx = test_function.reshape(n, self.num_objectives)

        gx[:, 0] = 600 * P / (x5 * x10 * x10) - 14000
        gx[:, 1] = 6 * P * (L * 2) / (x4 * x9 * x9) - 14000
        gx[:, 2] = 6 * P * (L * 3) / (x3 * x8 * x8) - 14000
        gx[:, 3] = 6 * P * (L * 4) / (x2 * x7 * x7) - 14000
        gx[:, 4] = 6 * P * (L * 5) / (x1 * x6 * x6) - 14000
        gx[:, 5] = P * L**3 * (1 / L + 7 / L + 19 / L + 37 / L + 61 / L) / (3 * E) - 2.7
        gx[:, 6] = x10 / x5 - 20
        gx[:, 7] = x9 / x4 - 20
        gx[:, 8] = x8 / x3 - 20
        gx[:, 9] = x7 / x2 - 20
        gx[:, 10] = x6 / x1 - 20

        return gx, fx
