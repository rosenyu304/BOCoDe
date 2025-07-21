import torch

from ..base import BenchmarkProblem, DataType


class SpeedReducer(BenchmarkProblem):
    """
    Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for
    global engineering optimization. Engineering computations 29(5):464–483.
    """

    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 9

    # 7D objective, 1 constraint, X = n-by-7

    tags = {"single_objective", "constrained", "7D"}

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=1,
            num_constraints=9,
            bounds=[
                (3.6, 2.6),
                (0.8, 0.7),
                (28, 17),
                (8.3, 7.3),
                (8.3, 7.3),
                (3.9, 2.9),
                (5.5, 5),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        gx = torch.zeros((n, self.num_constraints))

        b = X[:, 0]
        m = X[:, 1]
        z = X[:, 2]
        L1 = X[:, 3]
        L2 = X[:, 4]
        d1 = X[:, 5]
        d2 = X[:, 6]

        C1 = 0.7854 * b * m * m
        C2 = 3.3333 * z * z + 14.9334 * z - 43.0934
        C3 = 1.508 * b * (d1 * d1 + d2 * d2)
        C4 = 7.4777 * (d1 * d1 * d1 + d2 * d2 * d2)
        C5 = 0.7854 * (L1 * d1 * d1 + L2 * d2 * d2)

        test_function = -(C1 * C2 - C3 + C4 + C5)

        fx = test_function.reshape(n, self.num_objectives)

        gx[:, 0] = 27 / (b * m * m * z) - 1
        gx[:, 1] = 397.5 / (b * m * m * z * z) - 1
        gx[:, 2] = 1.93 * L1**3 / (m * z * d1**4) - 1
        gx[:, 3] = 1.93 * L2**3 / (m * z * d2**4) - 1
        gx[:, 4] = (
            torch.sqrt((745 * L1 / (m * z)) ** 2 + 1.69 * 1e6) / (110 * d1**3) - 1
        )
        gx[:, 5] = (
            torch.sqrt((745 * L2 / (m * z)) ** 2 + 157.5 * 1e6) / (85 * d2**3) - 1
        )
        gx[:, 6] = m * z / 40 - 1
        gx[:, 7] = 5 * m / (b) - 1
        gx[:, 8] = b / (12 * m) - 1

        return gx, fx
