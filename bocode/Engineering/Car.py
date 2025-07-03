import torch

from ..base import BenchmarkProblem, DataType


class Car(BenchmarkProblem):
    """
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
    algorithm. Computers & Structures 89(23-24):2325–2336
    """

    # 11D objective, 10 constraints, X = n-by-11

    available_dimensions = 11
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 10

    tags = {"single_objective", "constrained", "11D"}

    def __init__(self):
        super().__init__(
            dim=11,
            num_objectives=1,
            num_constraints=10,
            bounds=[
                (0.5, 1.5),
                (0.45, 1.35),
                (0.5, 1.5),
                (0.5, 1.5),
                (0.5, 1.5),
                (0.5, 1.5),
                (0.5, 1.5),
                (0.192, 0.345),
                (0.192, 0.345),
                (-20, 0),
                (-20, 0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        test_function = -torch.tensor(
            [
                [4.90],
                [6.67],
                [6.98],
                [4.01],
                [1.78],
                [0],
                [2.73],
                [0],
                [0],
                [0],
                [0],
                [1.98],
            ]
        )
        X_1 = torch.cat((X, torch.tensor([[1] * n]).T), 1)
        fx = X_1 @ test_function

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
        x11 = X[:, 10]

        gx[:, 0] = (
            1.16
            - 0.3717 * x2 * x4
            - 0.00931 * x2 * x10
            - 0.484 * x3 * x9
            + 0.01343 * x6 * x10
            - 1
        )

        gx[:, 1] = (
            0.261
            - 0.0159 * x1 * x2
            - 0.188 * x1 * x8
            - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.0008757 * x5 * x10
            + 0.08045 * x6 * x9
            + 0.00139 * x8 * x11
            + 0.00001575 * x10 * x11
        ) - 0.9

        gx[:, 2] = (
            0.214
            + 0.00817 * x5
            - 0.131 * x1 * x8
            - 0.0704 * x1 * x9
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.0208 * x3 * x8
            + 0.121 * x3 * x9
            - 0.00364 * x5 * x6
            + 0.0007715 * x5 * x10
            - 0.0005354 * x6 * x10
            + 0.00121 * x8 * x11
        ) - 0.9

        gx[:, 3] = (
            0.74
            - 0.061 * x2
            - 0.163 * x3 * x8
            + 0.001232 * x3 * x10
            - 0.166 * x7 * x9
            + 0.227 * x2 * x2
            - 0.9
        )

        gx[:, 4] = (
            28.98
            + 3.818 * x3
            - 4.2 * x1 * x2
            + 0.0207 * x5 * x10
            + 6.63 * x6 * x9
            - 7.7 * x7 * x8
            + 0.32 * x9 * x10
            - 32
        )

        gx[:, 5] = (
            33.86
            + 2.95 * x3
            + 0.1792 * x10
            - 5.057 * x1 * x2
            - 11.0 * x2 * x8
            - 0.0215 * x5 * x10
            - 9.98 * x7 * x8
            + 22.0 * x8 * x9
            - 32
        )

        gx[:, 6] = 46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10 - 32

        gx[:, 7] = (
            4.72
            - 0.5 * x4
            - 0.19 * x2 * x3
            - 0.0122 * x4 * x10
            + 0.009325 * x6 * x10
            + 0.000191 * x11**2
            - 4
        )

        gx[:, 8] = (
            10.58
            - 0.674 * x1 * x2
            - 1.95 * x2 * x8
            + 0.02054 * x3 * x10
            - 0.0198 * x4 * x10
            + 0.028 * x6 * x10
            - 9.9
        )

        gx[:, 9] = (
            16.45
            - 0.489 * x3 * x7
            - 0.843 * x5 * x6
            + 0.0432 * x9 * x10
            - 0.0556 * x9 * x11
            - 0.000786 * x11**2
            - 15.7
        )

        return gx, fx
