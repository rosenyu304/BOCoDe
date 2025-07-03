"""
https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation/
"""

import torch

from ...base import BenchmarkProblem, DataType


class CEC2020_p1(BenchmarkProblem):
    """
    CEC2020 Problem 1
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 9
    num_constraints = 8

    def __init__(self):
        super().__init__(
            dim=9,
            num_objectives=1,
            num_constraints=8,
            #  x_opt=[[0] * 9],
            optimum=[189.31162966],
            bounds=[
                (0, 10),
                (0, 200),
                (0, 100),
                (0, 200),
                (1000, 2000000),
                (0, 600),
                (100, 600),
                (100, 600),
                (100, 900),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = 35 * X[:, 0] ** 0.6 + 35 * X[:, 1] ** 0.6

        # Equality constraints
        h = np.zeros((n_samples, 8))
        h[:, 0] = 200 * X[:, 0] * X[:, 3] - X[:, 2]
        h[:, 1] = 200 * X[:, 1] * X[:, 5] - X[:, 4]
        h[:, 2] = X[:, 2] - 10000 * (X[:, 6] - 100)
        h[:, 3] = X[:, 4] - 10000 * (300 - X[:, 6])
        h[:, 4] = X[:, 2] - 10000 * (600 - X[:, 7])
        h[:, 5] = X[:, 4] - 10000 * (900 - X[:, 8])
        h[:, 6] = (
            X[:, 3] * np.log(np.abs(X[:, 7] - 100) + 1e-8)
            - X[:, 3] * np.log(600 - X[:, 6] + 1e-8)
            - X[:, 7]
            + X[:, 6]
            + 500
        )
        h[:, 7] = (
            X[:, 5] * np.log(np.abs(X[:, 8] - X[:, 6]) + 1e-8)
            - X[:, 5] * np.log(600)
            - X[:, 8]
            + X[:, 6]
            + 600
        )

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p2(BenchmarkProblem):
    """
    CEC2020 Problem 2
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 11
    num_constraints = 9

    def __init__(self):
        super().__init__(
            dim=11,
            num_objectives=1,
            num_constraints=9,
            #  x_opt=[[0] * 11],
            optimum=[7049.036954],
            bounds=[
                (1e4, 0.819e6),
                (1e4, 1.131e6),
                (1e4, 2.05e6),
                (0, 0.05074),
                (0, 0.05074),
                (0, 0.05074),
                (100, 200),
                (100, 300),
                (100, 300),
                (100, 300),
                (100, 400),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = (
            (X[:, 0] / (120 * X[:, 3])) ** 0.6
            + (X[:, 1] / (80 * X[:, 4])) ** 0.6
            + (X[:, 2] / (40 * X[:, 5])) ** 0.6
        )

        # Equality constraints
        h = np.zeros((n_samples, 9))
        h[:, 0] = X[:, 0] - 1e4 * (X[:, 6] - 100)
        h[:, 1] = X[:, 1] - 1e4 * (X[:, 7] - X[:, 6])
        h[:, 2] = X[:, 2] - 1e4 * (500 - X[:, 7])
        h[:, 3] = X[:, 0] - 1e4 * (300 - X[:, 8])
        h[:, 4] = X[:, 1] - 1e4 * (400 - X[:, 9])
        h[:, 5] = X[:, 2] - 1e4 * (600 - X[:, 10])
        h[:, 6] = (
            X[:, 3] * np.log(np.abs(X[:, 8] - 100) + 1e-8)
            - X[:, 3] * np.log(300 - X[:, 6] + 1e-8)
            - X[:, 8]
            - X[:, 6]
            + 400
        )
        h[:, 7] = (
            X[:, 4] * np.log(np.abs(X[:, 9] - X[:, 6]) + 1e-8)
            - X[:, 4] * np.log(np.abs(400 - X[:, 7]) + 1e-8)
            - X[:, 9]
            + X[:, 6]
            - X[:, 7]
            + 400
        )
        h[:, 8] = (
            X[:, 5] * np.log(np.abs(X[:, 10] - X[:, 7]) + 1e-8)
            - X[:, 5] * np.log(100)
            - X[:, 10]
            + X[:, 7]
            + 100
        )

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p3(BenchmarkProblem):
    """
    CEC2020 Problem 3
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 7
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 7],
            optimum=[-4529.1197395],
            bounds=[
                (1000, 2000),
                (0, 100),
                (2000, 4000),
                (0, 100),
                (0, 100),
                (0, 20),
                (0, 200),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = (
            -1.715 * X[:, 0]
            - 0.035 * X[:, 0] * X[:, 5]
            - 4.0565 * X[:, 2]
            - 10.0 * X[:, 1]
            + 0.063 * X[:, 2] * X[:, 4]
        )

        # No equality constraints
        h = (n_samples, 0)

        # Inequality constraints
        g = np.zeros((n_samples, 14))
        g[:, 0] = (
            0.0059553571 * X[:, 5] ** 2 * X[:, 0]
            + 0.88392857 * X[:, 2]
            - 0.1175625 * X[:, 5] * X[:, 0]
            - X[:, 0]
        )
        g[:, 1] = (
            1.1088 * X[:, 0]
            + 0.1303533 * X[:, 0] * X[:, 5]
            - 0.0066033 * X[:, 0] * X[:, 5] ** 2
            - X[:, 2]
        )
        g[:, 2] = (
            6.66173269 * X[:, 5] ** 2
            + 172.39878 * X[:, 4]
            - 56.596669 * X[:, 3]
            - 191.20592 * X[:, 5]
            - 10000
        )
        g[:, 3] = (
            1.08702 * X[:, 5]
            + 0.32175 * X[:, 3]
            - 0.03762 * X[:, 5] ** 2
            - X[:, 4]
            + 56.85075
        )
        g[:, 4] = (
            0.006198 * X[:, 6] * X[:, 3] * X[:, 2]
            + 2462.3121 * X[:, 1]
            - 25.125634 * X[:, 1] * X[:, 3]
            - X[:, 2] * X[:, 3]
        )
        g[:, 5] = (
            161.18996 * X[:, 2] * X[:, 3]
            + 5000.0 * X[:, 1] * X[:, 3]
            - 489510.0 * X[:, 1]
            - X[:, 2] * X[:, 3] * X[:, 6]
        )
        g[:, 6] = 0.33 * X[:, 6] - X[:, 4] + 44.333333
        g[:, 7] = 0.022556 * X[:, 4] - 0.007595 * X[:, 6] - 1.0
        g[:, 8] = 0.00061 * X[:, 2] - 0.0005 * X[:, 0] - 1.0
        g[:, 9] = 0.819672 * X[:, 0] - X[:, 2] + 0.819672
        g[:, 10] = 24500.0 * X[:, 1] - 250.0 * X[:, 1] * X[:, 3] - X[:, 2] * X[:, 3]
        g[:, 11] = (
            1020.4082 * X[:, 3] * X[:, 1]
            + 1.2244898 * X[:, 2] * X[:, 3]
            - 100000.0 * X[:, 1]
        )
        g[:, 12] = (
            6.25 * X[:, 0] * X[:, 5] + 6.25 * X[:, 0] - 7.625 * X[:, 2] - 100000.0
        )
        g[:, 13] = 1.22 * X[:, 2] - X[:, 5] * X[:, 0] - X[:, 0] + 1.0

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p4(BenchmarkProblem):
    """
    CEC2020 Problem 4
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 4
    num_constraints = 4

    def __init__(self):
        super().__init__(
            dim=6,
            num_objectives=1,
            num_constraints=4,
            #  x_opt=[[0] * 6],
            optimum=[-0.38826043623],
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1), (1e-5, 16), (1e-5, 16)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Constants
        k1 = 0.09755988
        k2 = 0.99 * k1
        k3 = 0.0391908
        k4 = 0.9 * k3

        # Objective function
        f = -X[:, 3]

        # Equality constraints
        h = np.zeros((n_samples, 4))
        h[:, 0] = X[:, 0] + k1 * X[:, 1] * X[:, 4] - 1
        h[:, 1] = X[:, 1] - X[:, 0] + k2 * X[:, 1] * X[:, 5]
        h[:, 2] = X[:, 2] + X[:, 0] + k3 * X[:, 2] * X[:, 4] - 1
        h[:, 3] = X[:, 3] - X[:, 2] + X[:, 1] - X[:, 0] + k4 * X[:, 3] * X[:, 5]

        # Inequality constraints
        g = np.zeros((n_samples, 1))
        g[:, 0] = np.sqrt(X[:, 4]) + np.sqrt(X[:, 5]) - 4

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p5(BenchmarkProblem):
    """
    CEC2020 Problem 5
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 9
    num_constraints = 4

    def __init__(self):
        super().__init__(
            dim=9,
            num_objectives=1,
            num_constraints=4,
            #  x_opt=[[0] * 9],
            optimum=[-400.0056],
            bounds=[
                (0, 100),
                (0, 200),
                (0, 100),
                (0, 100),
                (0, 100),
                (0, 100),
                (0, 200),
                (0, 100),
                (0, 200),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = -(
            9 * X[:, 0]
            + 15 * X[:, 1]
            - 6 * X[:, 2]
            - 16 * X[:, 3]
            - 10 * (X[:, 4] + X[:, 5])
        )

        # Inequality constraints
        g = np.zeros((n_samples, 2))
        g[:, 0] = X[:, 8] * X[:, 6] + 2 * X[:, 4] - 2.5 * X[:, 0]
        g[:, 1] = X[:, 8] * X[:, 7] + 2 * X[:, 5] - 1.5 * X[:, 1]

        # Equality constraints
        h = np.zeros((n_samples, 4))
        h[:, 0] = X[:, 6] + X[:, 7] - X[:, 2] - X[:, 3]
        h[:, 1] = X[:, 0] - X[:, 6] - X[:, 4]
        h[:, 2] = X[:, 1] - X[:, 7] - X[:, 5]
        h[:, 3] = X[:, 8] * X[:, 6] + X[:, 8] * X[:, 7] - 3 * X[:, 2] - X[:, 3]

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p6(BenchmarkProblem):
    """
    CEC2020 Problem 6
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 38
    num_constraints = 32

    def __init__(self):
        super().__init__(
            dim=38,
            num_objectives=1,
            num_constraints=32,
            #  x_opt=[[0] * 38],
            optimum=[1.8638304088],
            bounds=[
                (0, 90),
                (0, 150),
                (0, 90),
                (0, 150),
                (0, 90),
                (0, 90),
                (0, 150),
                (0, 90),
                (0, 90),
                (0, 90),
                (0, 150),
                (0, 150),
                (0, 90),
                (0, 90),
                (0, 150),
                (0, 90),
                (0, 150),
                (0, 90),
                (0, 150),
                (0, 90),
                (0, 1),
                (0, 1.2),
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 0.5),
                (0, 1),
                (0, 1),
                (0, 0.5),
                (0, 0.5),
                (0, 0.5),
                (0, 1.2),
                (0, 0.5),
                (0, 1.2),
                (0, 1.2),
                (0, 0.5),
                (0, 1.2),
                (0, 1.2),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = 0.9979 + 0.00432 * X[:, 4] + 0.01517 * X[:, 12]

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        # Equality constraints
        h = np.zeros((n_samples, 32))
        h[:, 0] = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] - 300
        h[:, 1] = X[:, 5] - X[:, 6] - X[:, 7]
        h[:, 2] = X[:, 8] - X[:, 9] - X[:, 10] - X[:, 11]
        h[:, 3] = X[:, 13] - X[:, 14] - X[:, 15] - X[:, 16]
        h[:, 4] = X[:, 17] - X[:, 18] - X[:, 19]
        h[:, 5] = X[:, 4] * X[:, 20] - X[:, 5] * X[:, 21] - X[:, 8] * X[:, 22]
        h[:, 6] = X[:, 4] * X[:, 23] - X[:, 5] * X[:, 24] - X[:, 8] * X[:, 25]
        h[:, 7] = X[:, 4] * X[:, 26] - X[:, 5] * X[:, 27] - X[:, 8] * X[:, 28]
        h[:, 8] = X[:, 12] * X[:, 29] - X[:, 13] * X[:, 30] - X[:, 17] * X[:, 31]
        h[:, 9] = X[:, 12] * X[:, 32] - X[:, 13] * X[:, 33] - X[:, 17] * X[:, 34]
        h[:, 10] = X[:, 12] * X[:, 35] - X[:, 13] * X[:, 36] - X[:, 17] * X[:, 37]
        h[:, 11] = 1 / 3 * X[:, 0] + X[:, 14] * X[:, 30] - X[:, 4] * X[:, 20]
        h[:, 12] = 1 / 3 * X[:, 0] + X[:, 14] * X[:, 33] - X[:, 4] * X[:, 23]
        h[:, 13] = 1 / 3 * X[:, 0] + X[:, 14] * X[:, 36] - X[:, 4] * X[:, 26]
        h[:, 14] = 1 / 3 * X[:, 1] + X[:, 9] * X[:, 22] - X[:, 12] * X[:, 29]
        h[:, 15] = 1 / 3 * X[:, 1] + X[:, 9] * X[:, 25] - X[:, 12] * X[:, 32]
        h[:, 16] = 1 / 3 * X[:, 1] + X[:, 9] * X[:, 28] - X[:, 12] * X[:, 35]
        h[:, 17] = (
            1 / 3 * X[:, 2]
            + X[:, 6] * X[:, 21]
            + X[:, 10] * X[:, 22]
            + X[:, 15] * X[:, 30]
            + X[:, 18] * X[:, 31]
            - 30
        )
        h[:, 18] = (
            1 / 3 * X[:, 2]
            + X[:, 6] * X[:, 24]
            + X[:, 10] * X[:, 25]
            + X[:, 15] * X[:, 33]
            + X[:, 18] * X[:, 34]
            - 50
        )
        h[:, 19] = (
            1 / 3 * X[:, 2]
            + X[:, 6] * X[:, 27]
            + X[:, 10] * X[:, 28]
            + X[:, 15] * X[:, 36]
            + X[:, 18] * X[:, 37]
            - 30
        )
        h[:, 20] = X[:, 20] + X[:, 23] + X[:, 26] - 1
        h[:, 21] = X[:, 21] + X[:, 24] + X[:, 27] - 1
        h[:, 22] = X[:, 22] + X[:, 25] + X[:, 28] - 1
        h[:, 23] = X[:, 29] + X[:, 32] + X[:, 35] - 1
        h[:, 24] = X[:, 30] + X[:, 33] + X[:, 36] - 1
        h[:, 25] = X[:, 31] + X[:, 34] + X[:, 37] - 1
        h[:, 26] = X[:, 24]
        h[:, 27] = X[:, 27]
        h[:, 28] = X[:, 22]
        h[:, 29] = X[:, 36]
        h[:, 30] = X[:, 31]
        h[:, 31] = X[:, 34]

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p7(BenchmarkProblem):
    """
    CEC2020 Problem 7
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 48
    num_constraints = 38

    def __init__(self):
        super().__init__(
            dim=48,
            num_objectives=1,
            num_constraints=38,
            #  x_opt=[[0] * 48],
            optimum=[1.5670451],
            bounds=[
                (0.0, 35.0),
                (0.0, 90.0),
                (0.0, 90.0),
                (0.0, 140.0),
                (0.0, 90.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 90.0),
                (0.0, 90.0),
                (0.0, 90.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 35.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.849999, 1.0),
                (0.0, 30.0),
                (0.849999, 1.0),
                (0.0, 30.0),
                (0.849999, 1.0),
                (0.0, 30.0),
                (0.0, 1.0),
                (0.849999, 1.0),
                (0.0, 30.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 30.0),
                (0.0, 1.0),
                (0.0, 30.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        c = np.array(
            [
                [0.23947, 0.75835],
                [-0.0139904, -0.0661588],
                [0.0093514, 0.0338147],
                [0.0077308, 0.0373349],
                [-0.0005719, 0.0016371],
                [0.0042656, 0.0288996],
            ]
        )

        # Objective function
        f = (
            c[0, 0]
            + (
                c[1, 0]
                + c[2, 0] * X[:, 23]
                + c[3, 0] * X[:, 27]
                + c[4, 0] * X[:, 32]
                + c[5, 0] * X[:, 33]
            )
            * X[:, 4]
            + c[0, 1]
            + (
                c[1, 1]
                + c[2, 1] * X[:, 25]
                + c[3, 1] * X[:, 30]
                + c[4, 1] * X[:, 37]
                + c[5, 1] * X[:, 38]
            )
            * X[:, 12]
        )

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        # Equality constraints
        h = np.zeros((n_samples, 38))
        h[:, 0] = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] - 300
        h[:, 1] = X[:, 5] - X[:, 6] - X[:, 7]
        h[:, 2] = X[:, 8] - X[:, 9] - X[:, 10] - X[:, 11]
        h[:, 3] = X[:, 13] - X[:, 14] - X[:, 15] - X[:, 16]
        h[:, 4] = X[:, 17] - X[:, 18] - X[:, 19]
        h[:, 5] = X[:, 5] * X[:, 20] - X[:, 23] * X[:, 24]
        h[:, 6] = X[:, 13] * X[:, 21] - X[:, 25] * X[:, 26]
        h[:, 7] = X[:, 8] * X[:, 22] - X[:, 27] * X[:, 28]
        h[:, 8] = X[:, 17] * X[:, 29] - X[:, 30] * X[:, 31]
        h[:, 9] = X[:, 24] - X[:, 4] * X[:, 32]
        h[:, 10] = X[:, 28] - X[:, 4] * X[:, 33]
        h[:, 11] = X[:, 34] - X[:, 4] * X[:, 35]
        h[:, 12] = X[:, 36] - X[:, 12] * X[:, 37]
        h[:, 13] = X[:, 26] - X[:, 12] * X[:, 38]
        h[:, 14] = X[:, 31] - X[:, 12] * X[:, 39]
        h[:, 15] = X[:, 24] - X[:, 5] * X[:, 20] - X[:, 8] * X[:, 40]
        h[:, 16] = X[:, 28] - X[:, 5] * X[:, 41] - X[:, 8] * X[:, 22]
        h[:, 17] = X[:, 34] - X[:, 5] * X[:, 42] - X[:, 8] * X[:, 43]
        h[:, 18] = X[:, 36] - X[:, 13] * X[:, 44] - X[:, 17] * X[:, 45]
        h[:, 19] = X[:, 26] - X[:, 13] * X[:, 21] - X[:, 17] * X[:, 46]
        h[:, 20] = X[:, 31] - X[:, 13] * X[:, 47] - X[:, 17] * X[:, 29]
        h[:, 21] = 1 / 3 * X[:, 0] + X[:, 14] * X[:, 44] - X[:, 24]
        h[:, 22] = 1 / 3 * X[:, 0] + X[:, 14] * X[:, 21] - X[:, 28]
        h[:, 23] = 1 / 3 * X[:, 0] + X[:, 14] * X[:, 47] - X[:, 34]
        h[:, 24] = 1 / 3 * X[:, 1] + X[:, 9] * X[:, 40] - X[:, 36]
        h[:, 25] = 1 / 3 * X[:, 1] + X[:, 9] * X[:, 22] - X[:, 26]
        h[:, 26] = 1 / 3 * X[:, 1] + X[:, 9] * X[:, 43] - X[:, 31]
        h[:, 27] = X[:, 32] + X[:, 33] + X[:, 35] - 1
        h[:, 28] = X[:, 20] + X[:, 41] + X[:, 42] - 1
        h[:, 29] = X[:, 40] + X[:, 22] + X[:, 43] - 1
        h[:, 30] = X[:, 37] + X[:, 38] + X[:, 39] - 1
        h[:, 31] = X[:, 44] + X[:, 21] + X[:, 47] - 1
        h[:, 32] = X[:, 45] + X[:, 46] + X[:, 29] - 1
        h[:, 33] = X[:, 42]
        h[:, 34] = X[:, 45]
        h[:, 35] = (
            1 / 3 * X[:, 2]
            + X[:, 6] * X[:, 20]
            + X[:, 10] * X[:, 40]
            + X[:, 15] * X[:, 44]
            + X[:, 18] * X[:, 45]
            - 30
        )
        h[:, 36] = (
            1 / 3 * X[:, 2]
            + X[:, 6] * X[:, 41]
            + X[:, 10] * X[:, 22]
            + X[:, 15] * X[:, 21]
            + X[:, 18] * X[:, 46]
            - 50
        )
        h[:, 37] = (
            1 / 3 * X[:, 2]
            + X[:, 6] * X[:, 42]
            + X[:, 10] * X[:, 43]
            + X[:, 15] * X[:, 47]
            + X[:, 18] * X[:, 29]
            - 30
        )

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p8(BenchmarkProblem):
    """
    CEC2020 Problem 8
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 2
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 2],
            optimum=[2.0],
            bounds=[(0, 1.6), (-0.51, 1.49)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        X[:, 1] = np.round(X[:, 1])

        # Objective function
        f = 2 * X[:, 0] + X[:, 1]

        # Inequality constraints
        g = np.zeros((n_samples, 2))
        g[:, 0] = 1.25 - X[:, 0] ** 2 - X[:, 1]
        g[:, 1] = X[:, 0] + X[:, 1] - 1.6

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p9(BenchmarkProblem):
    """
    CEC2020 Problem 9
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 3
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=1,
            #  x_opt=[[0] * 3],
            optimum=[2.557654574],
            bounds=[(0.5, 1.4), (0.5, 1.4), (-0.51, 1.49)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        X[:, 2] = np.round(X[:, 2])

        # Objective function
        f = -X[:, 2] + 2 * X[:, 0] + X[:, 1]

        # Equality constraints
        h = X[:, 0] - 2 * np.exp(-X[:, 1])

        # Inequality constraints
        g = -X[:, 0] + X[:, 1] + X[:, 2]

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g).unsqueeze(-1),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p10(BenchmarkProblem):
    """
    CEC2020 Problem 10
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 3],
            optimum=[1.0765430833],
            bounds=[(0.2, 1.0), (-2.22554, -1.0), (-0.51, 1.49)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        X[:, 2] = np.round(X[:, 2])

        # Objective function
        f = -0.7 * X[:, 2] + 5 * (X[:, 0] - 0.5) ** 2 + 0.8

        # Inequality constraints
        g = np.zeros((n_samples, 3))  # Initialize `g` with the appropriate shape
        g[:, 0] = -np.exp(X[:, 0] - 0.2) - X[:, 1]
        g[:, 1] = X[:, 1] + 1.1 * X[:, 2] + 1
        g[:, 2] = X[:, 0] - X[:, 2] - 0.2

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p11(BenchmarkProblem):
    """
    CEC2020 Problem 11
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 7
    num_constraints = 4

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=1,
            num_constraints=4,
            #  x_opt=[[0] * 7],
            optimum=[99.238463653],
            bounds=[
                (0, 20),
                (0, 20),
                (0, 10),
                (0, 10),
                (-0.51, 1.49),
                (-0.51, 1.49),
                (0, 40),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        x1 = X[:, 0]
        x2 = X[:, 1]
        v1 = X[:, 2]
        v2 = X[:, 3]
        y1 = np.round(X[:, 4])
        y2 = np.round(X[:, 5])
        x_ = X[:, 6]

        z1 = 0.9 * (1 - np.exp(-0.5 * v1)) * x1
        z2 = 0.8 * (1 - np.exp(-0.4 * v2)) * x2

        # Objective function
        f = 7.5 * y1 + 5.5 * y2 + 7 * v1 + 6 * v2 + 5 * x_

        # Equality constraints
        h = np.zeros((n_samples, 4))
        h[:, 0] = y1 + y2 - 1
        h[:, 1] = z1 + z2 - 10
        h[:, 2] = x1 + x2 - x_
        h[:, 3] = z1 * y1 + z2 * y2 - 10

        # Inequality constraints
        g = np.zeros((n_samples, 4))
        g[:, 0] = v1 - 10 * y1
        g[:, 1] = v2 - 10 * y2
        g[:, 2] = x1 - 20 * y1
        g[:, 3] = x2 - 20 * y2

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p12(BenchmarkProblem):
    """
    CEC2020 Problem 12
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 7
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 7],
            optimum=[2.9248305537],
            bounds=[
                (0, 100),
                (0, 100),
                (0, 100),
                (-0.51, 1.49),
                (-0.51, 1.49),
                (-0.51, 1.49),
                (-0.51, 1.49),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        y1 = np.round(X[:, 3])
        y2 = np.round(X[:, 4])
        y3 = np.round(X[:, 5])
        y4 = np.round(X[:, 6])

        # Objective function
        f = (
            (y1 - 1) ** 2
            + (y2 - 1) ** 2
            + (y3 - 1) ** 2
            - np.log(y4 + 1)
            + (x1 - 1) ** 2
            + (x2 - 2) ** 2
            + (x3 - 3) ** 2
        )

        # Inequality constraints
        g = np.zeros((n_samples, 9))
        g[:, 0] = x1 + x2 + x3 + y1 + y2 + y3 - 5
        g[:, 1] = y3**2 + x1**2 + x2**2 + x3**2 - 5.5
        g[:, 2] = x1 + y1 - 1.2
        g[:, 3] = x2 + y2 - 1.8
        g[:, 4] = x3 + y3 - 2.5
        g[:, 5] = x1 + y4 - 1.2
        g[:, 6] = y2**2 + x2**2 - 1.64
        g[:, 7] = y3**2 + x3**2 - 4.25
        g[:, 8] = y2**2 + x3**2 - 4.64

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p13(BenchmarkProblem):
    """
    CEC2020 Problem 13
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 5
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 5],
            optimum=[26887.0],
            bounds=[(27, 45), (27, 45), (27, 45), (77.51, 102.49), (32.51, 45.49)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        y1 = np.round(X[:, 3])
        y2 = np.round(X[:, 4])

        # Objective function
        f = -5.357854 * x1**2 - 0.835689 * y1 * x3 - 37.29329 * y1 + 40792.141

        a = np.array(
            [
                85.334407,
                0.0056858,
                0.0006262,
                0.0022053,
                80.51249,
                0.0071317,
                0.0029955,
                0.0021813,
                9.300961,
                0.0047026,
                0.0012547,
                0.0019085,
            ]
        )

        # Inequality constraints
        g = np.zeros((n_samples, 3))
        g[:, 0] = a[0] + a[1] * y2 * x3 + a[2] * y1 * x2 - a[3] * y1**2 * x3 - 92
        g[:, 1] = a[4] + a[5] * y2 * x3 + a[6] * y1 * x2 + a[7] * x1**2 - 90 - 20
        g[:, 2] = a[8] + a[9] * y1 * x2 + a[10] * y1 * x1 + a[11] * x1 * x2 - 20 - 5

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p14(BenchmarkProblem):
    """
    CEC2020 Problem 14
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 10
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=10,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 10],
            optimum=[53638.942722],
            bounds=[
                (0.51, 3.49),
                (0.51, 3.49),
                (0.51, 3.49),
                (250, 2500),
                (250, 2500),
                (250, 2500),
                (6, 20),
                (4, 16),
                (40, 700),
                (10, 450),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Constant
        S = np.array([[2, 3, 4], [4, 6, 3]])
        t = np.array([[8, 20, 8], [16, 4, 4]])
        H = 6000
        alp = 250
        beta = 0.6
        Q1 = 40000
        Q2 = 20000

        # Decision variable
        N1 = np.round(X[:, 0])
        N2 = np.round(X[:, 1])
        N3 = np.round(X[:, 2])
        V1 = X[:, 3]
        V2 = X[:, 4]
        V3 = X[:, 5]
        TL1 = X[:, 6]
        TL2 = X[:, 7]
        B1 = X[:, 8]
        B2 = X[:, 9]

        # Objective function
        f = alp * (N1 * V1**beta + N2 * V2**beta + N3 * V3**beta)

        # Inequality constraints
        g = np.zeros((n_samples, 10))
        g[:, 0] = Q1 * TL1 / B1 + Q2 * TL2 / B2 - H
        g[:, 1] = S[0, 0] * B1 + S[1, 0] * B2 - V1
        g[:, 2] = S[0, 1] * B1 + S[1, 1] * B2 - V2
        g[:, 3] = S[0, 2] * B1 + S[1, 2] * B2 - V3
        g[:, 4] = t[0, 0] - N1 * TL1
        g[:, 5] = t[0, 1] - N2 * TL1
        g[:, 6] = t[0, 2] - N3 * TL1
        g[:, 7] = t[1, 0] - N1 * TL2
        g[:, 8] = t[1, 1] - N2 * TL2
        g[:, 9] = t[1, 2] - N3 * TL2

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p15(BenchmarkProblem):
    """
    CEC2020 Problem 15
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 7
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 7],
            optimum=[2994.4244658],
            bounds=[
                (2.6, 3.6),
                (0.7, 0.8),
                (17, 28),
                (7.3, 8.3),
                (7.3, 8.3),
                (2.9, 3.9),
                (5, 5.5),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = (
            0.7854
            * X[:, 0]
            * X[:, 1] ** 2
            * (3.3333 * X[:, 2] ** 2 + 14.9334 * X[:, 2] - 43.0934)
            - 1.508 * X[:, 0] * (X[:, 5] ** 2 + X[:, 6] ** 2)
            + 7.477 * (X[:, 5] ** 3 + X[:, 6] ** 3)
            + 0.7854 * (X[:, 3] * X[:, 5] ** 2 + X[:, 4] * X[:, 6] ** 2)
        )

        # No equality constraints
        h = np.zeros((n_samples, 0))

        # Inequality constraints
        g = np.zeros((n_samples, 11))
        g[:, 0] = -X[:, 0] * X[:, 1] ** 2 * X[:, 2] + 27
        g[:, 1] = -X[:, 0] * X[:, 1] ** 2 * X[:, 2] ** 2 + 397.5
        g[:, 2] = -X[:, 1] * X[:, 5] ** 4 * X[:, 2] / X[:, 3] ** 3 + 1.93
        g[:, 3] = -X[:, 1] * X[:, 6] ** 4 * X[:, 2] / X[:, 4] ** 3 + 1.93
        g[:, 4] = (
            10
            * X[:, 5] ** -3
            * np.sqrt(16.91e6 + (745 * X[:, 3] / (X[:, 1] * X[:, 2])) ** 2)
            - 1100
        )
        g[:, 5] = (
            10
            * X[:, 6] ** -3
            * np.sqrt(157.5e6 + (745 * X[:, 4] / (X[:, 1] * X[:, 2])) ** 2)
            - 850
        )
        g[:, 6] = X[:, 1] * X[:, 2] - 40
        g[:, 7] = -X[:, 0] / X[:, 1] + 5
        g[:, 8] = X[:, 0] / X[:, 1] - 12
        g[:, 9] = 1.5 * X[:, 5] - X[:, 3] + 1.9
        g[:, 10] = 1.1 * X[:, 6] - X[:, 4] + 1.9

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p16(BenchmarkProblem):
    """
    CEC2020 Problem 16
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 14
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=14,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 14],
            optimum=[0.032213000814],
            bounds=[(0.001, 5)] * 14,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = (
            63098.88 * X[:, 1] * X[:, 3] * X[:, 11]
            + 5441.5 * X[:, 1] ** 2 * X[:, 11]
            + 115055.5 * X[:, 1] ** 1.664 * X[:, 5]
            + 6172.27 * X[:, 1] ** 2 * X[:, 5]
            + 63098.88 * X[:, 0] * X[:, 2] * X[:, 10]
            + 5441.5 * X[:, 0] ** 2 * X[:, 10]
            + 115055.5 * X[:, 0] ** 1.664 * X[:, 4]
            + 6172.27 * X[:, 0] ** 2 * X[:, 4]
            + 140.53 * X[:, 0] * X[:, 10]
            + 281.29 * X[:, 2] * X[:, 10]
            + 70.26 * X[:, 0] ** 2
            + 281.29 * X[:, 0] * X[:, 2]
            + 281.29 * X[:, 2] ** 2
            + 14437
            * X[:, 7] ** 1.8812
            * X[:, 11] ** 0.3424
            * X[:, 9]
            * X[:, 13] ** -1
            * X[:, 0] ** 2
            * X[:, 6]
            * X[:, 8] ** -1
            + 20470.2 * X[:, 6] ** 2.893 * X[:, 10] ** 0.316 * X[:, 0] ** 2
        )

        # No equality constraints
        h = np.zeros((n_samples, 0))

        # Inequality constraints
        g = np.zeros((n_samples, 15))
        g[:, 0] = 1.524 * X[:, 6] ** -1 - 1
        g[:, 1] = 1.524 * X[:, 7] ** -1 - 1
        g[:, 2] = 0.07789 * X[:, 0] - 2 * X[:, 6] ** -1 * X[:, 8] - 1
        g[:, 3] = (
            7.05305
            * X[:, 8] ** -1
            * X[:, 0] ** 2
            * X[:, 9]
            * X[:, 7] ** -1
            * X[:, 1] ** -1
            * X[:, 13] ** -1
            - 1
        )
        g[:, 4] = 0.0833 / X[:, 12] * X[:, 13] - 1
        g[:, 5] = 0.04771 * X[:, 9] * X[:, 7] ** 1.8812 * X[:, 11] ** 0.3424 - 1
        g[:, 6] = 0.0488 * X[:, 8] * X[:, 6] ** 1.893 * X[:, 10] ** 0.316 - 1
        g[:, 7] = 0.0099 * X[:, 0] / X[:, 2] - 1
        g[:, 8] = 0.0193 * X[:, 1] / X[:, 3] - 1
        g[:, 9] = 0.0298 * X[:, 0] / X[:, 4] - 1
        g[:, 10] = (
            47.136 * X[:, 1] ** 0.333 / (X[:, 9] * X[:, 11])
            - 1.333 * X[:, 7] * X[:, 12] ** 2.1195
            + 62.08 * X[:, 12] ** 2.1195 * X[:, 7] ** 0.2 / (X[:, 11] * X[:, 9])
            - 1
        )
        g[:, 11] = 0.056 * X[:, 1] / X[:, 5] - 1
        g[:, 12] = 2 / X[:, 8] - 1
        g[:, 13] = 2 / X[:, 9] - 1
        g[:, 14] = X[:, 11] / X[:, 10] - 1

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p17(BenchmarkProblem):
    """
    CEC2020 Problem 17
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 3],
            optimum=[0.012665232788],
            bounds=[(0.05, 2), (0.25, 1.3), (2.00, 15.0)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = X[:, 0] ** 2 * X[:, 1] * (X[:, 2] + 2)

        # No equality constraints
        h = np.zeros((n_samples, 0))

        # Inequality constraints
        g = np.zeros((n_samples, 4))
        g[:, 0] = 1 - (X[:, 1] ** 3 * X[:, 2]) / (71785 * X[:, 0] ** 4)
        g[:, 1] = (
            (4 * X[:, 1] ** 2 - X[:, 0] * X[:, 1])
            / (12566 * (X[:, 1] * X[:, 0] ** 3 - X[:, 0] ** 4))
            + 1 / (5108 * X[:, 0] ** 2)
            - 1
        )
        g[:, 2] = 1 - 140.45 * X[:, 0] / (X[:, 1] ** 2 * X[:, 2])
        g[:, 3] = (X[:, 0] + X[:, 1]) / 1.5 - 1

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p18(BenchmarkProblem):
    """
    CEC2020 Problem 18
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 4
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            x_opt=[[0.8125, 0.4375, 42.0984455958549, 176.6365958424394]],
            #  optimum=[6059.714335048436],
            bounds=[(0.0625, 99 * 0.0625), (0.0625, 99 * 0.0625), (10, 200), (10, 200)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        X[:, 0] = 0.0625 * np.round(X[:, 0])
        X[:, 1] = 0.0625 * np.round(X[:, 1])

        # Objective function
        f = (
            0.6224 * X[:, 0] * X[:, 2] * X[:, 3]
            + 1.7781 * X[:, 1] * X[:, 2] ** 2
            + 3.1661 * X[:, 0] ** 2 * X[:, 3]
            + 19.84 * X[:, 0] ** 2 * X[:, 2]
        )

        # No equality constraints
        h = np.zeros((n_samples, 0))

        # Inequality constraints
        g = np.zeros((n_samples, 4))
        g[:, 0] = -X[:, 0] + 0.0193 * X[:, 2]
        g[:, 1] = -X[:, 1] + 0.00954 * X[:, 2]
        g[:, 2] = (
            -np.pi * X[:, 2] ** 2 * X[:, 3] - (4 / 3) * np.pi * X[:, 2] ** 3 + 1296000
        )
        g[:, 3] = X[:, 3] - 240

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p19(BenchmarkProblem):
    """
    CEC2020 Problem 19
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 4
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 4],
            optimum=[1.6702177263],
            bounds=[(0.125, 2), (0.1, 10), (0.1, 10), (0.1, 2)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        X[:, 0] = 0.0625 * np.round(X[:, 0])
        X[:, 1] = 0.0625 * np.round(X[:, 1])

        # Objective function
        f = 1.10471 * X[:, 0] ** 2 * X[:, 1] + 0.04811 * X[:, 2] * X[:, 3] * (
            14 + X[:, 1]
        )

        # No equality constraints
        h = np.zeros((n_samples, 0))

        P = 6000
        L = 14
        delta_max = 0.25
        E = 30 * 1e6
        G = 12 * 1e6
        T_max = 13600
        sigma_max = 30000

        Pc = (
            4.013
            * E
            * np.sqrt(X[:, 2] ** 2 * X[:, 3] ** 6 / 30)
            / L**2
            * (1 - X[:, 2] / (2 * L) * np.sqrt(E / (4 * G)))
        )
        sigma = 6 * P * L / (X[:, 3] * X[:, 2] ** 2)
        delta = 6 * P * L**3 / (E * X[:, 2] ** 2 * X[:, 3])
        J = 2 * (
            np.sqrt(2)
            * X[:, 0]
            * X[:, 1]
            * (X[:, 1] ** 2 / 4 + (X[:, 0] + X[:, 2]) ** 2 / 4)
        )
        R = np.sqrt(X[:, 1] ** 2 / 4 + (X[:, 0] + X[:, 2]) ** 2 / 4)
        M = P * (L + X[:, 1] / 2)
        ttt = M * R / J
        tt = P / (np.sqrt(2) * X[:, 0] * X[:, 1])
        t = np.sqrt(tt**2 + 2 * tt * ttt * X[:, 1] / (2 * R) + ttt**2)

        # Inequality constraints
        g = np.zeros((n_samples, 5))
        g[:, 0] = t - T_max
        g[:, 1] = sigma - sigma_max
        g[:, 2] = X[:, 0] - X[:, 3]
        g[:, 3] = delta - delta_max
        g[:, 4] = P - Pc

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p20(BenchmarkProblem):
    """
    CEC2020 Problem 20
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 2
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            #  x_opt=[[0] * 2],
            optimum=[263.89584338],
            bounds=[(0, 1)] * 2,
        )

    def _evaluate_implementation(self, X):
        import numpy as np

        X = X.numpy()

        n_samples = X.shape[0]

        X[:, 0] = 0.0625 * np.round(X[:, 0])
        X[:, 1] = 0.0625 * np.round(X[:, 1])

        # Objective function
        f = (2 * np.sqrt(2) * X[:, 0] + X[:, 1]) * 100

        # No equality constraints
        h = np.zeros((n_samples, 0))

        # Inequality constraints
        g = np.zeros((n_samples, 3))
        g[:, 0] = (np.sqrt(2) * X[:, 0] + X[:, 1]) / (
            np.sqrt(2) * X[:, 0] ** 2 + 2 * X[:, 0] * X[:, 1]
        ) * 2 - 2
        g[:, 1] = X[:, 1] / (np.sqrt(2) * X[:, 0] ** 2 + 2 * X[:, 0] * X[:, 1]) * 2 - 2
        g[:, 2] = 1 / (np.sqrt(2) * X[:, 1] + X[:, 0]) * 2 - 2

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )
