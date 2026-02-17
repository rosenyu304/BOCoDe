"""
https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation/
"""

import torch

from ...base import BenchmarkProblem, DataType
from .helperFuncs import (
    OBJ11,
    function_fitness,
    ConsBar10,
    FE,
    lk,
    check,
)


class CEC2020_p21(BenchmarkProblem):
    """
    CEC2020 Problem 21
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
            #  X_opt=[[0] * 5],
            optimum=[0.2352424579],
            bounds=[(60.0, 80.0), (90.0, 110.0), (1.0, 3.0), (0.0, 1000.0), (2.0, 9.0)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Parameters
        Mf = 3
        Ms = 40
        Iz = 55
        n = 250
        Tmax = 15
        s = 1.5
        delta = 0.5
        Vsrmax = 10
        rho = 0.0000078
        pmax = 1
        mu = 0.6
        Lmax = 30
        delR = 20

        Rsr = 2 / 3 * (X[:, 1] ** 3 - X[:, 0] ** 3) / (X[:, 1] ** 2 * X[:, 0] ** 2)
        Vsr = np.pi * Rsr * n / 30
        A = np.pi * (X[:, 1] ** 2 - X[:, 0] ** 2)
        Prz = X[:, 3] / A
        w = np.pi * n / 30
        Mh = (
            2
            / 3
            * mu
            * X[:, 3]
            * X[:, 4]
            * (X[:, 1] ** 3 - X[:, 0] ** 3)
            / (X[:, 1] ** 2 - X[:, 0] ** 2)
        )
        T = Iz * w / (Mh + Mf)

        # Objective function
        f = np.pi * (X[:, 1] ** 2 - X[:, 0] ** 2) * X[:, 2] * (X[:, 4] + 1) * rho

        # Inequality constraints
        g = np.zeros((n_samples, 8))
        g[:, 0] = -X[:, 1] + X[:, 0] + delR
        g[:, 1] = (X[:, 4] + 1) * (X[:, 2] + delta) - Lmax
        g[:, 2] = Prz - pmax
        g[:, 3] = Prz * Vsr - pmax * Vsrmax
        g[:, 4] = Vsr - Vsrmax
        g[:, 5] = T - Tmax
        g[:, 6] = s * Ms - Mh
        g[:, 7] = -T

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p22(BenchmarkProblem):
    """
    CEC2020 Problem 22
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 9
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=9,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 9],
            optimum=[0.52576870748],
            bounds=[
                (16.51, 96.49),
                (13.51, 54.49),
                (13.51, 51.49),
                (16.51, 46.49),
                (13.51, 51.49),
                (47.51, 124.49),
                (0.51, 3.49),
                (0.51, 6.49),
                (0.51, 6.49),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Parameter initialization
        X = np.round(np.abs(X))
        Pind = np.array([3, 4, 5])
        mind = np.array([1.75, 2, 2.25, 2.5, 2.75, 3.0])

        N1 = X[:, 0]
        N2 = X[:, 1]
        N3 = X[:, 2]
        N4 = X[:, 3]
        N5 = X[:, 4]
        N6 = X[:, 5]

        p = Pind[X[:, 6].astype(int) - 1]
        m1 = mind[X[:, 7].astype(int) - 1]
        m2 = mind[X[:, 8].astype(int) - 1]

        i1 = N6 / N4
        i01 = 3.11

        i2 = N6 * (N1 * N3 + N2 * N4) / (N1 * N3 * (N6 - N4))
        i02 = 1.84

        iR = -(N2 * N6 / (N1 * N3))
        i0R = -3.11

        # Objective function
        f = np.max(np.column_stack([i1 - i01, i2 - i02, iR - i0R]), axis=1)

        Dmax = 220
        dlt22 = 0.5
        dlt33 = 0.5
        dlt55 = 0.5
        dlt35 = 0.5
        dlt34 = 0.5
        dlt56 = 0.5

        beta = np.arccos(
            ((N6 - N3) ** 2 + (N4 + N5) ** 2 - (N3 + N5) ** 2)
            / (2 * (N6 - N3) * (N4 + N5))
        )

        # Inequality constraints
        g = np.zeros((n_samples, 10))
        g[:, 0] = m2 * (N6 + 2.5) - Dmax
        g[:, 1] = m1 * (N1 + N2) + m1 * (N2 + 2) - Dmax
        g[:, 2] = m2 * (N4 + N5) + m2 * (N5 + 2) - Dmax
        g[:, 3] = np.abs(m1 * (N1 + N2) - m2 * (N6 - N3)) - m1 - m2
        g[:, 4] = -((N1 + N2) * np.sin(np.pi / p) - N2 - 2 - dlt22)
        g[:, 5] = -((N6 - N3) * np.sin(np.pi / p) - N3 - 2 - dlt33)
        g[:, 6] = -((N4 + N5) * np.sin(np.pi / p) - N5 - 2 - dlt55)
        g[:, 7] = np.where(
            np.isreal(beta),
            (N3 + N5 + 2 + dlt35) ** 2
            - (
                (N6 - N3) ** 2
                + (N4 + N5) ** 2
                - 2 * (N6 - N3) * (N4 + N5) * np.cos(2 * np.pi / p - beta)
            ),
            1e6,
        )
        g[:, 8] = -(N6 - 2 * N3 - N4 - 4 - 2 * dlt34)
        g[:, 9] = -(N6 - N4 - 2 * N5 - 4 - 2 * dlt56)

        # Equality constraints
        h = np.remainder(N6 - N4, p)

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p23(BenchmarkProblem):
    """
    CEC2020 Problem 23
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 5
    num_constraints = 3

    def __init__(self):
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=3,
            #  X_opt=[[0] * 5],
            optimum=[16.069868725],
            bounds=[(0, 60), (0, 60), (0, 90), (0, 90), (0, 90)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Parameter initialization
        d1 = X[:, 0] * 1e-3
        d2 = X[:, 1] * 1e-3
        d3 = X[:, 2] * 1e-3
        d4 = X[:, 3] * 1e-3
        w = X[:, 4] * 1e-3

        N = 350
        N1 = 750
        N2 = 450
        N3 = 250
        N4 = 150
        rho = 7200
        a = 3
        mu = 0.35
        s = 1.75 * 1e6
        t = 8 * 1e-3

        # Objective function
        f = (
            rho
            * w
            * np.pi
            / 4
            * (
                d1**2 * (1 + (N1 / N) ** 2)
                + d2**2 * (1 + (N2 / N) ** 2)
                + d3**2 * (1 + (N3 / N) ** 2)
                + d4**2 * (1 + (N4 / N) ** 2)
            )
        )

        C1 = np.pi * d1 / 2 * (1 + N1 / N) + (N1 / N - 1) ** 2 * d1**2 / (4 * a) + 2 * a
        C2 = np.pi * d2 / 2 * (1 + N2 / N) + (N2 / N - 1) ** 2 * d2**2 / (4 * a) + 2 * a
        C3 = np.pi * d3 / 2 * (1 + N3 / N) + (N3 / N - 1) ** 2 * d3**2 / (4 * a) + 2 * a
        C4 = np.pi * d4 / 2 * (1 + N4 / N) + (N4 / N - 1) ** 2 * d4**2 / (4 * a) + 2 * a

        R1 = np.exp(mu * (np.pi - 2 * np.arcsin((N1 / N - 1) * d1 / (2 * a))))
        R2 = np.exp(mu * (np.pi - 2 * np.arcsin((N2 / N - 1) * d2 / (2 * a))))
        R3 = np.exp(mu * (np.pi - 2 * np.arcsin((N3 / N - 1) * d3 / (2 * a))))
        R4 = np.exp(mu * (np.pi - 2 * np.arcsin((N4 / N - 1) * d4 / (2 * a))))

        P1 = (
            s
            * t
            * w
            * (1 - np.exp(-mu * (np.pi - 2 * np.arcsin((N1 / N - 1) * d1 / (2 * a)))))
            * np.pi
            * d1
            * N1
            / 60
        )
        P2 = (
            s
            * t
            * w
            * (1 - np.exp(-mu * (np.pi - 2 * np.arcsin((N2 / N - 1) * d2 / (2 * a)))))
            * np.pi
            * d2
            * N2
            / 60
        )
        P3 = (
            s
            * t
            * w
            * (1 - np.exp(-mu * (np.pi - 2 * np.arcsin((N3 / N - 1) * d3 / (2 * a)))))
            * np.pi
            * d3
            * N3
            / 60
        )
        P4 = (
            s
            * t
            * w
            * (1 - np.exp(-mu * (np.pi - 2 * np.arcsin((N4 / N - 1) * d4 / (2 * a)))))
            * np.pi
            * d4
            * N4
            / 60
        )

        # Inequality constraints
        g = np.zeros((n_samples, 8))
        g[:, 0] = -R1 + 2
        g[:, 1] = -R2 + 2
        g[:, 2] = -R3 + 2
        g[:, 3] = -R4 + 2
        g[:, 4] = -P1 + (0.75 * 745.6998)
        g[:, 5] = -P2 + (0.75 * 745.6998)
        g[:, 6] = -P3 + (0.75 * 745.6998)
        g[:, 7] = -P4 + (0.75 * 745.6998)

        # Equality constraints
        h = np.zeros((n_samples, 3))
        h[:, 0] = C1 - C2
        h[:, 1] = C1 - C3
        h[:, 2] = C1 - C4

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p24(BenchmarkProblem):
    """
    CEC2020 Problem 24
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
            #  X_opt=[[0] * 7],
            optimum=[2.5287918415],
            bounds=[
                (10, 150),
                (10, 150),
                (100, 200),
                (0, 50),
                (10, 150),
                (100, 300),
                (1, 3.14),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        a = X[:, 0]
        b = X[:, 1]
        c = X[:, 2]
        e = X[:, 3]
        ff = X[:, 4]
        l = X[:, 5]
        delta = X[:, 6]

        Ymin = 50
        Ymax = 100
        YG = 150
        Zmax = 99.9999

        # Fixed Calculations:
        # alpha_0 = np.arccos(
        #     (a**2 + (l**2 + e**2) - b**2) / (2 * a * np.sqrt(l**2 + e**2))
        # ) + np.arctan(e / l)

        beta_0 = np.arccos(
            (b**2 + (l**2 + e**2) - a**2) / (2 * b * np.sqrt(l**2 + e**2))
        ) - np.arctan(e / l)

        # alpha_m = np.arccos(
        #     (a**2 + ((l - Zmax) ** 2 + e**2) - b**2)
        #     / (2 * a * np.sqrt((l - Zmax) ** 2 + e**2))
        # ) + np.arctan(e / (l - Zmax))

        beta_m = np.arccos(
            (b**2 + ((l - Zmax) ** 2 + e**2) - a**2)
            / (2 * b * np.sqrt((l - Zmax) ** 2 + e**2))
        ) - np.arctan(e / (l - Zmax))

        # Objective function
        f = np.zeros(n_samples)
        for i in range(n_samples):
            f[i] = -OBJ11(X[i, :-1], 2) - OBJ11(X[i, :-1], 1)

        # Inequality constraints
        Yxmin = 2 * (e + ff + c * np.sin(beta_m + delta))
        Yxmax = 2 * (e + ff + c * np.sin(beta_0 + delta))
        g = np.zeros((X.shape[0], 7))
        g[:, 0] = Yxmin - Ymin
        g[:, 1] = -Yxmin
        g[:, 2] = Ymax - Yxmax
        g[:, 3] = Yxmax - YG
        g[:, 4] = l**2 + e**2 - (a + b) ** 2
        g[:, 5] = b**2 - (a - e) ** 2 - (l - Zmax) ** 2
        g[:, 6] = Zmax - l

        # No equality constraints
        h = np.zeros((n_samples, 0))

        tt = np.imag(f) != 0
        f[tt] = 1e4
        tt = np.imag(g) != 0
        g[tt] = 1e4

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p25(BenchmarkProblem):
    """
    CEC2020 Problem 25
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
            #  X_opt=[[0] * 7],
            optimum=[1616.1197651],
            bounds=[
                (10, 150),
                (10, 150),
                (100, 200),
                (0, 50),
                (10, 150),
                (100, 300),
                (1, 3.14),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        R = X[:, 0]
        Ro = X[:, 1]
        mu = X[:, 2]
        Q = X[:, 3]

        # Constants
        gamma = 0.0307
        C = 0.5
        n = -3.55
        C1 = 10.04
        Ws = 101000
        Pmax = 1000
        delTmax = 50
        hmin = 0.001
        gg = 386.4
        N = 750

        P = (np.log10(np.log10(8.122e6 * mu + 0.8)) - C1) / n
        delT = 2 * (10**P - 560)
        Ef = 9336 * Q * gamma * C * delT
        h = (2 * np.pi * N / 60) ** 2 * 2 * np.pi * mu / Ef * (
            R**4 / 4 - Ro**4 / 4
        ) - 1e-5
        Po = (6 * mu * Q / (np.pi * h**3)) * np.log(R / Ro)
        W = np.pi * Po / 2 * (R**2 - Ro**2) / (np.log(R / Ro) - 1e-5)

        # Objective function
        f = (Q * Po / 0.7 + Ef) / 12

        # Inequality constraints
        g = np.zeros((n_samples, 7))
        g[:, 0] = Ws - W
        g[:, 1] = Po - Pmax
        g[:, 2] = delT - delTmax
        g[:, 3] = hmin - h
        g[:, 4] = Ro - R
        g[:, 5] = gamma / (gg * Po) * (Q / (2 * np.pi * R * h)) - 0.001
        g[:, 6] = W / (np.pi * (R**2 - Ro**2) + 1e-5) - 5000

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p26(BenchmarkProblem):
    """
    CEC2020 Problem 26
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 22
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=22,
            num_objectives=1,
            num_constraints=0,
            #  X_opt=[[0] * 22],
            optimum=[35.359231973],
            bounds=[
                (6.51, 76.49),
                (6.51, 76.49),
                (6.51, 76.49),
                (6.51, 76.49),
                (6.51, 76.49),
                (6.51, 76.49),
                (6.51, 76.49),
                (6.51, 76.49),
                (0.51, 4.49),
                (0.51, 4.49),
                (0.51, 4.49),
                (0.51, 4.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
                (0.51, 9.49),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Parameter initialized
        X = np.round(X)

        Np1, Ng1, Np2, Ng2, Np3, Ng3, Np4, Ng4 = (
            X[:, 0],
            X[:, 1],
            X[:, 2],
            X[:, 3],
            X[:, 4],
            X[:, 5],
            X[:, 6],
            X[:, 7],
        )

        Pvalue = np.array([3.175, 5.715, 8.255, 12.7])
        b1 = Pvalue[X[:, 8].astype(int) - 1].T
        b2 = Pvalue[X[:, 9].astype(int) - 1].T
        b3 = Pvalue[X[:, 10].astype(int) - 1].T
        b4 = Pvalue[X[:, 11].astype(int) - 1].T

        XYvalue = np.array([12.7, 25.4, 38.1, 50.8, 63.5, 76.2, 88.9, 101.6, 114.3])
        xp1 = XYvalue[X[:, 12].astype(int) - 1].T
        xg1 = XYvalue[X[:, 13].astype(int) - 1].T
        xg2 = XYvalue[X[:, 14].astype(int) - 1].T
        xg3 = XYvalue[X[:, 15].astype(int) - 1].T
        xg4 = XYvalue[X[:, 16].astype(int) - 1].T
        yp1 = XYvalue[X[:, 17].astype(int) - 1].T
        yg1 = XYvalue[X[:, 18].astype(int) - 1].T
        yg2 = XYvalue[X[:, 19].astype(int) - 1].T
        yg3 = XYvalue[X[:, 20].astype(int) - 1].T
        yg4 = XYvalue[X[:, 21].astype(int) - 1].T

        # Value initialized
        c1 = np.sqrt((xg1 - xp1) ** 2 + (yg1 - yp1) ** 2)
        c2 = np.sqrt((xg2 - xp1) ** 2 + (yg2 - yp1) ** 2)
        c3 = np.sqrt((xg3 - xp1) ** 2 + (yg3 - yp1) ** 2)
        c4 = np.sqrt((xg4 - xp1) ** 2 + (yg4 - yp1) ** 2)

        CRmin = 1.4
        dmin = 25.4
        phi = 20 * np.pi / 180
        W = 55.9
        JR = 0.2
        Km = 1.6
        Ko = 1.5
        Lmax = 127
        sigma_H = 3290
        sigma_N = 2090
        w1 = 5000
        wmin = 245
        wmax = 255
        Cp = 464

        # Objective function
        f = (np.pi / 1000) * (
            b1 * c1**2 * (Np1**2 + Ng1**2) / (Np1 + Ng1) ** 2
            + b2 * c2**2 * (Np2**2 + Ng2**2) / (Np2 + Ng2) ** 2
            + b3 * c3**2 * (Np3**2 + Ng3**2) / (Np3 + Ng3) ** 2
            + b4 * c4**2 * (Np4**2 + Ng4**2) / (Np4 + Ng4) ** 2
        )

        # Inequality constraints
        g = np.zeros((n_samples, 87))
        g[:, 0] = (366000 / (np.pi * w1) + 2 * c1 * Np1 / (Np1 + Ng1)) * (
            ((Np1 + Ng1) ** 2) / (4 * b1 * c1**2 * Np1)
        ) - sigma_N * JR / (0.0167 * W * Ko * Km)
        g[:, 1] = (366000 * Ng1 / (np.pi * w1 * Np1) + 2 * c2 * Np2 / (Np2 + Ng2)) * (
            ((Np2 + Ng2) ** 2) / (4 * b2 * c2**2 * Np2)
        ) - sigma_N * JR / (0.0167 * W * Ko * Km)
        g[:, 2] = (
            366000 * Ng1 * Ng2 / (np.pi * w1 * Np1 * Np2) + 2 * c3 * Np3 / (Np3 + Ng3)
        ) * (((Np3 + Ng3) ** 2) / (4 * b3 * c3**2 * Np3)) - sigma_N * JR / (
            0.0167 * W * Ko * Km
        )
        g[:, 3] = (
            366000 * Ng1 * Ng2 * Ng3 / (np.pi * w1 * Np1 * Np2 * Np3)
            + 2 * c4 * Np4 / (Np4 + Ng4)
        ) * (((Np4 + Ng4) ** 2) / (4 * b4 * c4**2 * Np4)) - sigma_N * JR / (
            0.0167 * W * Ko * Km
        )
        g[:, 4] = (366000 / (np.pi * w1) + 2 * c1 * Np1 / (Np1 + Ng1)) * (
            ((Np1 + Ng1) ** 3) / (4 * b1 * c1**2 * Ng1 * Np1**2)
        ) - (sigma_H / Cp) ** 2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km)
        g[:, 5] = (366000 * Ng1 / (np.pi * w1 * Np1) + 2 * c2 * Np2 / (Np2 + Ng2)) * (
            ((Np2 + Ng2) ** 3) / (4 * b2 * c2**2 * Ng2 * Np2**2)
        ) - (sigma_H / Cp) ** 2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km)
        g[:, 6] = (
            366000 * Ng1 * Ng2 / (np.pi * w1 * Np1 * Np2) + 2 * c3 * Np3 / (Np3 + Ng3)
        ) * (((Np3 + Ng3) ** 3) / (4 * b3 * c3**2 * Ng3 * Np3**2)) - (
            sigma_H / Cp
        ) ** 2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km)
        g[:, 7] = (
            366000 * Ng1 * Ng2 * Ng3 / (np.pi * w1 * Np1 * Np2 * Np3)
            + 2 * c4 * Np4 / (Np4 + Ng4)
        ) * (((Np4 + Ng4) ** 3) / (4 * b4 * c4**2 * Ng4 * Np4**2)) - (
            sigma_H / Cp
        ) ** 2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km)
        g[:, 8] = (
            CRmin * np.pi * np.cos(phi)
            - Np1 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Np1 + (1 / Np1) ** 2)
            - Ng1 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Ng1 + (1 / Ng1) ** 2)
            + np.sin(phi) * (Np1 + Ng1) / 2
        )
        g[:, 9] = (
            CRmin * np.pi * np.cos(phi)
            - Np2 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Np2 + (1 / Np2) ** 2)
            - Ng2 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Ng2 + (1 / Ng2) ** 2)
            + np.sin(phi) * (Np2 + Ng2) / 2
        )
        g[:, 10] = (
            CRmin * np.pi * np.cos(phi)
            - Np3 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Np3 + (1 / Np3) ** 2)
            - Ng3 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Ng3 + (1 / Ng3) ** 2)
            + np.sin(phi) * (Np3 + Ng3) / 2
        )
        g[:, 11] = (
            CRmin * np.pi * np.cos(phi)
            - Np4 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Np4 + (1 / Np4) ** 2)
            - Ng4 * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / Ng4 + (1 / Ng4) ** 2)
            + np.sin(phi) * (Np4 + Ng4) / 2
        )
        g[:, 12] = dmin - 2 * c1 * Np1 / (Np1 + Ng1)
        g[:, 13] = dmin - 2 * c2 * Np2 / (Np2 + Ng2)
        g[:, 14] = dmin - 2 * c3 * Np3 / (Np3 + Ng3)
        g[:, 15] = dmin - 2 * c4 * Np4 / (Np4 + Ng4)
        g[:, 16] = dmin - 2 * c1 * Ng1 / (Np1 + Ng1)
        g[:, 17] = dmin - 2 * c2 * Ng2 / (Np2 + Ng2)
        g[:, 18] = dmin - 2 * c3 * Ng3 / (Np3 + Ng3)
        g[:, 19] = dmin - 2 * c4 * Ng4 / (Np4 + Ng4)
        g[:, 20] = xp1 + ((Np1 + 2) * c1 / (Np1 + Ng1)) - Lmax
        g[:, 21] = xg2 + ((Np2 + 2) * c2 / (Np2 + Ng2)) - Lmax
        g[:, 22] = xg3 + ((Np3 + 2) * c3 / (Np3 + Ng3)) - Lmax
        g[:, 23] = xg4 + ((Np4 + 2) * c4 / (Np4 + Ng4)) - Lmax
        g[:, 24] = -xp1 + ((Np1 + 2) * c1 / (Np1 + Ng1))
        g[:, 25] = -xg2 + ((Np2 + 2) * c2 / (Np2 + Ng2))
        g[:, 26] = -xg3 + ((Np3 + 2) * c3 / (Np3 + Ng3))
        g[:, 27] = -xg4 + ((Np4 + 2) * c4 / (Np4 + Ng4))
        g[:, 28] = yp1 + ((Np1 + 2) * c1 / (Np1 + Ng1)) - Lmax
        g[:, 29] = yg2 + ((Np2 + 2) * c2 / (Np2 + Ng2)) - Lmax
        g[:, 30] = yg3 + ((Np3 + 2) * c3 / (Np3 + Ng3)) - Lmax
        g[:, 31] = yg4 + ((Np4 + 2) * c4 / (Np4 + Ng4)) - Lmax
        g[:, 32] = -yp1 + ((Np1 + 2) * c1 / (Np1 + Ng1))
        g[:, 33] = -yg2 + ((Np2 + 2) * c2 / (Np2 + Ng2))
        g[:, 34] = -yg3 + ((Np3 + 2) * c3 / (Np3 + Ng3))
        g[:, 35] = -yg4 + ((Np4 + 2) * c4 / (Np4 + Ng4))
        g[:, 36] = xg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1)) - Lmax
        g[:, 37] = xg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2)) - Lmax
        g[:, 38] = xg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3)) - Lmax
        g[:, 39] = xg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4)) - Lmax
        g[:, 40] = -xg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1))
        g[:, 41] = -xg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2))
        g[:, 42] = -xg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3))
        g[:, 43] = -xg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4))
        g[:, 44] = yg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1)) - Lmax
        g[:, 45] = yg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2)) - Lmax
        g[:, 46] = yg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3)) - Lmax
        g[:, 47] = yg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4)) - Lmax
        g[:, 48] = -yg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1))
        g[:, 49] = -yg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2))
        g[:, 50] = -yg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3))
        g[:, 51] = -yg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4))
        g[:, 52] = (
            (0.945 * c1 - Np1 - Ng1) * (b1 - 5.715) * (b1 - 8.255) * (b1 - 12.70) * (-1)
        )
        g[:, 53] = (
            (0.945 * c2 - Np2 - Ng2) * (b2 - 5.715) * (b2 - 8.255) * (b2 - 12.70) * (-1)
        )
        g[:, 54] = (
            (0.945 * c3 - Np3 - Ng3) * (b3 - 5.715) * (b3 - 8.255) * (b3 - 12.70) * (-1)
        )
        g[:, 55] = (
            (0.945 * c4 - Np4 - Ng4) * (b4 - 5.715) * (b4 - 8.255) * (b4 - 12.70) * (-1)
        )
        g[:, 56] = (
            (0.646 * c1 - Np1 - Ng1) * (b1 - 3.175) * (b1 - 8.255) * (b1 - 12.70) * (+1)
        )
        g[:, 57] = (
            (0.646 * c2 - Np2 - Ng2) * (b2 - 3.175) * (b2 - 8.255) * (b2 - 12.70) * (+1)
        )
        g[:, 58] = (
            (0.646 * c3 - Np3 - Ng3) * (b3 - 3.175) * (b3 - 8.255) * (b3 - 12.70) * (+1)
        )
        g[:, 59] = (
            (0.646 * c4 - Np4 - Ng4) * (b4 - 3.175) * (b4 - 8.255) * (b4 - 12.70) * (+1)
        )
        g[:, 60] = (
            (0.504 * c1 - Np1 - Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 12.70) * (-1)
        )
        g[:, 61] = (
            (0.504 * c2 - Np2 - Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 12.70) * (-1)
        )
        g[:, 62] = (
            (0.504 * c3 - Np3 - Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 12.70) * (-1)
        )
        g[:, 63] = (
            (0.504 * c4 - Np4 - Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 12.70) * (-1)
        )
        g[:, 64] = (
            (0.0 * c1 - Np1 - Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 8.255) * (+1)
        )
        g[:, 65] = (
            (0.0 * c2 - Np2 - Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 8.255) * (+1)
        )
        g[:, 66] = (
            (0.0 * c3 - Np3 - Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 8.255) * (+1)
        )
        g[:, 67] = (
            (0.0 * c4 - Np4 - Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 8.255) * (+1)
        )
        g[:, 68] = (
            (-1.812 * c1 + Np1 + Ng1)
            * (b1 - 5.715)
            * (b1 - 8.255)
            * (b1 - 12.70)
            * (-1)
        )
        g[:, 69] = (
            (-1.812 * c2 + Np2 + Ng2)
            * (b2 - 5.715)
            * (b2 - 8.255)
            * (b2 - 12.70)
            * (-1)
        )
        g[:, 70] = (
            (-1.812 * c3 + Np3 + Ng3)
            * (b3 - 5.715)
            * (b3 - 8.255)
            * (b3 - 12.70)
            * (-1)
        )
        g[:, 71] = (
            (-1.812 * c4 + Np4 + Ng4)
            * (b4 - 5.715)
            * (b4 - 8.255)
            * (b4 - 12.70)
            * (-1)
        )
        g[:, 72] = (
            (-0.945 * c1 + Np1 + Ng1)
            * (b1 - 3.175)
            * (b1 - 8.255)
            * (b1 - 12.70)
            * (+1)
        )
        g[:, 73] = (
            (-0.945 * c2 + Np2 + Ng2)
            * (b2 - 3.175)
            * (b2 - 8.255)
            * (b2 - 12.70)
            * (+1)
        )
        g[:, 74] = (
            (-0.945 * c3 + Np3 + Ng3)
            * (b3 - 3.175)
            * (b3 - 8.255)
            * (b3 - 12.70)
            * (+1)
        )
        g[:, 75] = (
            (-0.945 * c4 + Np4 + Ng4)
            * (b4 - 3.175)
            * (b4 - 8.255)
            * (b4 - 12.70)
            * (+1)
        )
        g[:, 76] = (
            (-0.945 * c4 + Np4 + Ng4) * (b4 - 3.175) * (b4 - 8.255) * (b4 - 12.70) * 1
        )
        g[:, 77] = (
            (-0.646 * c1 + Np1 + Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 12.70) * -1
        )
        g[:, 78] = (
            (-0.646 * c2 + Np2 + Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 12.70) * -1
        )
        g[:, 79] = (
            (-0.646 * c3 + Np2 + Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 12.70) * -1
        )
        g[:, 80] = (
            (-0.646 * c4 + Np3 + Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 12.70) * -1
        )
        g[:, 81] = (
            (-0.504 * c1 + Np1 + Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 8.255) * 1
        )
        g[:, 82] = (
            (-0.504 * c2 + Np2 + Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 8.255) * 1
        )
        g[:, 83] = (
            (-0.504 * c3 + Np3 + Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 8.255) * 1
        )
        g[:, 84] = (
            (-0.504 * c4 + Np4 + Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 8.255) * 1
        )
        g[:, 85] = wmin - w1 * (Np1 * Np2 * Np3 * Np4) / (Ng1 * Ng2 * Ng3 * Ng4)
        g[:, 86] = -wmax + w1 * (Np1 * Np2 * Np3 * Np4) / (Ng1 * Ng2 * Ng3 * Ng4)
        g[np.isinf(g)] = 1e6

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p27(BenchmarkProblem):
    """
    CEC2020 Problem 27
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
            #  X_opt=[[0] * 10],
            optimum=[524.45076066],
            bounds=[
                (0.0000645, 0.005),
            ]
            * 10,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))
        g = np.zeros((n_samples, 3))

        for i in range(n_samples):
            # Objective function
            f[i, 0] = function_fitness(X[i, :])

            # Inequality constraint
            c, _ = ConsBar10(X[i, :])
            g[i, :] = c

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p28(BenchmarkProblem):
    """
    CEC2020 Problem 28
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
            #  X_opt=[[0] * 10],
            optimum=[14614.135715],
            bounds=[
                (125.0, 150.0),
                (10.5, 31.5),
                (4.51, 50.49),
                (0.515, 0.6),
                (0.515, 0.6),
                (0.4, 0.5),
                (0.6, 0.7),
                (0.3, 0.4),
                (0.02, 0.1),
                (0.6, 0.85),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = 160
        d = 90
        Bw = 30
        T = D - d - 2 * X[:, 1]

        Dm = X[:, 0]
        Db = X[:, 1]
        Z = np.round(X[:, 2])
        fi = X[:, 3]
        fo = X[:, 4]
        KDmin = X[:, 5]
        KDmax = X[:, 6]
        eps = X[:, 7]
        e = X[:, 8]
        chi = X[:, 9]

        phi_o = 2 * np.pi - 2 * np.arccos(
            (
                ((D - d) * 0.5 - 0.75 * T) ** 2
                + (0.5 * D - 0.25 * T - Db) ** 2
                - (0.5 * d + 0.25 * T) ** 2
            )
            / (2 * (0.5 * (D - d) - 0.75 * T) * (0.5 * D - 0.25 * T - Db))
        )

        gamma = Db / Dm

        fc = (
            37.91
            * (
                1
                + (
                    1.04
                    * ((1 - gamma) / (1 + gamma)) ** 1.72
                    * (fi * (2 * fo - 1) / (fo * (2 * fi - 1))) ** 0.41
                )
                ** (10 / 3)
            )
            ** (-0.3)
            * (gamma**0.3 * (1 - gamma) ** 1.39 / (1 + gamma) ** (1 / 3))
            * (2 * fi / (2 * fi - 1)) ** 0.41
        )

        # Objective function
        ind = np.where(Db > 25.4)
        f = fc * Z ** (2 / 3) * Db**1.8
        f[ind] = 3.647 * fc[ind] * Z[ind] ** (2 / 3) * Db[ind] ** 1.4

        # Inequality constraints
        g = np.zeros((n_samples, 9))
        g[:, 0] = Z - 1 - phi_o / (2 * np.arcsin(Db / Dm))
        g[:, 1] = KDmin * (D - d) - 2 * Db
        g[:, 2] = 2 * Db - KDmax * (D - d)
        g[:, 3] = chi * Bw - Db
        g[:, 4] = 0.5 * (D + d) - Dm
        g[:, 5] = Dm - (0.5 + e) * (D + d)
        g[:, 6] = eps * Db - 0.5 * (D - Dm - Db)
        g[:, 7] = 0.515 - fi
        g[:, 8] = 0.515 - fo

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p29(BenchmarkProblem):
    """
    CEC2020 Problem 29
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
            #  X_opt=[[0] * 4],
            optimum=[2964895.4173],
            bounds=[(20, 50), (1, 10), (20, 50), (0.1, 60)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Objective function
        f = (
            8.61
            * 1e5
            * X[:, 0] ** 0.5
            * X[:, 1]
            * X[:, 2] ** (-2 / 3)
            * X[:, 3] ** (-1 / 2)
            + 3.69 * 1e4 * X[:, 2]
            + 7.72 * 1e8 * X[:, 0] ** (-1) * X[:, 1] ** 0.219
            - 765.43 * 1e6 * X[:, 0] ** (-1)
        )

        # Inequality constraints
        g = np.zeros((n_samples, 1))
        g[:, 0] = X[:, 3] * X[:, 1] ** (-2) + X[:, 1] ** (-2) - 1

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p30(BenchmarkProblem):
    """
    CEC2020 Problem 30
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
            #  X_opt=[[0] * 3],
            optimum=[2.6138840583],
            bounds=[(0.51, 70.49), (0.6, 3), (0.51, 42.49)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        x1 = np.round(X[:, 0])
        x2 = X[:, 1]
        d = np.array(
            [
                0.009,
                0.0095,
                0.0104,
                0.0118,
                0.0128,
                0.0132,
                0.014,
                0.015,
                0.0162,
                0.0173,
                0.018,
                0.020,
                0.023,
                0.025,
                0.028,
                0.032,
                0.035,
                0.041,
                0.047,
                0.054,
                0.063,
                0.072,
                0.080,
                0.092,
                0.0105,
                0.120,
                0.135,
                0.148,
                0.162,
                0.177,
                0.192,
                0.207,
                0.225,
                0.244,
                0.263,
                0.283,
                0.307,
                0.331,
                0.362,
                0.394,
                0.4375,
                0.500,
            ]
        )

        x3 = d[np.clip(np.round(X[:, 2]).astype(int), 0, len(d) - 1)]
        x3 = x3.reshape(-1)

        # Objective function
        f = (np.pi**2 * x2 * x3**2 * (x1 + 2)) / 4

        cf = (4 * x2 / x3 - 1) / (4 * x2 / x3 - 4) + 0.615 * x3 / x2
        K = (11.5 * 10**6 * x3**4) / (8 * x1 * x2**3)
        lf = 1000 / K + 1.05 * (x1 + 2) * x3
        sigp = 300 / K

        # Inequality constraints
        g = np.zeros((n_samples, 8))
        g[:, 0] = (8000 * cf * x2) / (np.pi * x3**3) - 189000
        g[:, 1] = lf - 14
        g[:, 2] = 0.2 - x3
        g[:, 3] = x2 - 3
        g[:, 4] = 3 - x2 / x3
        g[:, 5] = sigp - 6
        g[:, 6] = sigp + 700 / K + 1.05 * (x1 + 2) * x3 - lf
        g[:, 7] = 1.25 - 700 / K

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p31(BenchmarkProblem):
    """
    CEC2020 Problem 31
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
            #  X_opt=[[0] * 4],
            optimum=[0.0],
            bounds=[(12, 60)] * 4,
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
        x4 = X[:, 3]

        # Objective function
        f = (1 / 6.931 - (x1 * x2) / (x3 * x4)) ** 2

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p32(BenchmarkProblem):
    """
    CEC2020 Problem 32
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
            #  X_opt=[[0] * 5],
            optimum=[-30665.538672],
            bounds=[
                (78.0, 102.0),
                (33.0, 45.0),
                (27.0, 45.0),
                (27.0, 45.0),
                (27.0, 45.0),
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
        x4 = X[:, 3]
        x5 = X[:, 4]

        # Objective function
        f = 5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141

        # Parameters
        G1 = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
        G2 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3**2
        G3 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4

        # Inequality constraints
        g = np.zeros((X.shape[0], 6))
        g[:, 0] = G1 - 92
        g[:, 1] = -G1
        g[:, 2] = G2 - 110
        g[:, 3] = -G2 + 90
        g[:, 4] = G3 - 25
        g[:, 5] = -G3 + 20

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p33(BenchmarkProblem):
    """
    CEC2020 Problem 33
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 30
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=30,
            num_objectives=1,
            num_constraints=0,
            #  X_opt=[[0] * 30],
            optimum=[2.639346497],
            bounds=[(0.001, 1.0)] * 30,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]
        ps = n_samples

        nely = 10
        nelx = 3
        penal = 3
        f = np.zeros((ps, 1))  # Objective function values
        g = np.zeros((ps, nely * nelx))  # Gradient values (flattened)

        for i in range(ps):
            Xsplice = np.array([X[i, 0:10], X[i, 10:20], X[i, 20:30]]).T

            # FE-analysis
            U = FE(3, 10, Xsplice, 3)

            # Objective function and sensitivity analysis
            KE = lk()
            c = 0.0
            # Q: Should this be X or Xsplice in np.zeros_like(X)?
            dc = np.zeros_like(Xsplice)

            for ely in range(nely):
                for elx in range(nelx):
                    n1 = (nely + 1) * (elx - 1) + ely
                    n2 = (nely + 1) * elx + ely
                    Ue = U[
                        [
                            2 * n1 - 1,
                            2 * n1,
                            2 * n2 - 1,
                            2 * n2,
                            2 * n2 + 1,
                            2 * n2 + 2,
                            2 * n1 + 1,
                            2 * n1 + 2,
                        ],
                        0,
                    ]
                    c += Xsplice[ely, elx] ** penal * np.dot(Ue.T, np.dot(KE, Ue))
                    dc[ely, elx] = (
                        -penal
                        * Xsplice[ely, elx] ** (penal - 1)
                        * np.dot(Ue.T, np.dot(KE, Ue))
                    )

            # Filtering of sensitivities
            dc = check(3, 10, 1.5, Xsplice, dc)
            f[i, 0] = c
            g[i, :] = dc.flatten()

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g).unsqueeze(-1),
            -torch.from_numpy(f),
        )
