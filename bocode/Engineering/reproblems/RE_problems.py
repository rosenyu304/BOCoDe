"""
RE benchmark problems (Tanabe & Ishibuchi, 2020).

16 unconstrained multi-objective problems where original constraints are
reformulated as penalty objectives.

Reference:
    Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective
    Problem Suite" Applied Soft Computing. 89: 106078 (2020)
"""

import math

import torch

from ...base import BenchmarkProblem, DataType


class RE21(BenchmarkProblem):
    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma
        super().__init__(
            dim=4,
            num_objectives=2,
            num_constraints=0,
            bounds=[
                (tmp_val, 3 * tmp_val),
                (math.sqrt(2.0) * tmp_val, 3 * tmp_val),
                (math.sqrt(2.0) * tmp_val, 3 * tmp_val),
                (tmp_val, 3 * tmp_val),
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

        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = L * ((2 * x1) + math.sqrt(2.0) * x2 + torch.sqrt(x3) + x4)
        fx[:, 1] = ((F * L) / E) * (
            (2.0 / x1)
            + (2.0 * math.sqrt(2.0) / x2)
            - (2.0 * math.sqrt(2.0) / x3)
            + (2.0 / x4)
        )

        return gx, fx


class RE22(BenchmarkProblem):
    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self):
        # NOTE: Original has typo "3,10" (tuple) instead of "3.10" (float)
        self.feasible_vals = torch.tensor(
            [
                0.20,
                0.31,
                0.40,
                0.44,
                0.60,
                0.62,
                0.79,
                0.80,
                0.88,
                0.93,
                1.0,
                1.20,
                1.24,
                1.32,
                1.40,
                1.55,
                1.58,
                1.60,
                1.76,
                1.80,
                1.86,
                2.0,
                2.17,
                2.20,
                2.37,
                2.40,
                2.48,
                2.60,
                2.64,
                2.79,
                2.80,
                3.0,
                3.08,
                3.10,
                3.16,
                3.41,
                3.52,
                3.60,
                3.72,
                3.95,
                3.96,
                4.0,
                4.03,
                4.20,
                4.34,
                4.40,
                4.65,
                4.74,
                4.80,
                4.84,
                5.0,
                5.28,
                5.40,
                5.53,
                5.72,
                6.0,
                6.16,
                6.32,
                6.60,
                7.11,
                7.20,
                7.80,
                7.90,
                8.0,
                8.40,
                8.69,
                9.0,
                9.48,
                10.27,
                11.0,
                11.06,
                11.85,
                12.0,
                13.0,
                14.0,
                15.0,
            ]
        )
        super().__init__(
            dim=3,
            num_objectives=2,
            num_constraints=0,
            bounds=[
                (0.2, 15.0),
                (0.0, 20.0),
                (0.0, 40.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        vals = self.feasible_vals.to(X.device)
        idx = torch.abs(vals.unsqueeze(0) - X[:, 0].unsqueeze(1)).argmin(dim=1)
        x1 = vals[idx]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (29.4 * x1) + (0.6 * x2 * x3)

        g0 = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
        g1 = 4.0 - (x3 / x2)
        g0 = torch.where(g0 < 0, -g0, torch.zeros_like(g0))
        g1 = torch.where(g1 < 0, -g1, torch.zeros_like(g1))
        fx[:, 1] = g0 + g1

        return gx, fx


class RE23(BenchmarkProblem):
    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=2,
            num_constraints=0,
            bounds=[
                (1.0, 100.0),
                (1.0, 100.0),
                (10.0, 200.0),
                (10.0, 240.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = 0.0625 * torch.round(X[:, 0])
        x2 = 0.0625 * torch.round(X[:, 1])
        x3 = X[:, 2]
        x4 = X[:, 3]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (
            (0.6224 * x1 * x3 * x4)
            + (1.7781 * x2 * x3 * x3)
            + (3.1661 * x1 * x1 * x4)
            + (19.84 * x1 * x1 * x3)
        )

        g0 = x1 - (0.0193 * x3)
        g1 = x2 - (0.00954 * x3)
        g2 = (
            (math.pi * x3 * x3 * x4)
            + ((4.0 / 3.0) * (math.pi * x3 * x3 * x3))
            - 1296000
        )
        g0 = torch.where(g0 < 0, -g0, torch.zeros_like(g0))
        g1 = torch.where(g1 < 0, -g1, torch.zeros_like(g1))
        g2 = torch.where(g2 < 0, -g2, torch.zeros_like(g2))
        fx[:, 1] = g0 + g1 + g2

        return gx, fx


class RE24(BenchmarkProblem):
    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=2,
            num_objectives=2,
            num_constraints=0,
            bounds=[
                (0.5, 4.0),
                (0.5, 50.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = x1 + (120 * x2)

        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)

        g0 = 1 - (sigma_b / sigma_b_max)
        g1 = 1 - (tau / tau_max)
        g2 = 1 - (delta / delta_max)
        g3 = 1 - (sigma_b / sigma_k)
        g0 = torch.where(g0 < 0, -g0, torch.zeros_like(g0))
        g1 = torch.where(g1 < 0, -g1, torch.zeros_like(g1))
        g2 = torch.where(g2 < 0, -g2, torch.zeros_like(g2))
        g3 = torch.where(g3 < 0, -g3, torch.zeros_like(g3))
        fx[:, 1] = g0 + g1 + g2 + g3

        return gx, fx


class RE25(BenchmarkProblem):
    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self):
        self.feasible_vals = torch.tensor(
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
                0.02,
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
                0.08,
                0.092,
                0.105,
                0.12,
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
                0.5,
            ]
        )
        super().__init__(
            dim=3,
            num_objectives=2,
            num_constraints=0,
            bounds=[
                (1.0, 70.0),
                (0.6, 3.0),
                (0.09, 0.5),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = torch.round(X[:, 0])
        x2 = X[:, 1]
        vals = self.feasible_vals.to(X.device)
        idx = torch.abs(vals.unsqueeze(0) - X[:, 2].unsqueeze(1)).argmin(dim=1)
        x3 = vals[idx]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (math.pi * math.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0

        Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
        Fmax = 1000.0
        S = 189000.0
        G = 11.5 * 1e6
        K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
        lmax = 14.0
        lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
        Fp = 300.0
        sigmaP = Fp / K
        sigmaPM = 6
        sigmaW = 1.25

        g0 = -((8 * Cf * Fmax * x2) / (math.pi * x3 * x3 * x3)) + S
        g1 = -lf + lmax
        g2 = -3 + (x2 / x3)
        g3 = -sigmaP + sigmaPM
        g4 = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
        g5 = sigmaW - ((Fmax - Fp) / K)

        g_list = [g0, g1, g2, g3, g4, g5]
        penalty = torch.zeros(n, device=X.device)
        for g in g_list:
            penalty = penalty + torch.where(g < 0, -g, torch.zeros_like(g))
        fx[:, 1] = penalty

        return gx, fx


class RE31(BenchmarkProblem):
    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=3,
            num_constraints=0,
            bounds=[
                (0.00001, 100.0),
                (0.00001, 100.0),
                (1.0, 3.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = x1 * torch.sqrt(16.0 + (x3 * x3)) + x2 * torch.sqrt(1.0 + x3 * x3)
        fx[:, 1] = (20.0 * torch.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        g0 = 0.1 - fx[:, 0]
        g1 = 100000.0 - fx[:, 1]
        g2 = 100000 - ((80.0 * torch.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g0 = torch.where(g0 < 0, -g0, torch.zeros_like(g0))
        g1 = torch.where(g1 < 0, -g1, torch.zeros_like(g1))
        g2 = torch.where(g2 < 0, -g2, torch.zeros_like(g2))
        fx[:, 2] = g0 + g1 + g2

        return gx, fx


class RE32(BenchmarkProblem):
    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=3,
            num_constraints=0,
            bounds=[
                (0.125, 5.0),
                (0.1, 10.0),
                (0.1, 10.0),
                (0.125, 5.0),
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

        P = 6000
        L = 14
        E = 30 * 1e6
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        fx[:, 1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
        R = torch.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
        J = 2 * math.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (math.sqrt(2) * x1 * x2)
        tmpVar = (
            tauDash * tauDash
            + ((2 * tauDash * tauDashDash * x2) / (2 * R))
            + (tauDashDash * tauDashDash)
        )
        tau = torch.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = (
            4.013
            * E
            * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0)
            / (L * L)
        )
        tmpVar2 = (x3 / (2 * L)) * math.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g0 = tauMax - tau
        g1 = sigmaMax - sigma
        g2 = x4 - x1
        g3 = PC - P

        g_list = [g0, g1, g2, g3]
        penalty = torch.zeros(n, device=X.device)
        for g in g_list:
            penalty = penalty + torch.where(g < 0, -g, torch.zeros_like(g))
        fx[:, 2] = penalty

        return gx, fx


class RE33(BenchmarkProblem):
    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=3,
            num_constraints=0,
            bounds=[
                (55.0, 80.0),
                (75.0, 110.0),
                (1000.0, 3000.0),
                (11.0, 20.0),
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

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        fx[:, 1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (
            x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)
        )

        g0 = (x2 - x1) - 20.0
        g1 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g2 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow(
            (x2 * x2 - x1 * x1), 2
        )
        g3 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (
            x2 * x2 - x1 * x1
        ) - 900.0

        g_list = [g0, g1, g2, g3]
        penalty = torch.zeros(n, device=X.device)
        for g in g_list:
            penalty = penalty + torch.where(g < 0, -g, torch.zeros_like(g))
        fx[:, 2] = penalty

        return gx, fx


class RE34(BenchmarkProblem):
    available_dimensions = 5
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=5,
            num_objectives=3,
            num_constraints=0,
            bounds=[(1.0, 3.0)] * 5,
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

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (
            1640.2823
            + (2.3573285 * x1)
            + (2.3220035 * x2)
            + (4.5688768 * x3)
            + (7.7213633 * x4)
            + (4.4559504 * x5)
        )
        fx[:, 1] = (
            6.5856
            + (1.15 * x1)
            - (1.0427 * x2)
            + (0.9738 * x3)
            + (0.8364 * x4)
            - (0.3695 * x1 * x4)
            + (0.0861 * x1 * x5)
            + (0.3628 * x2 * x4)
            - (0.1106 * x1 * x1)
            - (0.3437 * x3 * x3)
            + (0.1764 * x4 * x4)
        )
        fx[:, 2] = (
            -0.0551
            + (0.0181 * x1)
            + (0.1024 * x2)
            + (0.0421 * x3)
            - (0.0073 * x1 * x2)
            + (0.024 * x2 * x3)
            - (0.0118 * x2 * x4)
            - (0.0204 * x3 * x4)
            - (0.008 * x3 * x5)
            - (0.0241 * x2 * x2)
            + (0.0109 * x4 * x4)
        )

        return gx, fx


class RE35(BenchmarkProblem):
    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=3,
            num_constraints=0,
            bounds=[
                (2.6, 3.6),
                (0.7, 0.8),
                (17.0, 28.0),
                (7.3, 8.3),
                (7.3, 8.3),
                (2.9, 3.9),
                (5.0, 5.5),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = torch.round(X[:, 2])
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (
            0.7854
            * x1
            * (x2 * x2)
            * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934)
            - 1.508 * x1 * (x6 * x6 + x7 * x7)
            + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
            + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
        )

        tmpVar = torch.pow((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
        fx[:, 1] = torch.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        g0 = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g1 = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g2 = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g3 = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g4 = -(x2 * x3) + 40.0
        g5 = -(x1 / x2) + 12.0
        g6 = -5.0 + (x1 / x2)
        g7 = -1.9 + x4 - 1.5 * x6
        g8 = -1.9 + x5 - 1.1 * x7
        g9 = -fx[:, 1] + 1300.0
        tmpVar = torch.pow((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g10 = -torch.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0

        g_list = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        penalty = torch.zeros(n, device=X.device)
        for g in g_list:
            penalty = penalty + torch.where(g < 0, -g, torch.zeros_like(g))
        fx[:, 2] = penalty

        return gx, fx


class RE36(BenchmarkProblem):
    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=3,
            num_constraints=0,
            bounds=[(12.0, 60.0)] * 4,
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = torch.round(X[:, 0])
        x2 = torch.round(X[:, 1])
        x3 = torch.round(X[:, 2])
        x4 = torch.round(X[:, 3])

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        fx[:, 1] = torch.stack([x1, x2, x3, x4], dim=1).max(dim=1).values

        g0 = 0.5 - (fx[:, 0] / 6.931)
        g0 = torch.where(g0 < 0, -g0, torch.zeros_like(g0))
        fx[:, 2] = g0

        return gx, fx


class RE37(BenchmarkProblem):
    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 3
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=3,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * 4,
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        xAlpha = X[:, 0]
        xHA = X[:, 1]
        xOA = X[:, 2]
        xOPTT = X[:, 3]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (
            0.692
            + (0.477 * xAlpha)
            - (0.687 * xHA)
            - (0.080 * xOA)
            - (0.0650 * xOPTT)
            - (0.167 * xAlpha * xAlpha)
            - (0.0129 * xHA * xAlpha)
            + (0.0796 * xHA * xHA)
            - (0.0634 * xOA * xAlpha)
            - (0.0257 * xOA * xHA)
            + (0.0877 * xOA * xOA)
            - (0.0521 * xOPTT * xAlpha)
            + (0.00156 * xOPTT * xHA)
            + (0.00198 * xOPTT * xOA)
            + (0.0184 * xOPTT * xOPTT)
        )
        fx[:, 1] = (
            0.153
            - (0.322 * xAlpha)
            + (0.396 * xHA)
            + (0.424 * xOA)
            + (0.0226 * xOPTT)
            + (0.175 * xAlpha * xAlpha)
            + (0.0185 * xHA * xAlpha)
            - (0.0701 * xHA * xHA)
            - (0.251 * xOA * xAlpha)
            + (0.179 * xOA * xHA)
            + (0.0150 * xOA * xOA)
            + (0.0134 * xOPTT * xAlpha)
            + (0.0296 * xOPTT * xHA)
            + (0.0752 * xOPTT * xOA)
            + (0.0192 * xOPTT * xOPTT)
        )
        fx[:, 2] = (
            0.370
            - (0.205 * xAlpha)
            + (0.0307 * xHA)
            + (0.108 * xOA)
            + (1.019 * xOPTT)
            - (0.135 * xAlpha * xAlpha)
            + (0.0141 * xHA * xAlpha)
            + (0.0998 * xHA * xHA)
            + (0.208 * xOA * xAlpha)
            - (0.0301 * xOA * xHA)
            - (0.226 * xOA * xOA)
            + (0.353 * xOPTT * xAlpha)
            - (0.0497 * xOPTT * xOA)
            - (0.423 * xOPTT * xOPTT)
            + (0.202 * xHA * xAlpha * xAlpha)
            - (0.281 * xOA * xAlpha * xAlpha)
            - (0.342 * xHA * xHA * xAlpha)
            - (0.245 * xHA * xHA * xOA)
            + (0.281 * xOA * xOA * xHA)
            - (0.184 * xOPTT * xOPTT * xAlpha)
            - (0.281 * xHA * xAlpha * xOA)
        )

        return gx, fx


class RE41(BenchmarkProblem):
    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 4
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=4,
            num_constraints=0,
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

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.78 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )
        fx[:, 1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3

        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        fx[:, 2] = 0.5 * (Vmbp + Vfd)

        g0 = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g1 = 0.32 - (
            0.261
            - 0.0159 * x1 * x2
            - 0.06486 * x1
            - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.0154464 * x6
        )
        g2 = 0.32 - (
            0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2 * x2
        )
        g3 = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g4 = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g5 = 32 - (
            33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728
        )
        g6 = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g7 = 4 - fx[:, 1]
        g8 = 9.9 - Vmbp
        g9 = 15.7 - Vfd

        g_list = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
        penalty = torch.zeros(n, device=X.device)
        for g in g_list:
            penalty = penalty + torch.where(g < 0, -g, torch.zeros_like(g))
        fx[:, 3] = penalty

        return gx, fx


class RE42(BenchmarkProblem):
    available_dimensions = 6
    input_type = DataType.CONTINUOUS
    num_objectives = 4
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=6,
            num_objectives=4,
            num_constraints=0,
            bounds=[
                (150.0, 274.32),
                (20.0, 32.31),
                (13.0, 25.0),
                (10.0, 11.71),
                (14.0, 18.0),
                (0.63, 0.75),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g_const = 9.8065
        Fn = V / torch.pow(g_const * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (torch.pow(displacement, 2.0 / 3.0) * torch.pow(x_Vk, 3.0)) / (
            a + (b * Fn)
        )
        outfit_weight = (
            1.0
            * torch.pow(x_L, 0.8)
            * torch.pow(x_B, 0.6)
            * torch.pow(x_D, 0.3)
            * torch.pow(x_CB, 0.1)
        )
        steel_weight = (
            0.034
            * torch.pow(x_L, 1.7)
            * torch.pow(x_B, 0.7)
            * torch.pow(x_D, 0.4)
            * torch.pow(x_CB, 0.5)
        )
        machinery_weight = 0.17 * torch.pow(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
            (2000.0 * torch.pow(steel_weight, 0.85))
            + (3500.0 * outfit_weight)
            + (2400.0 * torch.pow(power, 0.8))
        )
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * torch.pow(torch.clamp(DWT, min=1e-10), 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * torch.pow(torch.clamp(DWT, min=1e-10), 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * torch.pow(torch.clamp(DWT, min=1e-10), 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = annual_costs / annual_cargo
        fx[:, 1] = light_ship_weight
        fx[:, 2] = -annual_cargo

        cf0 = (x_L / x_B) - 6.0
        cf1 = -(x_L / x_D) + 15.0
        cf2 = -(x_L / x_T) + 19.0
        cf3 = 0.45 * torch.pow(torch.clamp(DWT, min=1e-10), 0.31) - x_T
        cf4 = 0.7 * x_D + 0.7 - x_T
        cf5 = 500000.0 - DWT
        cf6 = DWT - 3000.0
        cf7 = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        cf8 = (KB + BMT - KG) - (0.07 * x_B)

        cf_list = [cf0, cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8]
        penalty = torch.zeros(n, device=X.device)
        for cf in cf_list:
            penalty = penalty + torch.where(cf < 0, -cf, torch.zeros_like(cf))
        fx[:, 3] = penalty

        return gx, fx


class RE61(BenchmarkProblem):
    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 6
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=6,
            num_constraints=0,
            bounds=[
                (0.01, 0.45),
                (0.01, 0.10),
                (0.01, 0.10),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = 106780.37 * (x2 + x3) + 61704.67
        fx[:, 1] = 3000 * x1
        fx[:, 2] = 305700 * 2289 * x2 / (0.06 * 2289) ** 0.65
        fx[:, 3] = 250 * 2289 * torch.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        fx[:, 4] = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)

        g0 = 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
        g1 = 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
        g2 = 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
        g3 = 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
        g4 = 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
        g5 = 2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)
        g6 = 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)

        g_list = [g0, g1, g2, g3, g4, g5, g6]
        penalty = torch.zeros(n, device=X.device)
        for g in g_list:
            penalty = penalty + torch.where(g < 0, -g, torch.zeros_like(g))
        fx[:, 5] = penalty

        return gx, fx


class RE91(BenchmarkProblem):
    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 9
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=9,
            num_constraints=0,
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

        # Stochastic variables
        x8 = 0.006 * torch.randn(n, device=X.device) + 0.345
        x9 = 0.006 * torch.randn(n, device=X.device) + 0.192
        x10 = 10 * torch.randn(n, device=X.device)
        x11 = 10 * torch.randn(n, device=X.device)

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, 0), device=X.device)

        fx[:, 0] = (
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.75 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )

        fx[:, 1] = torch.clamp(
            (
                1.16
                - 0.3717 * x2 * x4
                - 0.00931 * x2 * x10
                - 0.484 * x3 * x9
                + 0.01343 * x6 * x10
            )
            / 1.0,
            min=0.0,
        )

        fx[:, 2] = torch.clamp(
            (
                0.261
                - 0.0159 * x1 * x2
                - 0.188 * x1 * x8
                - 0.019 * x2 * x7
                + 0.0144 * x3 * x5
                + 0.87570001 * x5 * x10
                + 0.08045 * x6 * x9
                + 0.00139 * x8 * x11
                + 0.00001575 * x10 * x11
            )
            / 0.32,
            min=0.0,
        )

        fx[:, 3] = torch.clamp(
            (
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
                + 0.00184 * x9 * x10
                - 0.018 * x2 * x2
            )
            / 0.32,
            min=0.0,
        )

        fx[:, 4] = torch.clamp(
            (
                0.74
                - 0.61 * x2
                - 0.163 * x3 * x8
                + 0.001232 * x3 * x10
                - 0.166 * x7 * x9
                + 0.227 * x2 * x2
            )
            / 0.32,
            min=0.0,
        )

        tmp = (
            (
                28.98
                + 3.818 * x3
                - 4.2 * x1 * x2
                + 0.0207 * x5 * x10
                + 6.63 * x6 * x9
                - 7.77 * x7 * x8
                + 0.32 * x9 * x10
            )
            + (
                33.86
                + 2.95 * x3
                + 0.1792 * x10
                - 5.057 * x1 * x2
                - 11 * x2 * x8
                - 0.0215 * x5 * x10
                - 9.98 * x7 * x8
                + 22 * x8 * x9
            )
            + (46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10)
        ) / 3
        fx[:, 5] = torch.clamp(tmp / 32, min=0.0)

        fx[:, 6] = torch.clamp(
            (
                4.72
                - 0.5 * x4
                - 0.19 * x2 * x3
                - 0.0122 * x4 * x10
                + 0.009325 * x6 * x10
                + 0.000191 * x11 * x11
            )
            / 4.0,
            min=0.0,
        )

        fx[:, 7] = torch.clamp(
            (
                10.58
                - 0.674 * x1 * x2
                - 1.95 * x2 * x8
                + 0.02054 * x3 * x10
                - 0.0198 * x4 * x10
                + 0.028 * x6 * x10
            )
            / 9.9,
            min=0.0,
        )

        fx[:, 8] = torch.clamp(
            (
                16.45
                - 0.489 * x3 * x7
                - 0.843 * x5 * x6
                + 0.0432 * x9 * x10
                - 0.0556 * x9 * x11
                - 0.000786 * x11 * x11
            )
            / 15.7,
            min=0.0,
        )

        return gx, fx
