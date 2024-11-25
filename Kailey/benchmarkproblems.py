import torch
import numpy as np
from base import BenchmarkProblem
import importlib
import random


class Car(BenchmarkProblem):

    r'''
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
    algorithm. Computers & Structures 89(23-24):2325–2336
    '''

    # 11D objective, 10 constraints, X = n-by-11

    tags = {"single_objective", "constrained", "11D"}

    def __init__(self):
        super().__init__(dim = 11, num_obj = 1, num_cons = 10, bounds = [[0.5, 1.5], [0.45, 1.35], [0.5, 1.5], [0.5, 1.5],
                                                                         [0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.192, 0.345],
                                                                         [0.192, 0.345], [0, -20], [0, -20]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        n = X.size(0)

        fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
            torch.zeros((n, 1)),
        )

        for i in range(n):

            x = X[i, :]

            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x

            test_function = -(
                1.98 + 4.90 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 2.73 * x7
            )

            gx1[i] = (
                1.16
                - 0.3717 * x2 * x4
                - 0.00931 * x2 * x10
                - 0.484 * x3 * x9
                + 0.01343 * x6 * x10
                - 1
            )

            gx2[i] = (
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

            gx3[i] = (
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

            gx4[i] = (
                0.74
                - 0.061 * x2
                - 0.163 * x3 * x8
                + 0.001232 * x3 * x10
                - 0.166 * x7 * x9
                + 0.227 * x2 * x2
                - 0.9
            )

            gx5[i] = (
                28.98
                + 3.818 * x3
                - 4.2 * x1 * x2
                + 0.0207 * x5 * x10
                + 6.63 * x6 * x9
                - 7.7 * x7 * x8
                + 0.32 * x9 * x10
                - 32
            )

            gx6[i] = (
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

            gx7[i] = 46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10 - 32

            gx8[i] = (
                4.72
                - 0.5 * x4
                - 0.19 * x2 * x3
                - 0.0122 * x4 * x10
                + 0.009325 * x6 * x10
                + 0.000191 * x11**2
                - 4
            )

            gx9[i] = (
                10.58
                - 0.674 * x1 * x2
                - 1.95 * x2 * x8
                + 0.02054 * x3 * x10
                - 0.0198 * x4 * x10
                + 0.028 * x6 * x10
                - 9.9
            )

            gx10[i] = (
                16.45
                - 0.489 * x3 * x7
                - 0.843 * x5 * x6
                + 0.0432 * x9 * x10
                - 0.0556 * x9 * x11
                - 0.000786 * x11**2
                - 15.7
            )

            fx[i] = test_function

        gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10), 1)

        return gx, fx



class CantileverBeam(BenchmarkProblem):

    r'''
    Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for
    global engineering optimization. Engineering computations 29(5):464–483
    '''

    # 10D objective, 11 constraints, X = n-by-10

    tags = {"single_objective", "constrained", "10D"}

    def __init__(self):
        super().__init__(dim = 10, num_obj = 1, num_cons = 11, bounds = [[1, 5], [1, 5], [1, 5], [1, 5], [1, 5],
                                                                         [30, 65], [30, 65], [30, 65], [30, 65], [30, 65]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
            torch.tensor([]),
        ) * 12

        n = X.size(0)

        for i in range(n):

            x = X[i, :]

            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x

            P = 50000
            E = 2 * 107
            L = 100

            test_function = -(
                x1 * x6 * L + x2 * x7 * L + x3 * x8 * L + x4 * x9 * L + x5 * x10 * L
            )
            fx = torch.cat((fx, torch.tensor([[test_function]])))

            gx1 = torch.cat((gx1, torch.tensor([[600 * P / (x5 * x10 * x10) - 14000]])))
            gx2 = torch.cat((gx2, torch.tensor([[6 * P * (L * 2) / (x4 * x9 * x9) - 14000]])))
            gx3 = torch.cat((gx3, torch.tensor([[6 * P * (L * 3) / (x3 * x8 * x8) - 14000]])))
            gx4 = torch.cat((gx4, torch.tensor([[6 * P * (L * 4) / (x2 * x7 * x7) - 14000]])))
            gx5 = torch.cat((gx5, torch.tensor([[6 * P * (L * 5) / (x1 * x6 * x6) - 14000]])))
            gx6 = torch.cat((gx6, torch.tensor([[P * L**3 * (1 / L + 7 / L + 19 / L + 37 / L + 61 / L) / (3 * E) - 2.7]])))
            gx7 = torch.cat((gx7, torch.tensor([[x10 / x5 - 20]])))
            gx8 = torch.cat((gx8, torch.tensor([[x9 / x4 - 20]])))
            gx9 = torch.cat((gx9, torch.tensor([[x8 / x3 - 20]])))
            gx10 = torch.cat((gx10, torch.tensor([[x7 / x2 - 20]])))
            gx11 = torch.cat((gx11, torch.tensor([[x6 / x1 - 20]])))

        gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11), 1)

        return gx, fx



class ThreeTruss(BenchmarkProblem):

    r'''
    Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for global engineering optimization.
    Engineering computations 29(5):464–483
    '''

    # 2D objective, 3 constraints, X = n-by-2

    tags = {"single_objective", "constrained", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 3, bounds = [[2, 5], [2, 4]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx, gx1, gx2, gx3 = (torch.tensor([]),) * 4

        n = X.size(0)

        for i in range(n):

            x = X[i, :]

            x1, x2 = x

            if x1 <= 1e-5:
                x1 = 1e-5
            if x2 <= 1e-5:
                x2 = 1e-5

            L = 100
            P = 2
            sigma = 2

            test_function = -(2 * np.sqrt(2) * x1 + x2) * L
            fx = torch.cat((fx, torch.tensor([[test_function]])))

            gx1 = torch.cat((gx1, torch.tensor([[(np.sqrt(2) * x1 + x2) / (np.sqrt(2) * x1 * x1 + 2 * x1 * x2) * P - sigma]])))
            gx2 = torch.cat((gx2, torch.tensor([[(x2) / (np.sqrt(2) * x1 * x1 + 2 * x1 * x2) * P - sigma]])))
            gx3 = torch.cat((gx3, torch.tensor([[(1) / (x1 + np.sqrt(2) * x2) * P - sigma]])))

        gx = torch.cat((gx1, gx2, gx3), 1)

        return gx, fx



class CompressionSpring(BenchmarkProblem):

    r'''
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly algorithm.
    Computers & Structures 89(23-24):2325–2336
    '''

    # 3D objective, 6 constraints, X = n-by-3

    tags = {"single_objective", "constrained", "3D"}

    def __init__(self):
        super().__init__(dim = 3, num_obj = 1, num_cons = 6, bounds = [[0.05, 1], [0.25, 1.3], [2, 15]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx, gx1, gx2, gx3, gx4 = (torch.tensor([]),) * 5

        n = X.size(0)

        for i in range(n):

            x = X[i, :]

            d, D, N = x

            test_function = -((N + 2) * D * d**2)
            fx = torch.cat((fx, torch.tensor([[test_function]])))

            gx1 = torch.cat((gx1, torch.tensor([[1 - (D * D * D * N / (71785 * d * d * d * d))]])))
            gx2 = torch.cat((gx2, torch.tensor([[(4 * D * D - D * d) / (12566 * (D * d * d * d - d * d * d * d)) + 1 / (5108 * d * d) - 1]])))
            gx3 = torch.cat((gx3, torch.tensor([[1 - 140.45 * d / (D * D * N)]])))
            gx4 = torch.cat((gx4, torch.tensor([[(D + d) / 1.5 - 1]])))

        gx = torch.cat((gx1, gx2, gx3, gx4), 1)

        return gx, fx



class ReinforcedConcreteBeam(BenchmarkProblem):

    r'''
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
    algorithm. Computers & Structures 89(23-24):2325–2336
    '''

    # 3D objective, 9 constraints, X = n-by-3

    tags = {"single_objective", "constrained", "continuous", "3D"}

    def __init__(self):
        super().__init__(dim = 3, num_obj = 1, num_cons = 9, bounds = [[0.2, 15], [28, 40], [5, 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx, gx1, gx2, gx3, gx4 = (torch.tensor([]),) * 5

        n = X.size(0)

        for i in range(n):

            x = X[i, :]

            As, h, b = x

            test_function = -(29.4 * As + 0.6 * b * h)
            fx = torch.cat((fx, torch.tensor([[test_function]])))

            gx1 = torch.cat((gx1, torch.tensor([[h / b - 4]])))
            gx2 = torch.cat((gx2, torch.tensor([[180 + 7.35 * As * As / b - As * h]])))

        gx = torch.cat((gx1, gx2), 1)

        return gx, fx



class PressureVessel(BenchmarkProblem):

    r'''
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
    algorithm. Computers & Structures 89(23-24):2325–2336
    '''

    # 4D objective, 4 constraints, X = n-by-4

    tags = {"single_objective", "constrained", "continuous", "4D"}

    def __init__(self):
        super().__init__(dim = 4, num_obj = 1, num_cons = 4, bounds = [[0.0625, 98 * 0.0625 + 0.0625],
                                                                       [0.0625, 98 * 0.0625 + 0.0625],
                                                                       [10, 200], [0, 200 - 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        C1, C2, C3, C4 = (0.6224, 1.7781, 3.1661, 19.84)
        fx, gx1, gx2, gx3, gx4 = (torch.tensor([]),) * 5

        n = X.size(0)

        for i in range(n):

            x = X[i, :]

            Ts, Th, R, L = x

            test_function = -(
                C1 * Ts * R * L + C2 * Th * R * R + C3 * Ts * Ts * L + C4 * Ts * Ts * R
            )
            fx = torch.cat((fx, torch.tensor([[test_function]])))

            gx1 = torch.cat((gx1, torch.tensor([[-Ts + 0.0193 * R]])))
            gx2 = torch.cat((gx2, torch.tensor([[-Th + 0.00954 * R]])))
            gx3 = torch.cat((gx3, torch.tensor([[(-1) * np.pi * R * R * L + (-1) * 4 / 3 * np.pi * R * R * R + 750 * 1728]])))
            gx4 = torch.cat((gx4, torch.tensor([[L - 240]])))

        gx = torch.cat((gx1, gx2, gx3, gx4), 1)

        return gx, fx



class HeatExchanger(BenchmarkProblem):

    r'''
    Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for global
    engineering optimization. Engineering computations 29(5):464–483
    '''

    # 8D objective, 6 constraints, X = n-by-8

    tags = {"single_objective", "constrained", "continuous", "8D"}

    def __init__(self):
        super().__init__(dim = 8, num_obj = 1, num_cons = 6, bounds = [[100, 10000], [1000, 10000],
                                                                       [1000, 10000], [10, 1000],
                                                                       [10, 1000], [10, 1000],
                                                                       [10, 1000], [10, 1000]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
            torch.tensor([]),
        ) * 12

        n = X.size(0)

        for i in range(n):

            x = X[i, :]

            x1, x2, x3, x4, x5, x6, x7, x8 = x

            test_function = -(x1 + x2 + x3)

            fx = torch.cat((fx, torch.tensor([[test_function]])))

            gx1 = torch.cat((gx1, torch.tensor([[0.0025 * (x4 + x6) - 1]])))
            gx2 = torch.cat((gx2, torch.tensor([[0.0025 * (x5 + x7 - x4) - 1]])))
            gx3 = torch.cat((gx3, torch.tensor([[0.01 * (x8 - x5) - 1]])))
            gx4 = torch.cat((gx4, torch.tensor([[833.33252 * x4 + 100 * x1 - x1 * x6 - 83333.333]])))
            gx5 = torch.cat((gx5, torch.tensor([[1250 * x5 + x2 * x4 - x2 * x7 - 125 * x4]])))
            gx6 = torch.cat((gx6, torch.tensor([[x3 * x5 - 2500 * x5 - x3 * x8 + 125 * 10000]])))

        gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6), 1)

        return gx, fx



class SpeedReducer(BenchmarkProblem):

    r'''
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
    algorithm. Computers & Structures 89(23-24):2325–2336
    '''

    # 7D objective, 9 constraints, X = n-by-7

    tags = {"single_objective", "constrained", "continuous", "7D"}

    def __init__(self):
        super().__init__(dim = 7, num_obj = 1, num_cons = 9, bounds = [[2.6, 3.6], [0.7, 0.8], [17, 28],
                                                                       [7.3, 8.3], [7.3, 8.3], [2.9, 3.9], [5, 5.5]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
                torch.tensor([]),
            ) * 12

        n = X.size(0)

        for i in range(n):

            x = X[i, :]

            b, m, z, L1, L2, d1, d2 = x

            C1 = 0.7854 * b * m * m
            C2 = 3.3333 * z * z + 14.9334 * z - 43.0934
            C3 = 1.508 * b * (d1 * d1 + d2 * d2)
            C4 = 7.4777 * (d1 * d1 * d1 + d2 * d2 * d2)
            C5 = 0.7854 * (L1 * d1 * d1 + L2 * d2 * d2)

            test_function = -(C1 * (C2) - C3 + C4 + C5)

            fx = torch.cat((fx, torch.tensor([[test_function]])))

            gx1 = torch.cat((gx1, torch.tensor([[27 / (b * m * m * z) - 1]])))
            gx2 = torch.cat((gx2, torch.tensor([[397.5 / (b * m * m * z * z) - 1]])))
            gx3 = torch.cat((gx3, torch.tensor([[1.93 * L1**3 / (m * z * d1**4) - 1]])))
            gx4 = torch.cat((gx4, torch.tensor([[1.93 * L2**3 / (m * z * d2**4) - 1]])))
            gx5 = torch.cat((gx5, torch.tensor([[np.sqrt((745 * L1 / (m * z)) ** 2 + 1.69 * 1e6)/ (110 * d1**3) - 1]])))
            gx6 = torch.cat((gx6,torch.tensor([[np.sqrt((745 * L2 / (m * z)) ** 2 + 157.5 * 1e6) / (85 * d2**3) - 1]])))
            gx7 = torch.cat((gx7, torch.tensor([[m * z / 40 - 1]])))
            gx8 = torch.cat((gx8, torch.tensor([[5 * m / (b) - 1]])))
            gx9 = torch.cat((gx9, torch.tensor([[b / (12 * m) - 1]])))

        gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9), 1)

        return gx, fx



class WeldedBeam(BenchmarkProblem):

    r'''
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
    algorithm. Computers & Structures 89(23-24):2325–2336
    '''

    # 4D objective, 5 constraints, X = n-by-4

    tags = {"single_objective", "constrained", "continuous", "4D"}

    def __init__(self):
        super().__init__(dim = 4, num_obj = 1, num_cons = 5, bounds = [[0.125, 10], [0.1, 15], [0.1, 10], [0.1, 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        n = X.shape[0]

        C1, C2, C3 = (1.10471, 0.04811, 14.0)
        fx, gx1, gx2, gx3, gx4, gx5 = (
            torch.zeros(n, 1),
            torch.zeros(n, 1),
            torch.zeros(n, 1),
            torch.zeros(n, 1),
            torch.zeros(n, 1),
            torch.zeros(n, 1),
        )

        for i in range(n):

            x = X[i, :]

            h, l, t, b = x

            test_function = -(C1 * h * h * l + C2 * t * b * (C3 + l))
            fx[i] = test_function

            tao_dx = 6000 / (np.sqrt(2) * h * l)

            tao_dxx = (
                6000
                * (14 + 0.5 * l)
                * np.sqrt(0.25 * (l**2 + (h + t) ** 2))
                / (2 * (0.707 * h * l * (l**2 / 12 + 0.25 * (h + t) ** 2)))
            )

            tao = np.sqrt(
                tao_dx**2
                + tao_dxx**2
                + l * tao_dx * tao_dxx / np.sqrt(0.25 * (l**2 + (h + t) ** 2))
            )

            sigma = 504000 / (t**2 * b)

            P_c = 64746 * (1 - 0.0282346 * t) * t * b**3

            delta = 2.1952 / (t**3 * b)

            gx1[i] = (-1) * (13600 - tao)
            gx2[i] = (-1) * (30000 - sigma)
            gx3[i] = (-1) * (b - h)
            gx4[i] = (-1) * (P_c - 6000)
            gx5[i] = (-1) * (0.25 - delta)

        gx = torch.cat((gx1, gx2, gx3, gx4, gx5), 1)

        return gx, fx



class KeaneBump(BenchmarkProblem):

    r'''
    Keane A (1994) Experiences with optimizers instructural design. In: Proceedings of the conference
    on adaptive computing in engineering design and control, pp 14–27
    '''

    # N-D objective, 2 constraints, X = n-by-dim

    tags = {"single_objective", "constrained", "continuous", "ND"}

    def __init__(self, dim=18):
        super().__init__(dim, num_obj = 1, num_cons = 2, bounds = [[0, 10] * dim])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx = torch.zeros(X.shape[0], 1).to(torch.float64)
        gx1 = torch.zeros(X.shape[0], 1).to(torch.float64)
        gx2 = torch.zeros(X.shape[0], 1).to(torch.float64)

        for i in range(X.shape[0]):
            x = X[i,:]

            cos4 = 0
            cos2 = 1
            sq_denom = 0

            pi_sum = 1
            sigma_sum = 0

            for j in range(X.shape[1]):
                cos4 += torch.cos(x[j]) ** 4
                cos2 *= torch.cos(x[j]) ** 2
                sq_denom += (j+1) * (x[j])**2

                pi_sum *= x[j]
                sigma_sum += x[j]

            test_function = torch.abs((cos4 - 2*cos2) / torch.sqrt(sq_denom))
            fx[i] = test_function

            gx1[i] = 0.75 - pi_sum
            gx2[i] = sigma_sum - 7.5* (X.shape[1])

        gx = torch.cat((gx1, gx2), 1)
        return gx, fx



class Ackley(BenchmarkProblem):

    r'''
    Eriksson D, Poloczek M (2021) Scalable constrained bayesian optimization.
    In: International Conference on Artificial Intelligence and Statistics, PMLR, pp 730–738
    '''

    # N-D objective, 2 constraints, X = n-by-dim

    tags = {"single_objective", "constrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 2, optimizers = [[0] * dim], optimum = [[0]], bounds = [[-5, 10]])

    def evaluate(self, X, to_verify = True):
        from botorch.test_functions import Ackley as Ackley_imported
        device = torch.device("cpu")
        dtype = torch.double

        X = super().scale(X, to_verify)

        fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
        fun.bounds[0, :].fill_(-5)
        fun.bounds[1, :].fill_(10)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        gx1 = torch.sum(X,1)  # sigma(x) <= 0
        gx1 = gx1.reshape((n, 1))

        gx2 = torch.norm(X, p=2, dim=1)-5  # norm_2(x) -3 <= 0
        gx2 = gx2.reshape((n, 1))

        gx = torch.cat((gx1, gx2), 1)

        return gx, fx



class Bukin(BenchmarkProblem):

    r'''

    '''

    # 2D objective, 0 constraints, X = n-by-2

    tags = {"single_objective", "unconstrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 0, bounds = [[-15.0, -5.0], [-3.0, 3.0]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        part1 = 100.0 * torch.sqrt(torch.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * torch.abs(X[..., 0] + 10.0)
        fx = part1 + part2

        return None, fx



class Goldstein(BenchmarkProblem):

    r'''
    LVGP paper: https://www.nature.com/articles/s41598-020-60652-9
    '''

    # 2D objective, 0 constraints, X = n-by-2

    tags = {"single_objective", "unconstrained", "continuous", "mixed", "2D"}

    def __init__(self):
        super().__init__(dim = 4, num_obj = 1, num_cons = 0, optimizers = [[0, -1]], optimum = [-3], bounds = [[-2, 2], [0, 1]])

    def evaluate(self, X, to_verify = True):

        def cont_to_disc(x, disc_values):
            # Convert continuous value to discrete value
            # Input:
            #   x: continuous value in [0, 1]
            #   disc_values: discrete values
            # Output: discrete value
            idx = torch.floor(x * len(disc_values)).long()
            return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]

        # x0: [-2, 2]
        # x1: {-2, -1, 0, 1, 2}
        X = super().scale(X, to_verify)
        X[:,1] = cont_to_disc(X[:,1], torch.tensor([-2, -1, 0, 1, 2]))

        fx = ((1 + (X[:,0] + X[:,1] +1)**2
            * (19 - 14*X[:,0] + 3*X[:,0]**2 -14*X[:,1]
                +6*X[:,0]*X[:,1] + 3*X[:,1]**2
                )
            ) *
            (
                30 + (2*X[:,0] - 3*X[:,1])**2
                * (18- 32*X[:,0] + 12*X[:,0]**2 + 48*X[:,1]
                    -36*X[:,0]*X[:,1] + 27*X[:,1]**2
                )
            ))

        return None, -fx.reshape(-1, 1)



class Rosenbrock(BenchmarkProblem):

    r'''

    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 0, bounds = [[-5, 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import Rosenbrock as Rosenbrock_imported

        fun = Rosenbrock_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx



class Griewank(BenchmarkProblem):

    r'''

    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 0, bounds = [[-600, 600]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import Griewank as Griewank_imported

        fun = Griewank_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx



class Levy(BenchmarkProblem):

    r'''

    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 0, bounds = [[-10, 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import Levy as Levy_imported

        fun = Levy_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx



class DixonPrice(BenchmarkProblem):

    r'''

    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 0, bounds = [[-10, 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import DixonPrice as DixonPrice_imported

        fun = DixonPrice_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx



class GearTrain(BenchmarkProblem):

    r'''
    Sandgren, E. (1990). Nonlinear Integer and Discrete Programming in Mechanical Design Optimization."
    ASME. J. Mech. Des. June 1990; 112(2): 223–229.
    '''

    # 4D objective, 0 constraints, X = n-by-4

    tags = {"single_objective", "unconstrained", "mixed", "4D"}

    def __init__(self):
        super().__init__(dim = 4, num_obj = 1, num_cons = 0, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        def cont_to_disc(x, disc_values):
            # Convert continuous value to discrete value
            # Input:
            #   x: continuous value in [0, 1]
            #   disc_values: discrete values
            # Output: discrete value
            idx = torch.floor(x * len(disc_values)).long()
            return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]

        X = cont_to_disc(X, torch.tensor(range(12, 61))) # x0, x1, x2, x3: {12, 13, ..., 60}

        fx = (1/6.931 - (X[:,0]*X[:,1])/(X[:,2]*X[:,3])).reshape(-1, 1)

        return None, fx



class EulerBernoulliBeamBending(BenchmarkProblem):

    r'''
    Cuesta Ramirez, J., Le Riche, R., Roustant, O. et al.
    (2022) A comparison of mixed-variables Bayesian optimization
    approaches. Adv. Model. and Simul. in Eng. Sci. 9, 6 .
    '''

    # 3D objective, 0 constraints, X = n-by-3

    tags = {"single_objective", "unconstrained", "mixed", "3D"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 0, optimizers = [[0.0, 0.43, 0.380]], optimum = [-1.287*10^-3], bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        def cont_to_disc(x, disc_values):
            # Convert continuous value to discrete value
            # Input:
            #   x: continuous value in [0, 1]
            #   disc_values: discrete values
            # Output: discrete value
            idx = torch.floor(x * len(disc_values)).long()
            return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]

        # x0: [0, 1]
        # x1: [0, 1]
        # x2: {0.083, 0.139, 0.380, 0.080, 0.133, 0.363, 0.086, 0.136, 0.360, 0.092, 0.138, 0.369}
        X[:,2] = cont_to_disc(X[:,2], torch.tensor([0.083, 0.139, 0.380, 0.080, 0.133, 0.363, 0.086, 0.136, 0.360, 0.092, 0.138, 0.369]))

        # BO comparison paper: https://amses-journal.springeropen.com/articles/10.1186/s40323-022-00218-8
        E = 600
        P = 600
        alpha = 60

        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

        L = 10 + 10 * x1
        S = 1 + x2
        I = x3

        D = P * L ** 3 / (3 * E * S**2 * I)
        y = D + alpha * L * S
        return None, -y.reshape(-1, 1)



class JLH1(BenchmarkProblem):

    r'''
    Jetton C, Li C, Hoyle C (2023) Constrained bayesian optimization methods using regression
    and classification gaussian processes as constraints. In: International Design Engineering
    Technical Conferences and Computers and Information in Engineering Conference, American
    Society of Mechanical Engineers, pV03BT03A033
    '''

    # 2D objective, 1 constraint, X = n-by-2

    tags = {"single_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 1, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx = []
        gx = []

        for x in X:
            test_function = (- (x[0]-0.5)**2 - (x[1]-0.5)**2 )
            fx.append(test_function)
            gx.append( x[0] + x[1] - 0.75 )

        fx = torch.reshape(torch.tensor(fx), (len(fx),1))
        gx = torch.reshape(torch.tensor(gx), (len(gx),1))

        return gx, fx



class JLH2(BenchmarkProblem):

    r'''
    Jetton C, Li C, Hoyle C (2023) Constrained bayesian optimization methods using regression
    and classification gaussian processes as constraints. In: International Design Engineering
    Technical Conferences and Computers and Information in Engineering Conference, American
    Society of Mechanical Engineers, pV03BT03A033
    '''

    # 2D objective, 1 constraint, X = n-by-2

    tags = {"single_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 1, bounds = [[-5, 0], [-5, 5]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx = []
        gx = []

        for x in X:

            ## Negative sign to make it a maximization problem
            test_function = - ( np.cos(2*x[0])*np.cos(x[1]) +  np.sin(x[0]) )

            fx.append(test_function)
            gx.append( ((x[0]+5)**2)/4 + (x[1]**2)/100 -2.5 )

        fx = torch.reshape(torch.tensor(fx), (len(fx),1))
        gx = torch.reshape(torch.tensor(gx), (len(gx),1))

        return gx, fx



class GKXWC1(BenchmarkProblem):

    r'''
    Gardner JR, Kusner MJ, Xu ZE, et al (2014) Bayesian optimization with inequality constraints.
    In: ICML, pp 937–945
    '''

    # 2D objective, 1 constraint, X = n-by-2

    tags = {"single_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 1, bounds = [[0, 6]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx = []
        gx = []
        for x in X:
            g = np.cos(x[0])*np.cos(x[1]) -  np.sin(x[0])*np.sin(x[1]) -0.5
            fx.append( - np.cos(2*x[0])*np.cos(x[1]) -  np.sin(x[0])  )
            gx.append( g )

        fx = torch.reshape(torch.tensor(fx), (len(fx),1))
        gx = torch.reshape(torch.tensor(gx), (len(gx),1))
        return gx, fx



class GKXWC2(BenchmarkProblem):

    r'''
    Gardner JR, Kusner MJ, Xu ZE, et al (2014) Bayesian optimization with inequality constraints.
    In: ICML, pp 937–945
    '''

    # 2D objective, 1 constraint, X = n-by-2

    tags = {"single_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 1, bounds = [[0, 6]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx = []
        gx = []

        for x in X:
            g = np.sin(x[0])*np.sin(x[1]) + 0.95
            fx.append( - np.sin(x[0]) - x[1]  ) # maximize -(x1^2 +x 2^2)
            gx.append( g )

        fx = torch.reshape(torch.tensor(fx), (len(fx),1))
        gx = torch.reshape(torch.tensor(gx), (len(gx),1))

        return gx, fx



class Michalewicz(BenchmarkProblem):

    r'''

    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        import math
        super().__init__(dim, num_obj = 1, num_cons = 0, bounds = [[0, math.pi]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import Michalewicz as Michalewicz_imported

        fun = Michalewicz_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx



class StyblinskiTang_Continuous(BenchmarkProblem):

    r'''

    '''

    # 10D objective, 0 constraints, X = n-by-10

    tags = {"single_objective", "unconstrained", "continuous", "10D"}

    def __init__(self):
        dim_ = 10
        super().__init__(dim = dim_, num_obj = 1, num_cons = 0, optimizers = [[-2.903534] * dim_], optimum = [[-39.16599] * dim_], bounds = [[-5, 5]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import StyblinskiTang as StyblinskiTang_imported

        return None, -StyblinskiTang_imported(X).view(-1, 1)



class StyblinskiTang_Mixed(BenchmarkProblem):

    r'''

    '''

    # 10D objective, 0 constraints, X = n-by-10

    tags = {"single_objective", "unconstrained", "mixed", "10D"}

    def __init__(self):
        super().__init__(dim = 10, num_obj = 1, num_cons = 0, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        def cont_to_disc(x, disc_values):
            # Convert continuous value to discrete value
            # Input:
            #   x: continuous value in [0, 1]
            #   disc_values: discrete values
            # Output: discrete value
            idx = torch.floor(x * len(disc_values)).long()
            return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]

        # X: {-5, -2.5, 0, 2.5, 5}^dim
        X = cont_to_disc(X, torch.tensor([-5, -2.5, 0, 2.5, 5]))

        from botorch.test_functions.synthetic import StyblinskiTang as StyblinskiTang_imported

        return None, -StyblinskiTang_imported(X).view(-1, 1)



class Truss10D(BenchmarkProblem):

    r'''
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    '''

    # 10D objective, 10 constraints, X = n-by-10

    tags = {"single_objective", "constrained", "10D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 10, num_obj = 1, num_cons = 10, bounds = [[0.1, 35]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type  import SupportType, MemberType

        def Truss10bar(A, E, Rho):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION     = 2
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            init_truss = truss

            # Truss settings:
            joints = [(720, 360), (720, 0), (360, 360), (360, 0), (0, 360), (0, 0)]
            supports = [SupportType.NO, SupportType.NO, SupportType.NO, SupportType.NO, SupportType.PIN, SupportType.PIN]
            forces = [(1, (0, -1e5)), (3, (0, -1e5))]
            members = [(2, 4), (0, 2), (3, 5), (1, 3), (2, 3), (0, 1), (3, 4), (2, 5), (1, 2), (0, 3)]


            # memberType : Member type which contain the information about
                            # 1) cross-sectional area,
                            # 2) Young's modulus,
                            # 3) density of this member.


            # Read data in this [.py]:
            for joint, support in zip(joints, supports):
                truss.AddNewJoint(joint, support)

            for jointID, force in forces:
                truss.AddExternalForce(jointID, force)

            index = 0
            for jointID0, jointID1 in members:
                # Default: 0.1, 1e7, 1

                memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                if (E != None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
                elif (E != None) & (Rho==None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.1)
                elif (E == None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), 10000000.0, Rho[index].item())

                truss.AddNewMember(jointID0, jointID1, memberType)
                index += 1

            # Do direct stiffness method:
            truss.Solve()

            # Dump all the structural analysis results into a .json file:
            # truss.DumpIntoJSON(TEST_OUTPUT_FILE)

            # Get result of structural analysis:
            displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
            return displace, forces, stress, resistance, truss, truss.weight

        E = 1e7 * torch.ones(10)
        Rho = 0.1 * torch.ones(10)

        n = X.size(0)

        fx = torch.zeros(n,1)

        # 10 bar stress constraints, 4 displacement constraints
        gx = torch.zeros(n,14)

        for ii in range(n):

            displace, _, stress, _, _, weights = Truss10bar(X[ii,:], E, Rho)

            fx[ii,0] = -weights           # Negate for maximizing optimization

            for ss in range(10):
                gx[ii,ss] = abs(stress[ss]) - 25000

            gx[ii,10] = max(abs(displace[0][0]), abs(displace[0][1])) - 2
            gx[ii,11] = max(abs(displace[1][0]), abs(displace[1][1])) - 2
            gx[ii,12] = max(abs(displace[2][0]), abs(displace[2][1])) - 2
            gx[ii,13] = max(abs(displace[3][0]), abs(displace[3][1])) - 2

        return gx, fx



class Truss25D(BenchmarkProblem):

    r'''
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    '''

    # 25D objective, 31 constraints, X = n-by-25

    tags = {"single_objective", "constrained", "25D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 25, num_obj = 1, num_cons = 31, bounds = [[0.1, 3.4]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type  import SupportType, MemberType

        def Truss25bar(A, E, Rho):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION     = 3
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            init_truss = truss

            # Truss settings:
            joints     = [(62.5, 100, 200), (137.5, 100, 200), (62.5, 137.5, 100), (137.5, 137.5, 100),
                        (137.5, 62.5, 100), (62.5, 62.5, 100), (0, 200, 0), (200, 200, 0), (200, 0, 0),
                        (0, 0, 0)]
            supports   = [SupportType.NO, SupportType.NO, SupportType.NO, SupportType.NO,
                          SupportType.NO, SupportType.NO, SupportType.PIN, SupportType.PIN,
                          SupportType.PIN, SupportType.PIN]
            forces     = [(0, (1000, -10000, -10000)), (1, (0, -10000, -10000)), (2, (500, 0, 0)), (5, (600, 0, 0))]
            members    = [(0, 1), (0, 3), (1, 2), (0, 4), (1, 5), (0, 2), (0, 5), (1, 3), (1, 4), (2, 5), (3, 4),
                          (2, 3), (4, 5), (2, 9), (5, 6), (3, 8), (4, 7), (2, 7), (3, 6), (5, 8), (4, 9), (2, 6),
                          (3, 7), (4, 8), (5, 9)]


            # memberType : Member type which contain the information about
                            # 1) cross-sectional area,
                            # 2) Young's modulus,
                            # 3) density of this member.


            # Read data in this [.py]:
            for joint, support in zip(joints, supports):
                truss.AddNewJoint(joint, support)

            for jointID, force in forces:
                truss.AddExternalForce(jointID, force)

            index = 0
            for jointID0, jointID1 in members:

                # Default: 0.1, 1e7, .1

                memberType = MemberType(A[index].item(), 3e7, .283)

                if (E != None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
                elif (E != None) & (Rho==None):
                    memberType = MemberType(A[index].item(), E[index].item(), .283)
                elif (E == None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), 3e7, Rho[index].item())



                # memberType = MemberType(A[index].item(), 1e7, .1)
                truss.AddNewMember(jointID0, jointID1, memberType)
                index += 1

            # Do direct stiffness method:
            truss.Solve()

            # Dump all the structural analysis results into a .json file:
            # truss.DumpIntoJSON(TEST_OUTPUT_FILE)

            # Get result of structural analysis:
            displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
            return displace, forces, stress, resistance, truss, truss.weight

        if X.size(1) == 25:
            A = X
        elif X.size(1) == 8:
            # Bars in 8 groups because of symmetry
            # (1) A1, (2) A2–A5, (3) A6–A9, (4) A10–A11, (5) A12–A13, (6) A14–A17, (7) A18–A21 and (8) A22–A25.
            A = torch.zeros(X.size(0), 25)
            A[:,0] = X[:,0]
            A[:,1:5] = X[:,1]
            A[:,5:9] = X[:,2]
            A[:,9:11] = X[:,3]
            A[:,11:13] = X[:,4]
            A[:,13:17] = X[:,5]
            A[:,17:21] = X[:,6]
            A[:,21:25] = X[:,7]

        E = 1e7 * torch.ones(25)
        Rho = 0.1 * torch.ones(25)


        n = X.size(0)

        fx = torch.zeros(n,1)

        # 25 bar stress constraints, 6 displacement constraints
        gx = torch.zeros(n,31)

        for ii in range(n):

            displace, _, stress, _, _, weights = Truss25bar(A[ii,:], E, Rho)

            fx[ii,0] = -weights           # Negate for maximizing optimization

            # Max stress less than 40ksi
            for ss in range(25):
                gx[ii,ss] = abs(stress[ss]) - 40000

            # Max displacement in x and y direction less than .35 inches
            for dd in range(6):
                # print(displace[dd])
                gx[ii,25+dd] = max(abs(displace[dd][0]), abs(displace[dd][1])) - 0.35


        return gx, fx



class Truss72D(BenchmarkProblem):

    r'''
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    '''

    # 72D objective, 88 constraints, X = n-by-72

    tags = {"single_objective", "constrained", "72D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 72, num_obj = 1, num_cons = 88, bounds = [[0.1, 33.5]])

    def evaluate(self, X, version = "4_forces", to_verify = True):
        X = super().scale(X, to_verify)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type  import SupportType, MemberType

        def Truss72bar(A, E, Rho, version="4_forces"):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION     = 3
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            init_truss = truss

            # Truss settings:
            joints = [
                    (0.0, 0.0, 0.0),
                    (120.0, 0.0, 0.0),
                    (120.0, 120.0, 0.0),
                    (0.0, 120.0, 0.0),
                    (0.0, 0.0, 60.0),
                    (120.0, 0.0, 60.0),
                    (120.0, 120.0, 60.0),
                    (0.0, 120.0, 60.0),
                    (0.0, 0.0, 120.0),
                    (120.0, 0.0, 120.0),
                    (120.0, 120.0, 120.0),
                    (0.0, 120.0, 120.0),
                    (0.0, 0.0, 180.0),
                    (120.0, 0.0, 180.0),
                    (120.0, 120.0, 180.0),
                    (0.0, 120.0, 180.0),
                    (0.0, 0.0, 240.0),
                    (120.0, 0.0, 240.0),
                    (120.0, 120.0, 240.0),
                    (0.0, 120.0, 240.0)
                ]





            supports = supports = [
                    SupportType.PIN,
                    SupportType.PIN,
                    SupportType.PIN,
                    SupportType.PIN,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO,
                    SupportType.NO
                ]




            if version == "4_forces":
                forces = [
                        (0, (0.0, 0.0, 0.0)),
                        (1, (0.0, 0.0, 0.0)),
                        (2, (0.0, 0.0, 0.0)),
                        (3, (0.0, 0.0, 0.0)),
                        (4, (0.0, 0.0, 0.0)),
                        (5, (0.0, 0.0, 0.0)),
                        (6, (0.0, 0.0, 0.0)),
                        (7, (0.0, 0.0, 0.0)),
                        (8, (0.0, 0.0, 0.0)),
                        (9, (0.0, 0.0, 0.0)),
                        (10, (0.0, 0.0, 0.0)),
                        (11, (0.0, 0.0, 0.0)),
                        (12, (0.0, 0.0, 0.0)),
                        (13, (0.0, 0.0, 0.0)),
                        (14, (0.0, 0.0, 0.0)),
                        (15, (0.0, 0.0, 0.0)),
                        (16, (0.0, 0.0, -5000.0)),
                        (17, (0.0, 0.0, -5000.0)),
                        (18, (0.0, 0.0, -5000.0)),
                        (19, (0.0, 0.0, -5000.0))
                    ]
            elif version == "single":
                forces = [
                    (0, (0.0, 0.0, 0.0)),
                    (1, (0.0, 0.0, 0.0)),
                    (2, (0.0, 0.0, 0.0)),
                    (3, (0.0, 0.0, 0.0)),
                    (4, (0.0, 0.0, 0.0)),
                    (5, (0.0, 0.0, 0.0)),
                    (6, (0.0, 0.0, 0.0)),
                    (7, (0.0, 0.0, 0.0)),
                    (8, (0.0, 0.0, 0.0)),
                    (9, (0.0, 0.0, 0.0)),
                    (10, (0.0, 0.0, 0.0)),
                    (11, (0.0, 0.0, 0.0)),
                    (12, (0.0, 0.0, 0.0)),
                    (13, (0.0, 0.0, 0.0)),
                    (14, (0.0, 0.0, 0.0)),
                    (15, (0.0, 0.0, 0.0)),
                    (16, (5000.0, 5000.0, -5000.0)),
                    (17, (0.0, 0.0, 0.0)),
                    (18, (0.0, 0.0, 0.0)),
                    (19, (0.0, 0.0, 0.0))
                ]
            else:
                raise ValueError("Invalid version, choose between '4_forces' and 'single'")




            members = [
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (1, 4),
                    (0, 5),
                    (1, 6),
                    (2, 5),
                    (2, 7),
                    (3, 6),
                    (0, 7),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (4, 6),
                    (5, 7),
                    (4, 8),
                    (5, 9),
                    (6, 10),
                    (7, 11),
                    (5, 8),
                    (4, 9),
                    (5, 10),
                    (6, 9),
                    (6, 11),
                    (7, 10),
                    (4, 11),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 8),
                    (8, 10),
                    (9, 11),
                    (8, 12),
                    (9, 13),
                    (10, 14),
                    (11, 15),
                    (9, 12),
                    (8, 13),
                    (9, 14),
                    (10, 13),
                    (10, 15),
                    (11, 14),
                    (8, 15),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 12),
                    (12, 14),
                    (13, 15),
                    (12, 16),
                    (13, 17),
                    (14, 18),
                    (15, 19),
                    (13, 16),
                    (12, 17),
                    (13, 18),
                    (14, 17),
                    (14, 19),
                    (15, 18),
                    (12, 19),
                    (15, 16),
                    (16, 17),
                    (17, 18),
                    (18, 19),
                    (19, 16),
                    (16, 18),
                    (17, 19)
                ]




            # memberType : Member type which contain the information about
                            # 1) cross-sectional area,
                            # 2) Young's modulus,
                            # 3) density of this member.


            # Read data in this [.py]:
            for joint, support in zip(joints, supports):
                truss.AddNewJoint(joint, support)

            for jointID, force in forces:
                truss.AddExternalForce(jointID, force)

            index = 0
            for jointID0, jointID1 in members:
                # memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                if (E != None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
                elif (E != None) & (Rho==None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.1)
                elif (E == None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), 10000000.0, Rho[index].item())


                truss.AddNewMember(jointID0, jointID1, memberType)
                index += 1

            # Do direct stiffness method:
            truss.Solve()

            # Dump all the structural analysis results into a .json file:
            # truss.DumpIntoJSON(TEST_OUTPUT_FILE)

            # Get result of structural analysis:
            displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
            return displace, forces, stress, resistance, truss, truss.weight

        if X.size(1) == 72:
            A = X
        elif X.size(1) == 16:
            # Bars in 16 groups because of symmetry
            # (1) A1–A4, (2) A5–A12, (3) A13–A16, (4) A17–A18, (5) A19–A22, (6) A23–A30, (7) A31–A34, (8) A35–A36,
            # (9) A37–A40, (10) A41–A48, (11) A49–A52, (12) A53–A54, (13) A55–A58, (14) A59–A66 (15), A67–A70, and (16) A71–A72.
            A = torch.zeros(X.size(0), 72)
            A[:,0:4] = X[:,0]
            A[:,4:12] = X[:,1]
            A[:,12:16] = X[:,2]
            A[:,16:18] = X[:,3]
            A[:,18:22] = X[:,4]
            A[:,22:30] = X[:,5]
            A[:,30:34] = X[:,6]
            A[:,34:36] = X[:,7]
            A[:,36:40] = X[:,8]
            A[:,40:48] = X[:,9]
            A[:,48:52] = X[:,10]
            A[:,52:54] = X[:,11]
            A[:,54:58] = X[:,12]
            A[:,58:66] = X[:,13]
            A[:,66:70] = X[:,14]
            A[:,70:72] = X[:,15]

        E = 1e7 * torch.ones(72)
        Rho = 0.1 * torch.ones(72)

        n = A.size(0)

        fx = torch.zeros(n,1)

        # 72 bar stress constraints, 16 displacement constraints
        gx = torch.zeros(n, 88)

        for ii in range(n):

            displace, _, stress, _, _, weights = Truss72bar(A[ii,:], E, Rho, version)

            fx[ii,0] = -weights           # Negate for maximizing optimization

            # Max stress less than 25000
            for ss in range(72):
                gx[ii,ss] = abs(stress[ss]) - 25000

            # Max displacement in x and y direction less than .25 inches
            for dd in range(4, 20): # 16 free nodes
                gx[ii,72+dd-4] = max(abs(displace[dd][0]), abs(displace[dd][1])) - 0.25


            return gx, fx



class Truss120D(BenchmarkProblem):

    r'''
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    '''

    # 120D objective, 121 constraints, X = n-by-120

    tags = {"single_objective", "constrained", "120D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 120, num_obj = 1, num_cons = 121, bounds = [[0.775, 20]])

    def evaluate(self, X, version = "4_forces", to_verify = True):
        X = super().scale(X, to_verify)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type  import SupportType, MemberType

        def Truss120bar(A, E, Rho):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION     = 3
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            init_truss = truss

            # Truss settings:
            joints = [
                (0.0, 0.0, 275.59),
                (273.26, 0.0, 230.31),
                (236.65010183813573, 136.62999999999997, 230.31),
                (136.62999999999997, 236.65010183813573, 230.31),
                (0.0, 273.26, 230.31),
                (-136.62999999999997, 236.65010183813573, 230.31),
                (-236.65010183813573, 136.62999999999997, 230.31),
                (-273.26, 0.0, 230.31),
                (-236.65010183813573, -136.62999999999997, 230.31),
                (-136.62999999999997, -236.65010183813573, 230.31),
                (0.0, -273.26, 230.31),
                (136.62999999999997, -236.65010183813573, 230.31),
                (236.65010183813573, -136.62999999999997, 230.31),
                (492.12, 0.0, 118.11),
                (475.3514176333763, 127.37002847585251, 118.11),
                (426.18842171039796, 246.05999999999997, 118.11),
                (347.9813891575237, 347.9813891575237, 118.11),
                (246.05999999999997, 426.18842171039796, 118.11),
                (127.37002847585251, 475.3514176333763, 118.11),
                (0.0, 492.12, 118.11),
                (-127.37002847585251, 475.3514176333763, 118.11),
                (-246.05999999999997, 426.18842171039796, 118.11),
                (-347.9813891575237, 347.9813891575237, 118.11),
                (-426.18842171039796, 246.05999999999997, 118.11),
                (-475.3514176333763, 127.37002847585251, 118.11),
                (-492.12, 0.0, 118.11),
                (-475.3514176333763, -127.37002847585251, 118.11),
                (-426.18842171039796, -246.05999999999997, 118.11),
                (-347.9813891575237, -347.9813891575237, 118.11),
                (-246.05999999999997, -426.18842171039796, 118.11),
                (-127.37002847585251, -475.3514176333763, 118.11),
                (0.0, -492.12, 118.11),
                (127.37002847585251, -475.3514176333763, 118.11),
                (246.05999999999997, -426.18842171039796, 118.11),
                (347.9813891575237, -347.9813891575237, 118.11),
                (426.18842171039796, -246.05999999999997, 118.11),
                (475.3514176333763, -127.37002847585251, 118.11),
                (625.59, 0.0, 0.0),
                (541.7768323535071, 312.79499999999996, 0.0),
                (312.79499999999996, 541.7768323535071, 0.0),
                (0.0, 625.59, 0.0),
                (-312.79499999999996, 541.7768323535071, 0.0),
                (-541.7768323535071, 312.79499999999996, 0.0),
                (-625.59, 0.0, 0.0),
                (-541.7768323535071, -312.79499999999996, 0.0),
                (-312.79499999999996, -541.7768323535071, 0.0),
                (0.0, -625.59, 0.0),
                (312.79499999999996, -541.7768323535071, 0.0),
                (541.7768323535071, -312.79499999999996, 0.0)
            ]

            supports = [
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN
            ]

            # print(len(joints))
            # print(len(supports))



            forces = [
                (0, (0.0, 0.0, -13490.0)),
                (1, (0.0, 0.0, -6744.0)),
                (2, (0.0, 0.0, -6744.0)),
                (3, (0.0, 0.0, -6744.0)),
                (4, (0.0, 0.0, -6744.0)),
                (5, (0.0, 0.0, -6744.0)),
                (6, (0.0, 0.0, -6744.0)),
                (7, (0.0, 0.0, -6744.0)),
                (8, (0.0, 0.0, -6744.0)),
                (9, (0.0, 0.0, -6744.0)),
                (10, (0.0, 0.0, -6744.0)),
                (11, (0.0, 0.0, -6744.0)),
                (12, (0.0, 0.0, -6744.0)),
                (13, (0.0, 0.0, -6744.0)),
                (14, (0.0, 0.0, -2248.0)),
                (15, (0.0, 0.0, -2248.0)),
                (16, (0.0, 0.0, -2248.0)),
                (17, (0.0, 0.0, -2248.0)),
                (18, (0.0, 0.0, -2248.0)),
                (19, (0.0, 0.0, -2248.0)),
                (20, (0.0, 0.0, -2248.0)),
                (21, (0.0, 0.0, -2248.0)),
                (22, (0.0, 0.0, -2248.0)),
                (23, (0.0, 0.0, -2248.0)),
                (24, (0.0, 0.0, -2248.0)),
                (25, (0.0, 0.0, -2248.0)),
                (26, (0.0, 0.0, -2248.0)),
                (27, (0.0, 0.0, -2248.0)),
                (28, (0.0, 0.0, -2248.0)),
                (29, (0.0, 0.0, -2248.0)),
                (30, (0.0, 0.0, -2248.0)),
                (31, (0.0, 0.0, -2248.0)),
                (32, (0.0, 0.0, -2248.0)),
                (33, (0.0, 0.0, -2248.0)),
                (34, (0.0, 0.0, -2248.0)),
                (35, (0.0, 0.0, -2248.0)),
                (36, (0.0, 0.0, -2248.0))
            ]



            members = [
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (0, 4),
                    (0, 5),
                    (0, 6),
                    (0, 7),
                    (0, 8),
                    (0, 9),
                    (0, 10),
                    (0, 11),
                    (0, 12),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (12, 1),
                    (1, 13),
                    (2, 15),
                    (3, 17),
                    (4, 19),
                    (5, 21),
                    (6, 23),
                    (7, 25),
                    (8, 27),
                    (9, 29),
                    (10, 31),
                    (11, 33),
                    (12, 35),
                    (1, 14),
                    (2, 14),
                    (2, 16),
                    (3, 16),
                    (3, 18),
                    (4, 18),
                    (4, 20),
                    (5, 20),
                    (5, 22),
                    (6, 22),
                    (6, 24),
                    (7, 24),
                    (7, 26),
                    (8, 26),
                    (8, 28),
                    (9, 28),
                    (9, 30),
                    (10, 30),
                    (10, 32),
                    (11, 32),
                    (11, 34),
                    (12, 34),
                    (12, 36),
                    (1, 36),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (16, 17),
                    (17, 18),
                    (18, 19),
                    (19, 20),
                    (20, 21),
                    (21, 22),
                    (22, 23),
                    (23, 24),
                    (24, 25),
                    (25, 26),
                    (26, 27),
                    (27, 28),
                    (28, 29),
                    (29, 30),
                    (30, 31),
                    (31, 32),
                    (32, 33),
                    (33, 34),
                    (34, 35),
                    (35, 36),
                    (36, 13),
                    (13, 37),
                    (15, 38),
                    (17, 39),
                    (19, 40),
                    (21, 41),
                    (23, 42),
                    (25, 43),
                    (27, 44),
                    (29, 45),
                    (31, 46),
                    (33, 47),
                    (35, 48),
                    (14, 37),
                    (14, 38),
                    (16, 38),
                    (16, 39),
                    (18, 39),
                    (18, 40),
                    (20, 40),
                    (20, 41),
                    (22, 41),
                    (22, 42),
                    (24, 42),
                    (24, 43),
                    (26, 43),
                    (26, 44),
                    (28, 44),
                    (28, 45),
                    (30, 45),
                    (30, 46),
                    (32, 46),
                    (32, 47),
                    (34, 47),
                    (34, 48),
                    (36, 48),
                    (36, 37)
                ]



            # memberType : Member type which contain the information about
                            # 1) cross-sectional area,
                            # 2) Young's modulus,
                            # 3) density of this member.


            # Read data in this [.py]:
            for joint, support in zip(joints, supports):
                truss.AddNewJoint(joint, support)

            for jointID, force in forces:
                truss.AddExternalForce(jointID, force)

            index = 0
            for jointID0, jointID1 in members:
                # memberType = MemberType(A[index].item(), 30450000, 0.288)
                # print(A.shape)
                memberType = MemberType(A[index].item(), 30450000, 0.288)

                if (E != None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
                elif (E != None) & (Rho==None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.288)
                elif (E == None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), 30450000, Rho[index].item())



                truss.AddNewMember(jointID0, jointID1, memberType)
                index += 1

            # Do direct stiffness method:
            truss.Solve()

            # Dump all the structural analysis results into a .json file:
            # truss.DumpIntoJSON(TEST_OUTPUT_FILE)

            # Get result of structural analysis:
            displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
            return displace, forces, stress, resistance, truss, truss.weight

        E = None
        Rho = None

        n = X.size(0)

        fx = torch.zeros(n,1)

        # 120 bar stress constraints, 1 displacement constraints
        gx = torch.zeros(n,121)

        for ii in range(n):
            # print(ii)
            # print(A[ii,:].shape)
            displace, _, stress, _, _, weights = Truss120bar(X[ii,:], None, None)

            if (E != None) & (Rho!=None):
                displace, _, stress, _, _, weights = Truss120bar(X[ii,:], E[ii,:], Rho[ii,:])
            elif (E != None) & (Rho==None):
                displace, _, stress, _, _, weights = Truss120bar(X[ii,:], E[ii,:], None)
            elif (E == None) & (Rho!=None):
                displace, _, stress, _, _, weights = Truss120bar(X[ii,:], None, Rho[ii,:])

            fx[ii,0] = -weights           # Negate for maximizing optimization

            # Max stress less than 34800
            for ss in range(120):
                gx[ii,ss] = abs(stress[ss]) - 34800

            # Max displacement in x and y direction less than
            MAX_DIST = 0
            for dd in range(len(displace)):
                if max(displace[dd]) > MAX_DIST:
                    MAX_DIST = max(abs(displace[dd]))
            gx[ii,120] = MAX_DIST - 0.1969



        return gx, fx



class Truss200D(BenchmarkProblem):

    r'''
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    '''

    # 200D objective, 200 constraints, X = n-by-200

    tags = {"single_objective", "constrained", "200D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 200, num_obj = 1, num_cons = 200, bounds = [[0.1, 33.7]])

    def evaluate(self, X, version = 1, to_verify = True):
        X = super().scale(X, to_verify)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type  import SupportType, MemberType

        def Truss200bar(A, E, Rho, version=1):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION     = 2
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            init_truss = truss

            # Truss settings (77 joints, 200 members):
            l1 = 240
            l2 = 144
            l3 = 360
            joints = []
            for row in range(11):
                if row % 2 == 0:
                    joints.extend([[col*l1, row*l2] for col in range(5)])
                else:
                    joints.extend([[col*l1/2, row*l2] for col in range(9)])
            joints.append([l1, 10*l2+l3])
            joints.append([3*l1, 10*l2+l3])


            supports = [SupportType.NO for _ in range(75)] + [SupportType.PIN, SupportType.PIN]


            nodes1 = [0, 5, 14, 19, 28, 33, 42, 47, 56, 61, 70]
            nodes2 = [
                0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25,
                27, 28, 29, 30, 31, 32, 33, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47, 49,
                51, 53, 55, 56, 57, 58, 59, 60, 61, 63, 65, 67, 69, 70, 71, 72, 73, 74
            ]
            if version == 1:
                forces = [[i, (1e3, 0)] for i in nodes1]
            elif version == 2:
                forces = [[i, (0, -1e4)] for i in nodes2]
            elif version == 3:
                forces = [[i, (1e3, 0)] for i in nodes1] + [[i, (0, -1e4)] for i in nodes2]


            members = []
            j_idx = 0
            row_members = np.array([
                [0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [0, 6], [1, 6], [1, 7], [1, 8], [2, 8], [2, 9], [2, 10],
                [3, 10], [3, 11], [3, 12], [4, 12], [4, 13],
                [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13],
                [5, 14], [6, 14], [6, 15], [7, 15], [8, 15], [8, 16], [9, 16], [10, 16],
                [10, 17], [11, 17], [12, 17], [12, 18], [13, 18],
            ])
            for row in range(5):
                # 38 each row
                members.extend((row_members + j_idx).tolist())
                j_idx += 14
            members.extend([[70+i, 71+i] for i in range(4)])
            members.extend([
                [70, 75], [71, 75], [72, 75], [72, 76], [73, 76], [74, 76]
            ])



            for joint, support in zip(joints, supports):
                truss.AddNewJoint(joint, support)

            for jointID, force in forces:
                truss.AddExternalForce(jointID, force)

            index = 0
            for jointID0, jointID1 in members:
                # memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                if (E != None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
                elif (E != None) & (Rho==None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.1)
                elif (E == None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), 10000000.0, Rho[index].item())


                truss.AddNewMember(jointID0, jointID1, memberType)
                index += 1

            # Do direct stiffness method:
            truss.Solve()

            # TrussPlotter(truss).Plot()

            # Dump all the structural analysis results into a .json file:
            # truss.DumpIntoJSON(TEST_OUTPUT_FILE)

            # Get result of structural analysis:
            displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
            return displace, forces, stress, resistance, truss, truss.weight


        if X.size(1) == 200:
            A = X
        elif X.size(1) == 29:
            # Bars in 29 groups because of symmetry
            # (1) A1-A4, (2) A5/8/11/14/17, (3) A19/20/21/22/23/24, (4) A18/25/56/63/94/101/132/139/170/177,
            # (5) A26/29/32/35/38, (6) A6/7/9/10/12/13/15/16/27/28/30/31/33/34/36/37, (7) A39/40/41/42, (8) A43/46/49/52/55,
            # (9) A57/58/59/60/61/62, (10) A64/67/70/73/76, (11) A44/45/47/48/50/51/53/54/65/66/68/69/71/72/74/75,
            # (12) A77/78/79/80, (13) A81/84/87/90/93, (14) A95/96/97/98/99/100, (15) A102/105/108/111/114,
            # (16) A82/83/85/86/88/89/91/92/103/104/106/107/109/110/112/113, (17) A115/116/117/118, (18) A119/122/125/128/131,
            # (19) A133/134/135/136/137/138, (20) A140/143/146/149/152, (21) A120/121/123/124/126/127/129/130/141/142/144/145/147/148/150/151,
            # (22) A153/154/155/156, (23) A157/160/163/166/169, (24) A171/172/173/174/175/176, (25) A178/181/184/187/190,
            # (26) A158/159/161/162/164/165/167/168/179/180/182/183/185/186/188/189, (27) A191/192/193/194, (28) A195/197/198/200, (29) A196/199

            A = torch.zeros(X.size(0), 200)
            A[:,0:4] = X[:,0]
            A[:, [4, 7, 10, 13, 16]] = X[:,1]
            A[:, [18, 19, 20, 21, 22, 23]] = A_[:,2]
            A[:, [17, 24, 55, 62, 93, 100, 131, 138, 169, 176]] = X[:,3]
            A[:, [25, 28, 31, 34, 37]] = X[:,4]
            A[:, [5, 6, 8, 9, 11, 12, 14, 15, 26, 27, 29, 30, 32, 33, 35, 36]] = X[:,5]
            A[:, [38, 39, 40, 41]] = X[:,6]
            A[:, [42, 45, 48, 51, 54]] = X[:,7]
            A[:, [56, 57, 58, 59, 60, 61]] = X[:,8]
            A[:, [63, 66, 69, 72, 75]] = X[:,9]
            A[:, [43, 44, 46, 47, 49, 50, 52, 53, 64, 65, 67, 68, 70, 71, 73, 74]] = X[:,10]
            A[:, [76, 77, 78, 79]] = X[:,11]
            A[:, [80, 83, 86, 89, 92]] = X[:,12]
            A[:, [94, 95, 96, 97, 98, 99]] = X[:,13]
            A[:, [101, 104, 107, 110, 113]] = X[:,14]
            A[:, [81, 82, 84, 85, 87, 88, 90, 91, 102, 103, 105, 106, 108, 109, 111, 112]] = X[:,15]
            A[:, [114, 115, 116, 117]] = X[:,16]
            A[:, [118, 121, 124, 127, 130]] = X[:,17]
            A[:, [132, 133, 134, 135, 136, 137]] = X[:,18]
            A[:, [139, 142, 145, 148, 151]] = X[:,19]
            A[:, [119, 120, 122, 123, 125, 126, 128, 129, 140, 141, 143, 144, 146, 147, 149, 150]] = X[:,20]
            A[:, [152, 153, 154, 155]] = X[:,21]
            A[:, [156, 159, 162, 165, 168]] = X[:,22]
            A[:, [170, 171, 172, 173, 174, 175]] = X[:,23]
            A[:, [177, 180, 183, 186, 189]] = X[:,24]
            A[:, [157, 158, 160, 161, 163, 164, 166, 167, 178, 179, 181, 182, 184, 185, 187, 188]] = X[:,25]
            A[:, [190, 191, 192, 193]] = X[:,26]
            A[:, [194, 196, 197, 199]] = X[:,27]
            A[:, [195, 198]] = X[:,28]


        E = 3e4 * torch.ones(200)
        Rho = 0.283 * torch.ones(200)

        n = A.size(0)

        fx = torch.zeros(n,1)

        # 200 bar stress constraints
        gx = torch.zeros(n, 200)

        for ii in range(n):

            displace, _, stress, _, _, weights = Truss200bar(A[ii,:], E, Rho, version)

            fx[ii,0] = -weights           # Negate for maximizing optimization

            # Max stress less than 10000
            for ss in range(200):
                if ss in stress:
                    gx[ii,ss] = abs(stress[ss]) - 10000
                else:
                    gx[ii,ss] = -10000


        return gx, fx



class MOPTA08Car(BenchmarkProblem):

    r'''

    '''

    # 124D objective, 68 constraints, X = n-by-124

    tags = {"single_objective", "constrained", "continuous", "124D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 124, num_obj = 1, num_cons = 68, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from platform import machine

        import numpy as np
        import torch
        import stat

        def MOPTA08_Car_single(x):
            # Get the current permissions of the file
            current_permissions = os.stat(os.getcwd()).st_mode

            # Add execute permissions for the owner, group, and others
            new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

            # Apply the new permissions
            os.chmod(os.getcwd(), new_permissions)

            sysarch = 64 if sys.maxsize > 2 ** 32 else 32

            machine = "x86_64"
            mopta_exectutable = "mopta08_elf64.bin"

            mopta_full_path = os.path.join(
                "mopta08", mopta_exectutable
            )

            directory_file_descriptor = tempfile.TemporaryDirectory()
            directory_name = Path(__file__).parent

            ##########################################################################################
            # Input here
            # if x == None:
            #     x = np.random.rand(124)
            #     print(x.shape)
            ##########################################################################################
            with open(os.path.join(directory_name, "input.txt"), "w+") as tmp_file:
                for _x in x:
                    tmp_file.write(f"{_x}\n")
            popen = subprocess.Popen(
                mopta_full_path,
                stdout=subprocess.PIPE,
                cwd=directory_name,
                shell=True,
            )
            popen.wait()

            with open(os.path.join(directory_name, "output.txt"), "r") as  tmp_file:
                output = (
                    tmp_file
                    .read()
                    .split("\n")
                )
            output = [x.strip() for x in output]
            output = np.array([float(x) for x in output if len(x) > 0])
            value = output[0]
            constraints = output[1:]

            return constraints, value


        GX =  torch.zeros(X.shape[0], 68)
        FX =  torch.zeros(X.shape[0], 1)
        for ii in range(X.shape[0]):
            input_x = X[ii,:].numpy()
            gx, fx = MOPTA08_Car_single(input_x)
            GX[ii,:] = torch.from_numpy(gx)
            FX[ii,:] = -fx

        return GX, FX



class MOPTA08Car_softpen(BenchmarkProblem):

    r'''

    '''

    # 124D objective, 68 constraints, X = n-by-124

    tags = {"single_objective", "constrained", "continuous", "124D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 124, num_obj = 1, num_cons = 68, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from platform import machine

        import numpy as np
        import torch
        import stat

        def MOPTA08_Car_single(x):
            # Get the current permissions of the file
            current_permissions = os.stat(os.getcwd()).st_mode

            # Add execute permissions for the owner, group, and others
            new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

            # Apply the new permissions
            os.chmod(os.getcwd(), new_permissions)

            sysarch = 64 if sys.maxsize > 2 ** 32 else 32

            machine = "x86_64"
            mopta_exectutable = "mopta08_elf64.bin"

            mopta_full_path = os.path.join(
                "mopta08", mopta_exectutable
            )

            directory_file_descriptor = tempfile.TemporaryDirectory()
            directory_name = Path(__file__).parent

            ##########################################################################################
            # Input here
            # if x == None:
            #     x = np.random.rand(124)
            #     print(x.shape)
            ##########################################################################################
            with open(os.path.join(directory_name, "input.txt"), "w+") as tmp_file:
                for _x in x:
                    tmp_file.write(f"{_x}\n")
            popen = subprocess.Popen(
                mopta_full_path,
                stdout=subprocess.PIPE,
                cwd=directory_name,
                shell=True,
            )
            popen.wait()

            with open(os.path.join(directory_name, "output.txt"), "r") as  tmp_file:
                output = (
                    tmp_file
                    .read()
                    .split("\n")
                )
            output = [x.strip() for x in output]
            output = np.array([float(x) for x in output if len(x) > 0])
            value = output[0]
            constraints = output[1:]

            return constraints, value


        GX =  torch.zeros(X.shape[0], 68)
        FX =  torch.zeros(X.shape[0], 1)
        for ii in range(X.shape[0]):
            input_x = X[ii,:].numpy()
            gx, fx = MOPTA08_Car_single(input_x)
            GX[ii,:] = torch.from_numpy(gx)
            FX[ii,:] = fx

        cost = GX
        cost[cost<0] = 0
        cost = cost.sum(dim=1).reshape(cost.shape[0], 1)
        FX = FX + cost

        return GX, -FX



class Mazda(BenchmarkProblem):

    r'''

    '''

    # 222D objective, 54 constraints, X = n-by-222

    tags = {"single_objective", "multi_objective", "constrained", "continuous", "222D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 222, num_obj = 1, num_cons = 68, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        import os
        import subprocess
        import stat
        import pandas as pd

        ##########################################
        # Scaling
        ##########################################

        # Define the path to your Excel file
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/Info_Mazda_CdMOBP_edited.xlsx'

        # Read the Excel file into a DataFrame
        dataframe = pd.read_excel(file_path, sheet_name='Explain_DV_and_Const.')

        # Display the DataFrame to ensure it has been read correctly
        bounds = dataframe.values[1:, 1:3]
        bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
        # print(bounds_tensor.shape)

        range_bounds = bounds_tensor[:,1] - bounds_tensor[:,0]

        scaled_samples = X * range_bounds + bounds_tensor[:,0]
        # print(scaled_samples)

        # Convert the torch tensor to a numpy array
        data_numpy_back = scaled_samples.numpy()

        # Create a pandas DataFrame from the numpy array
        dataframe_back = pd.DataFrame(data_numpy_back)

        # Write the DataFrame to a text file with space-separated values
        output_file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_vars_eval.txt'

        dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)
        #####################
        #####################


        #####################
        # Run Bash file
        #####################

        # Change the current working directory
        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2')

        # Get the current permissions of the file
        current_permissions = os.stat(os.getcwd()).st_mode

        # Add execute permissions for the owner, group, and others
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

        # Apply the new permissions
        os.chmod(os.getcwd(), new_permissions)

        # Script name
        script_name = 'run.sh'

        # Run the bash script in the background
        process = subprocess.Popen(['bash', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        process.wait()

        # Optional: capture the output and error messages
        stdout, stderr = process.communicate()

        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/')
        # print(os.getcwd())
        #####################
        #####################


        #####################
        # Read in objective and constraints
        #####################

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_objs_eval.txt'
        objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        objs_data_numpy = objs_dataframe.values

        # Convert the numpy array to a torch tensor
        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_cons_eval.txt'
        cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        cons_data_numpy = cons_dataframe.values

        # Convert the numpy array to a torch tensor
        cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

        return cons_data_tensor, -objs_data_tensor


class Mazda_softpen(BenchmarkProblem):

    r'''

    '''

    # 222D objective, 54 constraints, X = n-by-222

    tags = {"single_objective", "multi_objective", "constrained", "continuous", "222D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 222, num_obj = 1, num_cons = 68, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        import os
        import subprocess
        import stat
        import pandas as pd

        ##########################################
        # Scaling
        ##########################################

        # Define the path to your Excel file
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/Info_Mazda_CdMOBP_edited.xlsx'

        # Read the Excel file into a DataFrame
        dataframe = pd.read_excel(file_path, sheet_name='Explain_DV_and_Const.')

        # Display the DataFrame to ensure it has been read correctly
        bounds = dataframe.values[1:, 1:3]
        bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
        # print(bounds_tensor.shape)

        range_bounds = bounds_tensor[:,1] - bounds_tensor[:,0]

        bounds_tensor = bounds_tensor.to("cpu")
        range_bounds = range_bounds.to("cpu")
        init_samples = init_samples.to("cpu")

        scaled_samples = init_samples * range_bounds + bounds_tensor[:,0]
        # print(scaled_samples)

        # Convert the torch tensor to a numpy array
        data_numpy_back = scaled_samples.numpy()

        # Create a pandas DataFrame from the numpy array
        dataframe_back = pd.DataFrame(data_numpy_back)

        # Write the DataFrame to a text file with space-separated values
        output_file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_vars_eval.txt'

        dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)
        #####################
        #####################


        #####################
        # Run Bash file
        #####################

        # Change the current working directory
        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2')

        # Get the current permissions of the file
        current_permissions = os.stat(os.getcwd()).st_mode

        # Add execute permissions for the owner, group, and others
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

        # Apply the new permissions
        os.chmod(os.getcwd(), new_permissions)

        # Script name
        script_name = 'run.sh'

        # Run the bash script in the background
        process = subprocess.Popen(['bash', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        process.wait()

        # Optional: capture the output and error messages
        stdout, stderr = process.communicate()

        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/')
        # print(os.getcwd())
        #####################
        #####################


        #####################
        # Read in objective and constraints
        #####################

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_objs_eval.txt'
        objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        objs_data_numpy = objs_dataframe.values

        # Convert the numpy array to a torch tensor
        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_cons_eval.txt'
        cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        cons_data_numpy = cons_dataframe.values

        # Convert the numpy array to a torch tensor
        cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

        cost = cons_data_tensor
        cost[cost<0] = 0
        cost = cost.sum(dim=1).reshape(cost.shape[0], 1)
        objs_data_tensor = objs_data_tensor + cost

        return cons_data_tensor, -objs_data_tensor


problem_database = {Ackley: Ackley.tags,
                    Bukin: Bukin.tags,
                    CantileverBeam: CantileverBeam.tags,
                    Car: Car.tags,
                    CompressionSpring: CompressionSpring.tags,
                    DixonPrice: DixonPrice.tags,
                    EulerBernoulliBeamBending: EulerBernoulliBeamBending.tags,
                    GearTrain: GearTrain.tags,
                    GKXWC1: GKXWC1.tags,
                    GKXWC2: GKXWC2.tags,
                    Goldstein: Goldstein.tags,
                    Griewank: Griewank.tags,
                    HeatExchanger: HeatExchanger.tags,
                    JLH1: JLH1.tags,
                    JLH2: JLH2.tags,
                    KeaneBump: KeaneBump.tags,
                    Levy: Levy.tags,
                    Mazda: Mazda.tags,
                    Mazda_softpen: Mazda_softpen.tags,
                    Michalewicz: Michalewicz.tags,
                    MOPTA08Car: MOPTA08Car.tags,
                    MOPTA08Car_softpen: MOPTA08Car_softpen.tags,
                    PressureVessel: PressureVessel.tags,
                    ReinforcedConcreteBeam: ReinforcedConcreteBeam.tags,
                    Rosenbrock: Rosenbrock.tags,
                    SpeedReducer: SpeedReducer.tags,
                    StyblinskiTang_Continuous: StyblinskiTang_Continuous.tags,
                    StyblinskiTang_Mixed: StyblinskiTang_Mixed.tags,
                    ThreeTruss: ThreeTruss.tags,
                    Truss10D: Truss10D.tags,
                    Truss25D: Truss25D.tags,
                    Truss72D: Truss72D.tags,
                    Truss120D: Truss120D.tags,
                    Truss200D: Truss200D.tags,
                    WeldedBeam: WeldedBeam.tags}

def find_benchmark_problems(tags = None, extra_imports = False):
    """
    Returns a set of Benchmark Problems that each have all of the tags from
    at least one of tag's inner lists. (The inner list acts as an AND,
    and the list of lists acts as an OR.)

    Parameters:
        tags (2D list): a list of specified tags
            - tag options: "single_objective", "constrained", "unconstrained",
              "continuous", "mixed", "xD" (x = positive int or N for any)

    Returns:
        return_probs (set): satisfactory Benchmark Problems
    """
    return_probs = set()

    for set_of_tags in tags:
        for prob in problem_database:
            if not (extra_imports == False and "extra_imports" in problem_database[prob]):
                to_add = True
                for tag in set_of_tags:
                    if to_add and tag not in problem_database[prob]:
                        to_add = False
                if to_add:
                    return_probs.add(prob)

    return return_probs
