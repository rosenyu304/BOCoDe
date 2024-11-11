import torch
import numpy as np
from base import BenchmarkProblem


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

    # N-D objective (can take data of different dimention; we use 18), 2 constraints, X = n-by-18

    tags = {"single_objective", "constrained", "continuous", "18D"}

    def __init__(self):
        super().__init__(dim = 18, num_obj = 1, num_cons = 2, bounds = [[0, 10] * 18])

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



problem_database = {CantileverBeam: CantileverBeam.tags,
                    Car: Car.tags,
                    CompressionSpring: CompressionSpring.tags,
                    HeatExchanger: HeatExchanger.tags,
                    KeaneBump: KeaneBump.tags,
                    PressureVessel: PressureVessel.tags,
                    ReinforcedConcreteBeam: ReinforcedConcreteBeam.tags,
                    SpeedReducer: SpeedReducer.tags,
                    ThreeTruss: ThreeTruss.tags,
                    WeldedBeam: WeldedBeam.tags}

def find_benchmark_problems(tags = None, extra_imports = False):
    """
    Returns a set of Benchmark Problems that each have all of the tags from
    at least one of tag's inner lists. (The inner list acts as an AND,
    and the list of lists acts as an OR.)

    Parameters:
        tags (2D list): a list of specified tags
            - tag options: "single_objective", "constrained",
              "continuous", "xD" (x = 2, 3, 4, 7, 8)

    Returns:
        return_probs (set): satisfactory Benchmark Problems
    """
    return_probs = set()

    for set_of_tags in tags:
        for prob in problem_database:
            if not (extra_imports == False and "extra imports" in problem_database[prob]):
                to_add = True
                for tag in set_of_tags:
                    if to_add and tag not in problem_database[prob]:
                        to_add = False
                if to_add:
                    return_probs.add(prob)

    return return_probs


# print(find_benchmark_problems([["single_objective", "3D"]]))
