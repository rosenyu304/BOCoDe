import torch
import math
from base import BenchmarkProblem

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

        n = X.size(0)

        gx = torch.zeros((n, self.num_cons))

        Ts = X[:, 0]
        Th = X[:, 1]
        R = X[:, 2]
        L = X[:, 3]

        test_function = -(C1 * Ts * R * L + C2 * Th * R * R + C3 * Ts * Ts * L + C4 * Ts * Ts * R)
        fx = test_function.reshape(n, self.num_obj)

        gx[:, 0] = -Ts + 0.0193 * R
        gx[:, 1] = -Th + 0.00954 * R
        gx[:, 2] = (-1) * math.pi * R * R * L + (-1) * 4 / 3 * math.pi * R * R * R + 750 * 1728
        gx[:, 3] = L - 240

        return gx, fx
