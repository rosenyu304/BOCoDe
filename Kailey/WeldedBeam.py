import torch
from base import BenchmarkProblem

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

        n = X.size(0)

        gx = torch.zeros((n, self.num_cons))
        C1, C2, C3 = (1.10471, 0.04811, 14.0)

        h = X[:, 0]
        l = X[:, 1]
        t = X[:, 2]
        b = X[:, 3]

        test_function = -(C1 * h * h * l + C2 * t * b * (C3 + l))
        fx = test_function.reshape(n, self.num_obj)

        tao_dx = 6000 / (2**0.5 * h * l)

        tao_dxx = (
            6000
            * (14 + 0.5 * l)
            * (0.25 * (l**2 + (h + t) ** 2))**0.5
            / (2 * (0.707 * h * l * (l**2 / 12 + 0.25 * (h + t) ** 2)))
        )

        tao = (
            tao_dx**2
            + tao_dxx**2
            + l * tao_dx * tao_dxx / (0.25 * (l**2 + (h + t) ** 2))**0.5
        )**0.5

        sigma = 504000 / (t**2 * b)

        P_c = 64746 * (1 - 0.0282346 * t) * t * b**3

        delta = 2.1952 / (t**3 * b)

        gx[:, 0] = (-1) * (13600 - tao)
        gx[:, 1] = (-1) * (30000 - sigma)
        gx[:, 2] = (-1) * (b - h)
        gx[:, 3] = (-1) * (P_c - 6000)
        gx[:, 4] = (-1) * (0.25 - delta)

        return gx, fx
