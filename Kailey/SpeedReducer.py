import torch
from base import BenchmarkProblem

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

        n = X.size(0)

        gx = torch.zeros((n, self.num_cons))

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

        test_function = -(C1 * (C2) - C3 + C4 + C5)

        fx = test_function.reshape(n, self.num_obj)

        gx[:, 0] = 27 / (b * m * m * z) - 1
        gx[:, 1] = 397.5 / (b * m * m * z * z) - 1
        gx[:, 2] = 1.93 * L1**3 / (m * z * d1**4) - 1
        gx[:, 3] = 1.93 * L2**3 / (m * z * d2**4) - 1
        gx[:, 4] = ((745 * L1 / (m * z)) ** 2 + 1.69 * 1e6)**0.5 / (110 * d1**3) - 1
        gx[:, 5] = ((745 * L2 / (m * z)) ** 2 + 157.5 * 1e6)**0.5 / (85 * d2**3) - 1
        gx[:, 6] = m * z / 40 - 1
        gx[:, 7] = 5 * m / (b) - 1
        gx[:, 8] = b / (12 * m) - 1

        return gx, fx
