import torch
from .base import BenchmarkProblem

class WaterResources(BenchmarkProblem):

    r'''
    https://github.com/zi-w/Ensemble-Bayesian-Optimization/tree/4e6f9ed04833cc2e21b5906b1181bc067298f914
    '''

    # 3D objective, 7 constraints, X = 7-by-dim

    tags = {"multi_objective", "unconstrained", "continuous", "3D", "extra_imports"}

    def __init__(self):
        super().__init__(dim, num_obj = 5, num_cons = 7, bounds = [[0.01, 0.45], [0.01, 0.10], [0.01, 0.10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        n = X.size(0)
      
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = 

        gx = torch.zeros((n, self.num_cons))

        gx[:, 0] = 0.00139 / (x1 * x2) + 4.94 * x3 - 0.08 - 1
        gx[:, 1] = 0.0000306 / (x1 * x2) + 0.1082 * x3 - 0.00986 - 0.10
        gx[:, 2] = 12.307 / (x1 * x2) + 49408.24 * x3 - 4051.02 - 50000
        gx[:, 3] = 2.098 / (x1 * x2) + 8046.33 * x3 - 696.71 - 16000
        gx[:, 4] = 2.138 / (x1 * x2) + 7883.39 * x3 - 705.04 - 10000
        gx[:, 5] = 0.417 / (x1 * x2) + 1721.26 * x3 - 136.52 - 2000
        gx[:, 6] = 0.164 / (x1 * x2) + 631.13 * x3 - 54.48 - 550

        return gx, fx
