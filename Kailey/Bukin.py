import torch
from base import BenchmarkProblem

class Bukin(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/bukin6.html
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

        return None, -fx
