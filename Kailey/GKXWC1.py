import torch
from base import BenchmarkProblem

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

        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]

        gx = (torch.cos(x1) * torch.cos(x2) - torch.sin(x1) * torch.sin(x2) - 0.5).reshape(n, self.num_cons)
        fx = (- torch.cos(2 * x1) * torch.cos(x2) -  torch.sin(x1)).reshape(n, self.num_obj)

        return gx, fx
