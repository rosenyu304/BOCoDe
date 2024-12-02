import torch
from base import BenchmarkProblem

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

        # fx = []
        # gx = []

        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]

        # for sx in X:
        gx = (torch.sin(x1) * torch.sin(x2) + 0.95).reshape(n, self.num_cons)
        fx = (- torch.sin(x1) - x2).reshape(n, self.num_obj) # maximize -(x1^2 +x 2^2)
            # gx.append( g )

        # fx = torch.reshape(torch.tensor(fx), (len(fx),1))
        # gx = torch.reshape(torch.tensor(gx), (len(gx),1))

        return gx, fx
