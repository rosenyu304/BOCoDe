import torch
from base import BenchmarkProblem

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

        n = X.size(0)

        gx = torch.zeros((n, self.num_cons))

        fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
        fun.bounds[0, :].fill_(-5)
        fun.bounds[1, :].fill_(10)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        gx[:, 0] = torch.sum(X,1)
        gx[:, 1] = (torch.norm(X, p=2, dim=1)-5)

        return gx, fx
