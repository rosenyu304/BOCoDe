import torch
from base import BenchmarkProblem

class Griewank(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/griewank.html
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
