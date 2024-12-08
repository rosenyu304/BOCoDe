import torch
from base import BenchmarkProblem

class Levy(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/levy.html
    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 0, bounds = [[-10, 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import Levy as Levy_imported

        fun = Levy_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
