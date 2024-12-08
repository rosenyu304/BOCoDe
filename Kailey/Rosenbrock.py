import torch
from base import BenchmarkProblem

class Rosenbrock(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/rosen.html
    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2):
        super().__init__(dim, num_obj = 1, num_cons = 0, bounds = [[-5, 10]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import Rosenbrock as Rosenbrock_imported

        fun = Rosenbrock_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
