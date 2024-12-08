import torch
from base import BenchmarkProblem

class StyblinskiTang_continuous(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/stybtang.html
    '''

    # 10D objective, 0 constraints, X = n-by-10

    tags = {"single_objective", "unconstrained", "continuous", "10D", "extra_imports"}

    def __init__(self):
        dim_ = 10
        super().__init__(dim = dim_, num_obj = 1, num_cons = 0, optimizers = [[-2.903534] * dim_], optimum = [[-39.16599] * dim_], bounds = [[-5, 5]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import StyblinskiTang as StyblinskiTang_imported

        fun = StyblinskiTang_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
