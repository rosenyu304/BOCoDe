import torch
from base import BenchmarkProblem

class StyblinskiTang_continuous(BenchmarkProblem):

    r'''

    '''

    # 10D objective, 0 constraints, X = n-by-10

    tags = {"single_objective", "unconstrained", "continuous", "10D"}

    def __init__(self):
        dim_ = 10
        super().__init__(dim = dim_, num_obj = 1, num_cons = 0, optimizers = [[-2.903534] * dim_], optimum = [[-39.16599] * dim_], bounds = [[-5, 5]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        from botorch.test_functions.synthetic import StyblinskiTang as StyblinskiTang_imported

        return None, -StyblinskiTang_imported(X).view(-1, 1)

stm = StyblinskiTang_continuous()
X = ([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
gx, fx = stm.evaluate(X, to_verify = False)
print(gx, fx)
