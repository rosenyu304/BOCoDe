import torch
from base import BenchmarkProblem

class JLH1(BenchmarkProblem):

    r'''
    Jetton C, Li C, Hoyle C (2023) Constrained bayesian optimization methods using regression
    and classification gaussian processes as constraints. In: International Design Engineering
    Technical Conferences and Computers and Information in Engineering Conference, American
    Society of Mechanical Engineers, pV03BT03A033
    '''

    # 2D objective, 1 constraint, X = n-by-2

    tags = {"single_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 1, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]

        test_function = (- (x1-0.5)**2 - (x2-0.5)**2 )
        fx = test_function.reshape(n, self.num_obj)
        gx = (x1 + x2 - 0.75).reshape(n, self.num_cons)

        return gx, fx
