import torch
from base import BenchmarkProblem

class JLH2(BenchmarkProblem):

    r'''
    Jetton C, Li C, Hoyle C (2023) Constrained bayesian optimization methods using regression
    and classification gaussian processes as constraints. In: International Design Engineering
    Technical Conferences and Computers and Information in Engineering Conference, American
    Society of Mechanical Engineers, pV03BT03A033
    '''

    # 2D objective, 1 constraint, X = n-by-2

    tags = {"single_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 1, num_cons = 1, bounds = [[-5, 0], [-5, 5]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]

        ## Negative sign to make it a maximization problem
        test_function = - (torch.cos(2*x1)*torch.cos(x2) + torch.sin(x1))
        fx = test_function.reshape(n, self.num_obj)
        gx = (((x1 + 5)**2) / 4 + (x2**2) / 100 - 2.5).reshape(n, self.num_cons)

        return gx, fx
