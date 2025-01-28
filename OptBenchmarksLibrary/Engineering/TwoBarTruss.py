import torch
from .base import BenchmarkProblem

class TwoBarTruss(BenchmarkProblem):

    r'''
    S. S. Rao. Game theory approach for multiobjective structural optimization. 
    Computers and Structures 26(1):119–127, 1987  
    '''

    # 2D objective, 5 constraints, X = 2-by-dim

    tags = {"multi_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(dim = 2, num_obj = 2, num_cons = 5, bounds = [[]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        n = X.size(0)
      
        x1 = X[:, 0]
        x2 = X[:, 1]

        fx = torch.zeros((n, self.num_obj))
        # negate for maximization
        fx[:, 0] = 

        gx = torch.zeros((n, self.num_cons))
        gx[:, 0] =

        return gx, fx
