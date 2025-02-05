import torch
from ..base import *

class Michalewicz(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/michal.html
    '''

    def __init__(self, dim=2):
        
        tags = ["Michalewicz",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim", 
                "IMPORTS: BoTorch",
               ]
        
        import math
        super().__init__(dim, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [[0, math.pi]], 
                         tags = tags)

    def _evaluate_implementation(self, X):

        from botorch.test_functions.synthetic import Michalewicz as Michalewicz_imported

        fun = Michalewicz_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
