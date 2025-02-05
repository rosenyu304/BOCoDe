import torch
from ..base import *

class DixonPrice(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/dixonpr.html
    '''

    def __init__(self, dim=2):
        
        tags = ["DixonPrice",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim", 
                "IMPORTS: BoTorch",
               ]
        
        super().__init__(dim, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [[-10, 10]],
                         tags = tags
                        )

    def _evaluate_implementation(self, X):

        from botorch.test_functions.synthetic import DixonPrice as DixonPrice_imported

        fun = DixonPrice_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
