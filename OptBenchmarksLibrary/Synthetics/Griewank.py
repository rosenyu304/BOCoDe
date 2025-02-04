import torch
from ..base import *

class Griewank(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/griewank.html
    '''

    def __init__(self, dim=2):

        tags = ["Griewank",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim", 
                "IMPORTS: BoTorch",
               ]
        
        super().__init__(dim, 
                         num_obj = 1, 
                         num_cons = 0, 
                         bounds = [[-600, 600]],
                         tags = tags
                        )

    def _evaluate_implementation(self, X):

        from botorch.test_functions.synthetic import Griewank as Griewank_imported

        fun = Griewank_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
