import torch
from ..base import *

class Levy(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/levy.html
    '''

    def __init__(self, dim=2):

        tags = ["Levy",
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
                         bounds = [[-10, 10]],
                         tags = tags
                        )

    def _evaluate_implementation(self, X):

        from botorch.test_functions.synthetic import Levy as Levy_imported

        fun = Levy_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx








