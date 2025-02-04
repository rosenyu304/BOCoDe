import torch
from ..base import *

class Rosenbrock(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/rosen.html
    '''

    def __init__(self, dim=2):

        tags = ["Rosenbrock",
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
                         bounds = [[-5, 10]],
                         tags = tags,
                        )

    def _evaluate_implementation(self, X):

        from botorch.test_functions.synthetic import Rosenbrock as Rosenbrock_imported

        fun = Rosenbrock_imported(dim=self.dim, negate=True)
        fun.bounds[0, :].fill_(self.bounds[0][0])
        fun.bounds[1, :].fill_(self.bounds[0][1])

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
