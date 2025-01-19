import torch
from ..base import *

class Rastrigin(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/stybtang.html
    '''

    def __init__(self, dim=2):

        tags = ["Rastrigin",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim", 
                "IMPORTS: BoTorch",
               ]
        
        super().__init__(dim = dim, 
                         num_obj = 1, 
                         num_cons = 0,  
                         optimum = [[0] * dim], 
                         bounds = [[-5.12, 5.12]],
                         tags=tags,
                        )

    def _evaluate_implementation(self, X):

        from botorch.test_functions.synthetic import Rastrigin as Rastrigin_imported

        fun = Rastrigin_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx



