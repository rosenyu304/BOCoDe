import torch
from ..base import *

class StyblinskiTang(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/stybtang.html
    '''

    def __init__(self, dim=10):

        tags = ["StyblinskiTang",
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
                         optimum = [[-39.16599] * dim], 
                         bounds = [[-5, 5]],
                         tags=tags,
                        )

    def _evaluate_implementation(self, X):

        from botorch.test_functions.synthetic import StyblinskiTang as StyblinskiTang_imported

        fun = StyblinskiTang_imported(dim=self.dim, negate=True)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
