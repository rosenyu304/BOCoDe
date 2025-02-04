import torch
from ..base import *

class Goldstein(BenchmarkProblem):

    r'''
    (Alt Name: Goldstein-Price)
    LVGP paper: https://www.nature.com/articles/s41598-020-60652-9
    '''

    def __init__(self, is_mixed = True, debug = False):

        tags = ["Goldstein",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous / Mixed", 
                "SCALABLE: 2-Dim", 
                "IMPORTS: N/A",
               ]
        
        super().__init__(dim = 2, 
                         num_obj = 1, 
                         num_cons = 0, 
                         optimum = [-3], 
                         bounds = [[-2, 2], [0, 1]],
                         is_mixed = is_mixed,
                         debug = debug,
                         tags = tags
                        )

    def _evaluate_implementation(self, X):

        # x0: [-2, 2]
        # x1: {-2, -1, 0, 1, 2}

        if self.is_mixed:
            X[:,1] = super().cont_to_disc(X[:,1], torch.tensor([-2, -1, 0, 1, 2]))

        if self.debug:
            print(f'X: {X}')

        fx = -((1 + (X[:,0] + X[:,1] +1)**2
                * (19 - 14*X[:,0] + 3*X[:,0]**2 -14*X[:,1]
                    +6*X[:,0]*X[:,1] + 3*X[:,1]**2
                    )
                ) *
                (
                    30 + (2*X[:,0] - 3*X[:,1])**2
                    * (18- 32*X[:,0] + 12*X[:,0]**2 + 48*X[:,1]
                        -36*X[:,0]*X[:,1] + 27*X[:,1]**2
                    )
                ))

        return None, fx
