import torch
from ..base import *

class LassoSyntHigh(BenchmarkProblem):

    r'''
    ...
    '''

    def __init__(self):
        
        tags = ["LassoSyntHigh",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: 300-Dim", 
                "IMPORTS: LassoBench",
               ]
        
        super().__init__(dim=300, 
                         num_obj = 1, 
                         num_cons = 0, 
                         bounds = [[-1, 1]], 
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_high')
        for i in range(X.shape[0]):
            fx[i,0] = -synt_bench.evaluate(X[i,:].to(torch.double).numpy())


        return None, fx

