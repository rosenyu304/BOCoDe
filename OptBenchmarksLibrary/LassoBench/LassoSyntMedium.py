import torch
from ..base import *

# Prevents SSL certificate validity error when fetching data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LassoSyntMedium(BenchmarkProblem):

    r'''
    ...
    '''

    def __init__(self):

        tags = ["LassoSyntMedium",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: 100-Dim", 
                "IMPORTS: LassoBench",
               ]
        
        super().__init__(dim=100, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [(-1, 1)]*100, 
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_medium')
        for i in range(X.shape[0]):
            fx[i,0] = -synt_bench.evaluate(X[i,:].to(torch.double).numpy())


        return None, fx

