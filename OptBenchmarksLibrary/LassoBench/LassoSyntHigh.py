import torch
from ..base import *

# Prevents SSL certificate validity error when fetching data
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

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
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [[-1, 1]]*300, 
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_high')
        for i in range(X.shape[0]):
            fx[i,0] = -synt_bench.evaluate(X[i,:].to(torch.double).numpy())


        return None, fx

