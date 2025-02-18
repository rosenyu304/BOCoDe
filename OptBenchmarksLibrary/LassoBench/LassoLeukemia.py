import torch
from ..base import *

# Prevents SSL certificate validity error when fetching data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LassoLeukemia(BenchmarkProblem):

    r'''
    ...
    '''

    def __init__(self):
        
        tags = ["LassoLeukemia",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: 7129-Dim", 
                "IMPORTS: LassoBench",
               ]
        
        super().__init__(dim=7129, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [(-1, 1)]*7129, 
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        real_bench = LassoBench.RealBenchmark(pick_data='leukemia')
        for i in range(X.shape[0]):
            # loss = real_bench.evaluate(X[i,:].numpy())
            fx[i,0] = -real_bench.evaluate(X[i,:].to(torch.double).numpy())

        return None, fx
    
