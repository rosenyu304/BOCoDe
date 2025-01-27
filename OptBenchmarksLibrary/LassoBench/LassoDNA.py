import torch
from ..base import *

class LassoDNA(BenchmarkProblem):

    r'''
    ...
    '''

    def __init__(self):
        
        tags = ["LassoDNA",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: 180-Dim", 
                "IMPORTS: LassoBench",
               ]
        
        super().__init__(dim=180, 
                         num_obj = 1, 
                         num_cons = 0, 
                         bounds = [[-1, 1]], 
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        real_bench = LassoBench.RealBenchmark(pick_data='DNA')
        for i in range(X.shape[0]):
            # loss = real_bench.evaluate(X[i,:].numpy())
            fx[i,0] = -real_bench.evaluate(X[i,:].to(torch.double).numpy())


        return None, fx
    
