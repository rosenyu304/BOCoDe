import torch
from ..base import *

# Prevents SSL certificate validity error when fetching data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

class LassoBreastCancer(BenchmarkProblem):

    r'''
    ...
    '''

    def __init__(self):
        
        tags = ["LassoBreastCancer",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: 10-Dim", 
                "IMPORTS: LassoBench",
               ]

        super().__init__(dim=10, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [(-1, 1)]*10, 
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        real_bench = LassoBench.RealBenchmark(pick_data='breast_cancer')
        for i in range(X.shape[0]):
            # loss = real_bench.evaluate(X[i,:].numpy())
            fx[i,0] = -real_bench.evaluate(X[i,:].to(torch.double).numpy())


        return None, fx
    
