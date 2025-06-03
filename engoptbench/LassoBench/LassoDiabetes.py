import torch
from ..base import *

# Prevents SSL certificate validity error when fetching data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

r'''
    Sources:
    (1) Šehić Kenan, Gramfort Alexandre, Salmon Joseph and Nardi Luigi, "LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso", Proceedings of the 1st International Conference on Automated Machine Learning, 2022.
'''

class LassoDiabetes(BenchmarkProblem):

    r'''
    ...
    '''

    available_dimensions = 8
    num_objectives = 1

    def __init__(self):
        
        tags = ["LassoDNA",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: 8-Dim", 
                "IMPORTS: LassoBench",
               ]
        
        super().__init__(dim=8, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [(-1, 1)]*8, 
                         x_opt=[[1,1,-1,1,1,1,1,1]],
                         optimum=[-0.9961401840014688],
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        real_bench = LassoBench.RealBenchmark(pick_data='diabetes')
        for i in range(X.shape[0]):
            # loss = real_bench.evaluate(X[i,:].numpy())
            fx[i,0] = -real_bench.evaluate(X[i,:].to(torch.double).numpy())

        return None, fx
    
