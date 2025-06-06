import torch
from ..base import *

# Prevents SSL certificate validity error when fetching data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

r'''
    Sources:
    (1) Šehić Kenan, Gramfort Alexandre, Salmon Joseph and Nardi Luigi, "LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso", Proceedings of the 1st International Conference on Automated Machine Learning, 2022.
    (2) https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer
'''

class LassoBreastCancer(BenchmarkProblem):

    r'''
    ...
    '''

    available_dimensions = 10
    num_objectives = 1

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
                         x_opt=[[1, 1, 0.40279274, 0.23285974, -1, 1, -1, 1, 1, 1]],
                         optimum=[-6.138225721920359],
                         tags=tags)

    def _evaluate_implementation(self, X):

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        real_bench = LassoBench.RealBenchmark(pick_data='breast_cancer')
        for i in range(X.shape[0]):
            # loss = real_bench.evaluate(X[i,:].numpy())
            fx[i,0] = -real_bench.evaluate(X[i,:].to(torch.double).numpy())

        return None, fx
    
