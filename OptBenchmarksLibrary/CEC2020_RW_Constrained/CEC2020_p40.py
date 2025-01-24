import torch
import numpy as np
from .base import BenchmarkProblem

class CEC2020_p40(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 40
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=76, 
                         num_obj=1, 
                         num_cons=76, 
                         optimizers=[[0] * 76], 
                         optimum=[[0]], 
                         bounds=[[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [0, 2], [0, 2]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        # Objective function
        f = 35 * X[:, 0]**0.6 + 35 * X[:, 1]**0.6

        # Equality constraints
        h = np.zeros((n_samples, 8))
        h[:, 0] = 200 * X[:, 0] * X[:, 3] - X[:, 2]
        h[:, 1] = 200 * X[:, 1] * X[:, 5] - X[:, 4]
        h[:, 2] = X[:, 2] - 10000 * (X[:, 6] - 100)
        h[:, 3] = X[:, 4] - 10000 * (300 - X[:, 6])
        h[:, 4] = X[:, 2] - 10000 * (600 - X[:, 7])
        h[:, 5] = X[:, 4] - 10000 * (900 - X[:, 8])
        h[:, 6] = X[:, 3] * np.log(np.abs(X[:, 7] - 100) + 1e-8) - X[:, 3] * np.log(600 - X[:, 6] + 1e-8) - X[:, 7] + X[:, 6] + 500
        h[:, 7] = X[:, 5] * np.log(np.abs(X[:, 8] - X[:, 6]) + 1e-8) - X[:, 5] * np.log(600) - X[:, 8] + X[:, 6] + 600

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
    

