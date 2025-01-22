import torch
import numpy as np
from .base import BenchmarkProblem

class CEC2020_p1(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 1
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=9, 
                         num_obj=1, 
                         num_cons=8, 
                         optimizers=[[0] * 9], 
                         optimum=[[0]], 
                         bounds=[[0, 10], [0, 200], [0, 100], [0, 200], [1000, 2000000], [0, 600], [100, 600], [100, 600], [100, 900]],
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

        # Constraints
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

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)


class CEC2020_p2(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 2
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=11, 
                         num_obj=1, 
                         num_cons=9, 
                         optimizers=[[0] * 9], 
                         optimum=[[0]], 
                         bounds=[[1e4, 0.819e6], [1e4, 1.131e6], [1e4, 2.05e6], [0, 0.05074], [0, 0.05074], [0, 0.05074], [100, 200], [100, 300], [100, 300], [100, 300], [100, 400]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        # Objective function
        f = (X[:, 0] / (120 * X[:, 3]))**0.6 + (X[:, 1] / (80 * X[:, 4]))**0.6 + (X[:, 2] / (40 * X[:, 5]))**0.6

        # Constraints
        h = np.zeros((n_samples, 9))
        h[:, 0] = X[:, 0] - 1e4 * (X[:, 6] - 100)
        h[:, 1] = X[:, 1] - 1e4 * (X[:, 7] - X[:, 6])
        h[:, 2] = X[:, 2] - 1e4 * (500 - X[:, 7])
        h[:, 3] = X[:, 0] - 1e4 * (300 - X[:, 8])
        h[:, 4] = X[:, 1] - 1e4 * (400 - X[:, 9])
        h[:, 5] = X[:, 2] - 1e4 * (600 - X[:, 10])
        h[:, 6] = X[:, 3] * np.log(np.abs(X[:, 8] - 100) + 1e-8) - X[:, 3] * np.log(300 - X[:, 6] + 1e-8) - X[:, 8] - X[:, 6] + 400
        h[:, 7] = X[:, 4] * np.log(np.abs(X[:, 9] - X[:, 6]) + 1e-8) - X[:, 4] * np.log(np.abs(400 - X[:, 7]) + 1e-8) - X[:, 9] + X[:, 6] - X[:, 7] + 400
        h[:, 8] = X[:, 5] * np.log(np.abs(X[:, 10] - X[:, 7]) + 1e-8) - X[:, 5] * np.log(100) - X[:, 10] + X[:, 7] + 100

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)
