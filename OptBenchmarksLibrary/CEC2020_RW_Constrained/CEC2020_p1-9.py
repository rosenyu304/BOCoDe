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

        # Inequality constraints
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
                         optimizers=[[0] * 11], 
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

        # Equality constraints
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


class CEC2020_p3(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 3
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=7, 
                         num_obj=1, 
                         num_cons=0, 
                         optimizers=[[0] * 7], 
                         optimum=[[0]], 
                         bounds=[[1000, 2000], [0, 100], [2000, 4000], [0, 100], [0, 100], [0, 20], [0, 200]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        # Objective function
        f = (-1.715 * X[:, 0] - 0.035 * X[:, 0] * X[:, 5] - 4.0565 * X[:, 2] - 10.0 * X[:, 1] + 0.063 * X[:, 2] * X[:, 4])

        # No equality constraints
        h = ((n_samples, 0)) 

        # Inequality constraints
        g = np.zeros((n_samples, 14))  
        g[:, 0] = 0.0059553571 * X[:, 5] ** 2 * X[:, 0] + 0.88392857 * X[:, 2] - 0.1175625 * X[:, 5] * X[:, 0] - X[:, 0]
        g[:, 1] = 1.1088 * X[:, 0] + 0.1303533 * X[:, 0] * X[:, 5] - 0.0066033 * X[:, 0] * X[:, 5] ** 2 - X[:, 2]
        g[:, 2] = (6.66173269 * X[:, 5] ** 2 + 172.39878 * X[:, 4] - 56.596669 * X[:, 3] - 191.20592 * X[:, 5] - 10000)
        g[:, 3] = 1.08702 * X[:, 5] + 0.32175 * X[:, 3] - 0.03762 * X[:, 5] ** 2 - X[:, 4] + 56.85075
        g[:, 4] = 0.006198 * X[:, 6] * X[:, 3] * X[:, 2] + 2462.3121 * X[:, 1] - 25.125634 * X[:, 1] * X[:, 3] - X[:, 2] * X[:, 3]
        g[:, 5] = 161.18996 * X[:, 2] * X[:, 3] + 5000.0 * X[:, 1] * X[:, 3] - 489510.0 * X[:, 1] - X[:, 2] * X[:, 3] * X[:, 6]
        g[:, 6] = 0.33 * X[:, 6] - X[:, 4] + 44.333333
        g[:, 7] = 0.022556 * X[:, 4] - 0.007595 * X[:, 6] - 1.0
        g[:, 8] = 0.00061 * X[:, 2] - 0.0005 * X[:, 0] - 1.0
        g[:, 9] = 0.819672 * X[:, 0] - X[:, 2] + 0.819672
        g[:, 10] = 24500.0 * X[:, 1] - 250.0 * X[:, 1] * X[:, 3] - X[:, 2] * X[:, 3]
        g[:, 11] = 1020.4082 * X[:, 3] * X[:, 1] + 1.2244898 * X[:, 2] * X[:, 3] - 100000.0 * X[:, 1]
        g[:, 12] = 6.25 * X[:, 0] * X[:, 5] + 6.25 * X[:, 0] - 7.625 * X[:, 2] - 100000.0
        g[:, 13] = 1.22 * X[:, 2] - X[:, 5] * X[:, 0] - X[:, 0] + 1.0

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)


class CEC2020_p4(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 4
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=6, 
                         num_obj=1, 
                         num_cons=4, 
                         optimizers=[[0] * 6], 
                         optimum=[[0]], 
                         bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [1e-5, 16], [1e-5, 16]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        # Constants
        k1 = 0.09755988
        k2 = 0.99 * k1
        k3 = 0.0391908
        k4 = 0.9 * k3
        
        # Objective function
        f = -X[:, 3]
        
        # Equality constraints
        h = np.zeros((n_samples, 4))
        h[:, 0] = X[:, 0] + k1 * X[:, 1] * X[:, 4] - 1
        h[:, 1] = X[:, 1] - X[:, 0] + k2 * X[:, 1] * X[:, 5]
        h[:, 2] = X[:, 2] + X[:, 0] + k3 * X[:, 2] * X[:, 4] - 1
        h[:, 3] = X[:, 3] - X[:, 2] + X[:, 1] - X[:, 0] + k4 * X[:, 3] * X[:, 5]
        
        # Inequality constraints
        g = np.zeros((n_samples, 1))
        g[:, 0] = np.sqrt(X[:, 4]) + np.sqrt(X[:, 5]) - 4

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)



class CEC2020_p5(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 5
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=9, 
                         num_obj=1, 
                         num_cons=4, 
                         optimizers=[[0] * 9], 
                         optimum=[[0]], 
                         bounds=[[0, 100], [0, 200], [0, 100], [0, 100], [0, 100], [0, 100], [0, 200], [0, 100], [0, 200]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        # Objective function
        f = -(9 * x[:, 0] + 15 * x[:, 1] - 6 * x[:, 2] - 16 * x[:, 3] - 10 * (x[:, 4] + x[:, 5]))
        
        # Inequality constraints
        g = np.zeros((x.shape[0], 2))
        g[:, 0] = x[:, 8] * x[:, 6] + 2 * x[:, 4] - 2.5 * x[:, 0]
        g[:, 1] = x[:, 8] * x[:, 7] + 2 * x[:, 5] - 1.5 * x[:, 1]
        
        # Equality constraints
        h = np.zeros((x.shape[0], 4))
        h[:, 0] = x[:, 6] + x[:, 7] - x[:, 2] - x[:, 3]
        h[:, 1] = x[:, 0] - x[:, 6] - x[:, 4]
        h[:, 2] = x[:, 1] - x[:, 7] - x[:, 5]
        h[:, 3] = x[:, 8] * x[:, 6] + x[:, 8] * x[:, 7] - 3 * x[:, 2] - x[:, 3]

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)



class CEC2020_p6(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 6
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=38, 
                         num_obj=1, 
                         num_cons=32, 
                         optimizers=[[0] * 38], 
                         optimum=[[0]], 
                         bounds=[[0, 90], [0, 150], [0, 90], [0, 150], [0, 90], [0, 90], [0, 150], [0, 90], [0, 90], [0, 90], [0, 150], [0, 150], [0, 90], [0, 90], [0, 150], [0, 90], [0, 150], [0, 90], [0, 150], [0, 90], [0, 1], [0, 1.2], [0, 1], [0, 1], [0, 1], [0, 0.5], [0, 1], [0, 1], [0, 0.5], [0, 0.5], [0, 0.5], [0, 1.2], [0, 0.5], [0, 1.2], [0, 1.2], [0, 0.5], [0, 1.2], [0, 1.2]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        # Objective function
        f = 0.9979 + 0.00432 * x[:, 4] + 0.01517 * x[:, 12]
        
        # No inequality constraints
        g = np.zeros((n_samples, 1))
        
        # Equality constraints
        h = np.zeros((n_samples, 32))
        h[:, 0] = x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] - 300
        h[:, 1] = x[:, 5] - x[:, 6] - x[:, 7]
        h[:, 2] = x[:, 8] - x[:, 9] - x[:, 10] - x[:, 11]
        h[:, 3] = x[:, 13] - x[:, 14] - x[:, 15] - x[:, 16]
        h[:, 4] = x[:, 17] - x[:, 18] - x[:, 19]
        h[:, 5] = x[:, 4] * x[:, 20] - x[:, 5] * x[:, 21] - x[:, 8] * x[:, 22]
        h[:, 6] = x[:, 4] * x[:, 23] - x[:, 5] * x[:, 24] - x[:, 8] * x[:, 25]
        h[:, 7] = x[:, 4] * x[:, 26] - x[:, 5] * x[:, 27] - x[:, 8] * x[:, 28]
        h[:, 8] = x[:, 12] * x[:, 29] - x[:, 13] * x[:, 30] - x[:, 17] * x[:, 31]
        h[:, 9] = x[:, 12] * x[:, 32] - x[:, 13] * x[:, 33] - x[:, 17] * x[:, 34]
        h[:, 10] = x[:, 12] * x[:, 35] - x[:, 13] * x[:, 36] - x[:, 17] * x[:, 37]
        h[:, 11] = 1 / 3 * x[:, 0] + x[:, 14] * x[:, 30] - x[:, 4] * x[:, 20]
        h[:, 12] = 1 / 3 * x[:, 0] + x[:, 14] * x[:, 33] - x[:, 4] * x[:, 23]
        h[:, 13] = 1 / 3 * x[:, 0] + x[:, 14] * x[:, 36] - x[:, 4] * x[:, 26]
        h[:, 14] = 1 / 3 * x[:, 1] + x[:, 9] * x[:, 22] - x[:, 12] * x[:, 29]
        h[:, 15] = 1 / 3 * x[:, 1] + x[:, 9] * x[:, 25] - x[:, 12] * x[:, 32]
        h[:, 16] = 1 / 3 * x[:, 1] + x[:, 9] * x[:, 28] - x[:, 12] * x[:, 35]
        h[:, 17] = (1 / 3 * x[:, 2] + x[:, 6] * x[:, 21] + x[:, 10] * x[:, 22] + x[:, 15] * x[:, 30] + x[:, 18] * x[:, 31] - 30)
        h[:, 18] = (1 / 3 * x[:, 2] + x[:, 6] * x[:, 24] + x[:, 10] * x[:, 25] + x[:, 15] * x[:, 33] + x[:, 18] * x[:, 34] - 50)
        h[:, 19] = (1 / 3 * x[:, 2] + x[:, 6] * x[:, 27] + x[:, 10] * x[:, 28] + x[:, 15] * x[:, 36] + x[:, 18] * x[:, 37] - 30)
        h[:, 20] = x[:, 20] + x[:, 23] + x[:, 26] - 1
        h[:, 21] = x[:, 21] + x[:, 24] + x[:, 27] - 1
        h[:, 22] = x[:, 22] + x[:, 25] + x[:, 28] - 1
        h[:, 23] = x[:, 29] + x[:, 32] + x[:, 35] - 1
        h[:, 24] = x[:, 30] + x[:, 33] + x[:, 36] - 1
        h[:, 25] = x[:, 31] + x[:, 34] + x[:, 37] - 1
        h[:, 26] = x[:, 24]
        h[:, 27] = x[:, 27]
        h[:, 28] = x[:, 22]
        h[:, 29] = x[:, 36]
        h[:, 30] = x[:, 31]
        h[:, 31] = x[:, 34]

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)



class CEC2020_p7(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 7
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=48, 
                         num_obj=1, 
                         num_cons=38, 
                         optimizers=[[0] * 48], 
                         optimum=[[0]], 
                         bounds=[[0.0, 35.0], [0.0, 90.0], [0.0, 90.0], [0.0, 140.0], [0.0, 90.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 90.0], [0.0, 90.0], [0.0, 90.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.849999, 1.0], [0.0, 30.0], [0.849999, 1.0], [0.0, 30.0], [0.849999, 1.0], [0.0, 30.0], [0.0, 1.0], [0.849999, 1.0], [0.0, 30.0], [0.0, 1.0], [0.0, 1.0], [0.0, 30.0], [0.0, 1.0], [0.0, 30.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]
        
        c = np.array([
            [0.23947, 0.75835],
            [-0.0139904, -0.0661588],
            [0.0093514, 0.0338147],
            [0.0077308, 0.0373349],
            [-0.0005719, 0.0016371],
            [0.0042656, 0.0288996]
        ])
        
        # Objective function
        f = (
            c[0, 0] + 
            (c[1, 0] + c[2, 0] * x[:, 23] + c[3, 0] * x[:, 27] + c[4, 0] * x[:, 32] + c[5, 0] * x[:, 33]) * x[:, 4] +
            c[0, 1] + 
            (c[1, 1] + c[2, 1] * x[:, 25] + c[3, 1] * x[:, 30] + c[4, 1] * x[:, 37] + c[5, 1] * x[:, 38]) * x[:, 12]
        )
        
        # No inequality constraints
        g = np.zeros(n_samples)
        
        # Equality constraints
        h = np.zeros((n_samples, 38))
        h[:, 0] = x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] - 300
        h[:, 1] = x[:, 5] - x[:, 6] - x[:, 7]
        h[:, 2] = x[:, 8] - x[:, 9] - x[:, 10] - x[:, 11]
        h[:, 3] = x[:, 13] - x[:, 14] - x[:, 15] - x[:, 16]
        h[:, 4] = x[:, 17] - x[:, 18] - x[:, 19]
        h[:, 5] = x[:, 5] * x[:, 20] - x[:, 23] * x[:, 24]
        h[:, 6] = x[:, 13] * x[:, 21] - x[:, 25] * x[:, 26]
        h[:, 7] = x[:, 8] * x[:, 22] - x[:, 27] * x[:, 28]
        h[:, 8] = x[:, 17] * x[:, 29] - x[:, 30] * x[:, 31]
        h[:, 9] = x[:, 24] - x[:, 4] * x[:, 32]
        h[:, 10] = x[:, 28] - x[:, 4] * x[:, 33]
        h[:, 11] = x[:, 34] - x[:, 4] * x[:, 35]
        h[:, 12] = x[:, 36] - x[:, 12] * x[:, 37]
        h[:, 13] = x[:, 26] - x[:, 12] * x[:, 38]
        h[:, 14] = x[:, 31] - x[:, 12] * x[:, 39]
        h[:, 15] = x[:, 24] - x[:, 5] * x[:, 20] - x[:, 8] * x[:, 40]
        h[:, 16] = x[:, 28] - x[:, 5] * x[:, 41] - x[:, 8] * x[:, 22]
        h[:, 17] = x[:, 34] - x[:, 5] * x[:, 42] - x[:, 8] * x[:, 43]
        h[:, 18] = x[:, 36] - x[:, 13] * x[:, 44] - x[:, 17] * x[:, 45]
        h[:, 19] = x[:, 26] - x[:, 13] * x[:, 21] - x[:, 17] * x[:, 46]
        h[:, 20] = x[:, 31] - x[:, 13] * x[:, 47] - x[:, 17] * x[:, 29]
        h[:, 21] = 1/3 * x[:, 0] + x[:, 14] * x[:, 44] - x[:, 24]
        h[:, 22] = 1/3 * x[:, 0] + x[:, 14] * x[:, 21] - x[:, 28]
        h[:, 23] = 1/3 * x[:, 0] + x[:, 14] * x[:, 47] - x[:, 34]
        h[:, 24] = 1/3 * x[:, 1] + x[:, 9] * x[:, 40] - x[:, 36]
        h[:, 25] = 1/3 * x[:, 1] + x[:, 9] * x[:, 22] - x[:, 26]
        h[:, 26] = 1/3 * x[:, 1] + x[:, 9] * x[:, 43] - x[:, 31]
        h[:, 27] = x[:, 32] + x[:, 33] + x[:, 35] - 1
        h[:, 28] = x[:, 20] + x[:, 41] + x[:, 42] - 1
        h[:, 29] = x[:, 40] + x[:, 22] + x[:, 43] - 1
        h[:, 30] = x[:, 37] + x[:, 38] + x[:, 39] - 1
        h[:, 31] = x[:, 44] + x[:, 21] + x[:, 47] - 1
        h[:, 32] = x[:, 45] + x[:, 46] + x[:, 29] - 1
        h[:, 33] = x[:, 42]
        h[:, 34] = x[:, 45]
        h[:, 35] = 1/3 * x[:, 2] + x[:, 6] * x[:, 20] + x[:, 10] * x[:, 40] + x[:, 15] * x[:, 44] + x[:, 18] * x[:, 45] - 30
        h[:, 36] = 1/3 * x[:, 2] + x[:, 6] * x[:, 41] + x[:, 10] * x[:, 22] + x[:, 15] * x[:, 21] + x[:, 18] * x[:, 46] - 50
        h[:, 37] = 1/3 * x[:, 2] + x[:, 6] * x[:, 42] + x[:, 10] * x[:, 43] + x[:, 15] * x[:, 47] + x[:, 18] * x[:, 29] - 30


        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)



class CEC2020_p8(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 8
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=2, 
                         num_obj=1, 
                         num_cons=0, 
                         optimizers=[[0] * 2], 
                         optimum=[[0]], 
                         bounds=[[0, 1.6], [-0.51, 1.49]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        x[:, 1] = np.round(x[:, 1]) 
        
        # Objective function
        f = 2 * x[:, 0] + x[:, 1]
        
        # Inequality constraints
        g = np.zeros((n_samples, 2))
        g[:, 0] = 1.25 - x[:, 0]**2 - x[:, 1]
        g[:, 1] = x[:, 0] + x[:, 1] - 1.6
        
        # No equality constraints
        h = np.zeros((n_samples, 0))

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)



class CEC2020_p9(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 9
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=3, 
                         num_obj=1, 
                         num_cons=1, 
                         optimizers=[[0] * 3], 
                         optimum=[[0]], 
                         bounds=[[0.5, 1.4], [0.5, 1.4], [-0.51, 1.49]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        x[:, 2] = np.round(x[:, 2])
        
        # Objective function
        f = -x[:, 2] + 2 * x[:, 0] + x[:, 1]
        
        # Equality constraints
        h = x[:, 0] - 2 * np.exp(-x[:, 1])
        
        # Inequality constraints
        g = -x[:, 0] + x[:, 1] + x[:, 2]

        if self.is_constrained:
            return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)

