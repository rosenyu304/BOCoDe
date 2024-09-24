import torch
import numpy as np
from botorch.test_functions import Ackley
device = torch.device("cpu")
dtype = torch.double

# class Bukin(SyntheticTestFunction):

#     dim = 2
#     _bounds = [(-15.0, -5.0), (-3.0, 3.0)]
#     _optimal_value = 0.0
#     _optimizers = [(-10.0, 1.0)]
#     _check_grad_at_opt: bool = False


# [docs]
#     def evaluate_true(self, X: Tensor) -> Tensor:
#         part1 = 100.0 * torch.sqrt(torch.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
#         part2 = 0.01 * torch.abs(X[..., 0] + 10.0)
#         return part1 + part2


def Bukin(X): 
    
    part1 = 100.0 * torch.sqrt(torch.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
    part2 = 0.01 * torch.abs(X[..., 0] + 10.0)
    fx = part1 + part2
    
    return fx

def Bukin_Scaling(X): 
    X_scaled = torch.zeros(X.shape)
    
    X_scaled[:,0]  = -(X[..., 0]*10+5)
    X_scaled[:,1]  = X[..., 0]*6-3

    
    return X_scaled








