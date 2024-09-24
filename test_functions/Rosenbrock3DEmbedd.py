import torch
import numpy as np
from botorch.test_functions.synthetic import Rosenbrock
device = torch.device("cpu")
dtype = torch.double


def Rosenbrock3DEmbedd(X_input):
    # assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    # Set function here:
    X = X_input[:,[1,2,3]]
    n = X.size(0)
    dimm = 3
    fun = Rosenbrock(dim=dimm, negate=True)
    fun.bounds[0, :].fill_(-2.0)
    fun.bounds[1, :].fill_(2.0)

    
    dim = fun.dim
    lb, ub = fun.bounds
    
    fx = fun(X)
    fx = fx.reshape((n, 1))
    gx = 0
    return gx, fx



def Rosenbrock3DEmbedd_Scaling(X):
    X_scaled = X.clone()
    
    X_scaled[:,1] = X_scaled[:,1]*4 - 2
    X_scaled[:,2] = X_scaled[:,2]*4 - 2
    X_scaled[:,3] = X_scaled[:,3]*4 - 2
    
    return X_scaled




