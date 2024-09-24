import torch
import numpy as np
from botorch.test_functions.synthetic import Griewank
device = torch.device("cpu")
dtype = torch.double


def Griewank2DEmbedd(X_input):
    # assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    # Set function here:
    X = X_input[:,[1,2]]
    
    n = X.size(0)
    dimm = 2
    fun = Griewank(dim=dimm, negate=True)
    fun.bounds[0, :].fill_(-600.0)
    fun.bounds[1, :].fill_(600.0)
    dim = fun.dim
    lb, ub = fun.bounds
    
    fx = fun(X)
    fx = fx.reshape((n, 1))
    gx = 0
    return gx, fx



def Griewank2DEmbedd_Scaling(X):
    X_scaled = X.clone()
    X_scaled = X_scaled * 1200 - 600

    return X_scaled




