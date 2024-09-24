import torch
import numpy as np
from botorch.test_functions.synthetic import Hartmann
device = torch.device("cpu")
dtype = torch.double


def Hartmann6DEmbedd(X_input):
    # assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    # Set function here:
    X = X_input[:,[0,1,2,3,4,5]]
    n = X.size(0)
    dimm = 6
    fun = Hartmann(dim=dimm, negate=True)
    fun.bounds[0, :].fill_(0.0)
    fun.bounds[1, :].fill_(1.0)

    
    dim = fun.dim
    lb, ub = fun.bounds
    
    fx = fun(X)
    fx = fx.reshape((n, 1))
    gx = 0
    return gx, fx



def Hartmann6DEmbedd_Scaling(X):
    X_scaled = X.clone()
    return X_scaled




