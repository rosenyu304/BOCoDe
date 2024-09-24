import torch
import numpy as np
from botorch.test_functions.synthetic import Ackley
device = torch.device("cpu")
dtype = torch.double


def Ackley2DEmbedd(X_input):
    # assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    # Set function here:
    X = X_input[:,[1,2]]
    
    n = X.size(0)
    dimm = 2
    fun = Ackley(dim=dimm, negate=True)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds
    
    fx = fun(X)
    fx = fx.reshape((n, 1))
    gx = 0
    return gx, fx



def Ackley2DEmbedd_Scaling(X):
    X_scaled = X.clone()
    X_scaled[:,1] = X_scaled[:,1]*15-5
    X_scaled[:,2] = X_scaled[:,2]*15-5
    return X_scaled




