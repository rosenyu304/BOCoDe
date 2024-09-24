import torch
import numpy as np
from botorch.test_functions import Ackley
device = torch.device("cpu")
dtype = torch.double



def AckleyND(individuals): 

    # assert torch.is_tensor(individuals) and individuals.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    
    #############################################################################
    #############################################################################
    # Set function here:
    dimm = individuals.shape[1]
    fun = Ackley(dim=dimm, negate=True)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds
    #############################################################################
    #############################################################################
    
    
    n = individuals.size(0)

    fx = fun(individuals)
    fx = fx.reshape((n, 1))

    #############################################################################
    ## Constraints
    gx1 = torch.sum(individuals,1)  # sigma(x) <= 0 
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.norm(individuals, p=2, dim=1)-5  # norm_2(x) -3 <= 0
    gx2 = gx2.reshape((n, 1))

    gx = torch.cat((gx1, gx2), 1)
    #############################################################################
    
    
    return gx, fx



def AckleyND_Scaling(X):
    # print(X.shape)
    # assert torch.is_tensor(X) and X.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    
    X_scaled = X*15.0 - 5.0
    
    return X_scaled






















