import torch
import numpy as np
from botorch.test_functions.synthetic import DixonPrice
device = torch.device("cpu")
dtype = torch.double



def DixonPriceND(individuals): 

    # assert torch.is_tensor(individuals) and individuals.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    
    #############################################################################
    #############################################################################
    # Set function here:
    dimm = individuals.shape[1]
    fun = DixonPrice(dim=dimm, negate=True)
    fun.bounds[0, :].fill_(-10.0)
    fun.bounds[1, :].fill_(10.0)
    dim = fun.dim
    lb, ub = fun.bounds
    #############################################################################
    #############################################################################
    
    
    n = individuals.size(0)

    fx = fun(individuals)
    fx = fx.reshape((n, 1))

    #############################################################################
    ## Constraints
    # gx1 = torch.sum(individuals,1)  # sigma(x) <= 0 
    # gx1 = gx1.reshape((n, 1))

    # gx2 = torch.norm(individuals, p=2, dim=1)-5  # norm_2(x) -3 <= 0
    # gx2 = gx2.reshape((n, 1))

    # gx = torch.cat((gx1, gx2), 1)
    #############################################################################
    
    return 0, fx
    # return gx, fx



def DixonPriceND_Scaling(X):

    # assert torch.is_tensor(X) and X.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    
    X_scaled = X*20-10
    
    return X_scaled





