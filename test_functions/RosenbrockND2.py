import torch
import numpy as np
from botorch.test_functions.synthetic import Rosenbrock, Levy, DixonPrice
device = torch.device("cpu")
dtype = torch.double



def RosenbrockND2(individuals): 

    # assert torch.is_tensor(individuals) and individuals.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    
    #############################################################################
    #############################################################################
    # Set function here:
    dimm = individuals.shape[1]
    
    
    Rosenbrockfun = Rosenbrock(dim=dimm, negate=True)
    Rosenbrockfun.bounds[0, :].fill_(-3.0)
    Rosenbrockfun.bounds[1, :].fill_(5.0)

    fx = Rosenbrockfun(individuals)
    fx = fx.reshape(individuals.shape[0],1)

    Levyfun = Levy(dim=dimm, negate=False)
    Levyfun.bounds[0, :].fill_(-3.0)
    Levyfun.bounds[1, :].fill_(5.0)


    DixonPricefun = DixonPrice(dim=dimm, negate=False)
    DixonPricefun.bounds[0, :].fill_(-3.0)
    DixonPricefun.bounds[1, :].fill_(5.0)


    G1 = Levyfun(individuals)  -1e3
    G2 = DixonPricefun(individuals)  -4e7
    
    gx = torch.cat((G1.reshape(individuals.shape[0],1), G2.reshape(individuals.shape[0],1)), 1)
    
    
    return gx, fx


def RosenbrockND2_Scaling(X):

    # assert torch.is_tensor(X) and X.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    
    X_scaled = X*8-3
    
    return X_scaled





