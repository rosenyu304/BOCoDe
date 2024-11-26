import torch
import numpy as np
from botorch.test_functions.synthetic import Branin

from .cont_to_disc import cont_to_disc


def Branin2DEmbedd(X_input):
    # assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    # Set function here:
    X = X_input[:,[1,2]]
    
    n = X.size(0)
    dimm = 2
    fun = Branin(negate=True)
    fun.bounds[0, 0].fill_(-5.0)
    fun.bounds[1, 0].fill_(10.0)
    fun.bounds[0, 1].fill_(0.0)
    fun.bounds[1, 1].fill_(15.0)
    dim = fun.dim
    lb, ub = fun.bounds
    
    fx = fun(X)
    fx = fx.reshape((n, 1))
    gx = 0
    return gx, fx



def Branin2DEmbedd_Scaling(X):
    X_scaled = X.clone()
    X_scaled[:,1] = X_scaled[:,1]*15-5
    X_scaled[:,2] = X_scaled[:,2]*15
    return X_scaled




def BraninMixed(X):
    # Optimal (disc): (-2.6, 10) -> -2.79118
    # LVGP paper: https://www.nature.com/articles/s41598-020-60652-9
    return -Branin()(X).view(-1, 1)


def Branin_MixedScaling(X):
    # x0: [-5, 10]
    # x1: {0, 5, 10, 15}
    X_scaled = X.clone()
    X_scaled[:, 0] = X[:, 0] * 15 - 5
    X_scaled[:, 1] = cont_to_disc(X[:, 1], torch.tensor([0, 5, 10, 15]))
    return X_scaled
