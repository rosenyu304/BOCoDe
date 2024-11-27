import torch
from botorch.test_functions.synthetic import Hartmann

from .cont_to_disc import cont_to_disc


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




def HartmannMixed(X):
    # BO comparison paper: https://amses-journal.springeropen.com/articles/10.1186/s40323-022-00218-8
    return -Hartmann(dim=6)(X).view(-1, 1)


def Hartmann_MixedScaling(X):
    # x4: {0.35, 0.257, 0.477, 0.312, 0.657}
    # x5: {0.150, 0.657, 0.512, 0.741}
    X_scaled = X.clone()
    X_scaled[:, 4] = cont_to_disc(X[:, 4], torch.tensor([0.35, 0.257, 0.477, 0.312, 0.657]))
    X_scaled[:, 5] = cont_to_disc(X[:, 5], torch.tensor([0.150, 0.657, 0.512, 0.741]))
    return X_scaled



