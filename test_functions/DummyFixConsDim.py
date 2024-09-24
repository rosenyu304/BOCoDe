import torch
import numpy as np
import math
from botorch.test_functions.synthetic import Michalewicz
device = torch.device("cpu")
dtype = torch.double



def DummyFixConsDim(individuals): 
    n = individuals.size(0)
    GX = torch.rand(n, 20)
    Y = torch.rand(n, 1)

    return GX, Y



def DummyFixConsDim_Scaling(X):
    
    X_scaled = X
    
    return X_scaled





