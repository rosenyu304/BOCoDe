
from TrussSolvers import *
import torch
import numpy as np





def Truss10D(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."

    n = individuals.size(0)
    
    fx = torch.zeros(n,1)

    # 10 bar stress constraints, 4 displacement constraints
    gx = torch.zeros(n,14)

    for ii in range(n):
        displace, stress, _, _, weights = Truss10bar(individuals[ii,:])
        fx[ii,0] = weights

        for ss in range(10):
            gx[ii,ss] = abs(stress[ss])
            
        gx[ii,10] = displace[1][1]
        gx[ii,11] = displace[2][1]
        gx[ii,12] = displace[4][1]
        gx[ii,13] = displace[5][1]

    return gx, fx

































