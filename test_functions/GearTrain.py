import torch

from .cont_to_disc import cont_to_disc
  
  
#
#
#   GearTrain: 4D mixed-variable problem
#
#   Reference:
#
#     Sandgren, E. (1990). Nonlinear Integer and Discrete 
#     Programming in Mechanical Design Optimization." ASME. 
#     J. Mech. Des. June 1990; 112(2): 223–229.
#
#


def GearTrain(X):
    # Integer Opt for Mech problems: https://asmedigitalcollection.asme.org/mechanicaldesign/article/112/2/223/417355
    return (1/6.931 - (X[:,0]*X[:,1])/(X[:,2]*X[:,3])).reshape(-1, 1)



def GearTrain_Scaling(X):
    # x0, x1, x2, x3: {12, 13, ..., 60}
    X_scaled = cont_to_disc(X, torch.tensor(range(12, 61)))
    return X_scaled