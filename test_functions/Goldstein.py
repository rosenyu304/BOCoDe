import torch

from .cont_to_disc import cont_to_disc

# LVGP paper: https://www.nature.com/articles/s41598-020-60652-9


def Goldstein(X):
    # Optimal: (0, -1) -> -3
    fx = ((1 + (X[:,0] + X[:,1] +1)**2 
          * (19 - 14*X[:,0] + 3*X[:,0]**2 -14*X[:,1]
             +6*X[:,0]*X[:,1] + 3*X[:,1]**2
            )
         ) *
         (
             30 + (2*X[:,0] - 3*X[:,1])**2
             * (18- 32*X[:,0] + 12*X[:,0]**2 + 48*X[:,1]
                -36*X[:,0]*X[:,1] + 27*X[:,1]**2
               )
         ))
    return -fx.reshape(-1, 1)

def Goldstein_Scaling(X):
    X_scaled = X*4 - 2
    return X_scaled


def Goldstein_Scaling(X):
    # x0: [-2, 2]
    # x1: {-2, -1, 0, 1, 2}
    X_scaled = torch.zeros(X.shape, dtype=dtype, device=device)
    X_scaled[:,0] = X[:,0] * 4 - 2
    X_scaled[:,1] = cont_to_disc(X[:,1], torch.tensor([-2, -1, 0, 1, 2]))
    return X_scaled