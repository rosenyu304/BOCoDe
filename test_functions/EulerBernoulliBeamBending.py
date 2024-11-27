import torch

from .cont_to_disc import cont_to_disc
  

#
#
#   EulerBernoulliBeamBending: 3D mixed-variable problem
#
#   Reference:
#     Cuesta Ramirez, J., Le Riche, R., Roustant, O. et al. 
#     (2022) A comparison of mixed-variables Bayesian optimization 
#     approaches. Adv. Model. and Simul. in Eng. Sci. 9, 6 .
#
#


def EulerBernoulliBeamBending(X):
    # Optimal: (0.0, 0.43, 0.380) -> -1.287*10^3
    # BO comparison paper: https://amses-journal.springeropen.com/articles/10.1186/s40323-022-00218-8
    E = 600
    P = 600
    alpha = 60

    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

    L = 10 + 10 * x1
    S = 1 + x2
    I = x3
    
    D = P * L ** 3 / (3 * E * S**2 * I)
    y = D + alpha * L * S
    return -y.reshape(-1, 1)



def EulerBernoulliBeamBending_MixedScaling(X):
    # x0: [0, 1]
    # x1: [0, 1]
    # x2: {0.083, 0.139, 0.380, 0.080, 0.133, 0.363, 0.086, 0.136, 0.360, 0.092, 0.138, 0.369}
    X_scaled = X.clone()
    X_scaled[:,0] = X[:,0]
    X_scaled[:,1] = X[:,1]
    X_scaled[:,2] = cont_to_disc(X[:,2], torch.tensor([0.083, 0.139, 0.380, 0.080, 0.133, 0.363, 0.086, 0.136, 0.360, 0.092, 0.138, 0.369]))
    return X_scaled




