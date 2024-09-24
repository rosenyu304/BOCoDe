# Author: Qiuyi

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

def cube(n_samples, dim=2):
    return torch.rand([n_samples, dim])

def sphere(n_samples, dim=1):
    X = torch.randn([n_samples, dim+1])
    return F.normalize(X, dim=-1)

def torus(n_samples, R=2, r=1):
    u, v = (torch.rand([n_samples, 2]) * 2 * torch.pi).chunk(2, -1)
    x = (R + r * torch.cos(u)) * torch.cos(v)
    y = (R + r * torch.cos(u)) * torch.sin(v)
    z = r * torch.sin(u)
    return torch.cat([x, y, z], dim=-1)

def mobius_strip(n_samples):
    u, v = torch.rand([n_samples, 2]).chunk(2, -1)
    u *= 2 * torch.pi
    x = (1 + v/2 * torch.cos(u/2)) * torch.cos(u)
    y = (1 + v/2 * torch.cos(u/2)) * torch.sin(u)
    z = v/2 * torch.sin(u/2)
    return torch.cat([x, y, z], dim=-1)

def klein_bottle(n_samples, R=2, P=1, e=0.1):
    u, v = (torch.rand([n_samples, 2]) * 2 * torch.pi).chunk(2, -1)
    x = R * (torch.cos(u/2) * torch.cos(v) - torch.sin(u/2) * torch.sin(v*2))
    y = R * (torch.sin(u/2) * torch.cos(v) + torch.cos(u/2) * torch.sin(v*2))
    z = P * torch.cos(u) * (1 + e * torch.sin(v))
    w = P * torch.sin(u) * (1 + e * torch.sin(v))
    return torch.cat([x, y, z, w], dim=-1)


################################ Immersions ################################
# For nonlinearly embedding datasets into high dimensional ambient spaces. #
############################################################################

def polynomial_immersion(dataset, ambient_dim, weight=(1,1,0)):
    """
    The graph of a continous function is homeomorphic to its domain.
    Immersion is performed by polynomial function.
    """

    X = (dataset - dataset.mean(0)) / (dataset.std(0) * 2)
    XX = (X.unsqueeze(1) * X.unsqueeze(2)).flatten(1)
    XXX = (X.view(len(X), 1, 1, -1) \
           * X.view(len(X), 1, -1, 1) \
            * X.view(len(X), -1, 1, 1)).flatten(1)
    
    in_dim = dataset.size(-1)
    out_dim = ambient_dim - in_dim

    # 2nd order polynomial function Y = f(X)
    W1 = torch.randn(X.size(-1), out_dim) * weight[0]
    W2 = torch.randn(XX.size(-1), out_dim) * weight[1]
    W3 = torch.randn(XXX.size(-1), out_dim) * weight[2]
    Y = X @ W1 + XX @ W2 + XXX @ W3

    # rescaling
    D = torch.cat([dataset, Y], dim=-1) # graph of polynomial f
    D = torch.as_tensor(PCA(D.size(-1)).fit_transform(D), dtype=torch.float)
    s = torch.rand(D.size(-1)) * (1-0.5) + 0.5
    D = D / D.std(0).clip(3e-7) * s
    return D