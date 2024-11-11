import torch
from botorch.test_functions.synthetic import StyblinskiTang

from .cont_to_disc import cont_to_disc



def StyblinskiTang(X):
    # Optimal (cont): [-2.903534]^dim -> -39.16599 * dim
    return -StyblinskiTang(X).view(-1, 1)


def StyblinskiTang_Scaling(X):
    # X: [-5, 5]^dim
    X_scaled = X * 10 - 5
    return X_scaled    


def StyblinskiTangMixed_Scaling(X):
    # X: {-5, -2.5, 0, 2.5, 5}^dim
    X_scaled = cont_to_disc(X, torch.tensor([-5, -2.5, 0, 2.5, 5]))
    return X_scaled


