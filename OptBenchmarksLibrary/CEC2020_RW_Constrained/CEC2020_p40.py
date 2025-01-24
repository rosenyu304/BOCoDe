import torch
import numpy as np
from .base import BenchmarkProblem

global initial_flag

class CEC2020_p40(BenchmarkProblem):
    
    r'''
    CEC2020 Problem 40
    ''

    def __init__(self, is_constrained=True, flag=''):
        super().__init__(dim=76, 
                         num_obj=1, 
                         num_cons=76, 
                         optimizers=[[0] * 76], 
                         optimum=[[0]], 
                         bounds=[[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [0, 2], [0, 2]],
                         is_constrained=is_constrained,
                         flag=flag
                        )

    def evaluate(self, X, to_verify=True):
        import numpy as np

        X = super().scale(X, to_verify)
        X = X.numpy()
        
        n_samples = X.shape[0]

        if initial_flag == 0:
            P = np.loadtxt('input data/FunctionPS2_P.txt')
            Q = np.loadtxt('input data/FunctionPS2_Q.txt')
            L = np.loadtxt('input data/FunctionPS14_linedata.txt')
            initial_flag = 1
        
        # Voltage initialization
        V = np.zeros(38, dtype=complex)
        V[0] = 1
        Pc = np.zeros(38)
        Qc = np.zeros(38)
        
        for i in range(n_samples):
            V[1:38] = X[i, 0:37] + 1j * X[i, 37:74]
            Pc[[33, 34, 35, 36, 37]] = 1 / np.array([5.102e-03, 1.502e-03, 4.506e-03, 2.253e-03, 2.253e-03])
            Qc[[33, 34, 35, 36, 37]] = 1 / np.array([0.05, 0.03, 0.05, 0.01, 0.1])
            w = X[i, 74]
            V[0] = X[i, 75] + 1e-5
        
            # Current calculation
            Y = ybus(L, w)
            I = Y @ V
            Ir = np.real(I)
            Im = np.imag(I)
            Vr = np.real(V)
            Vm = np.imag(V)
            Psp = Pc * (1 - w) - P[:, 0] * (np.abs(V) / P[:, 4])**P[:, 5]
            Qsp = Qc * (1 - np.sqrt(Vr**2 + Vm**2)) - Q[:, 0] * (np.abs(V) / Q[:, 4])**Q[:, 5]
            spI = np.conj((Psp + 1j * Qsp) / V)
            spIr = np.real(spI)
            spIm = np.imag(spI)
            delIr = Ir - spIr
            delIm = Im - spIm
            delP2 = Psp - (Vr * Ir + Vm * Im)
            delQ2 = Qsp - (Vm * Ir - Vr * Im)
        
            # Objective calculation
            f[i, 0] = np.sum(delP2[0:38]**2) + np.sum(delQ2[0:38]**2)
            h[i, :] = np.concatenate((delIr[0:38], delIm[0:38]))

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return torch.from_numpy(np.abs(h) - 1e-4), torch.from_numpy(g), -torch.from_numpy(f).unsqueeze(-1)
    

