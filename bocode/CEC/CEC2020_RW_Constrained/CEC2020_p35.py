"""
https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation/
"""

from pathlib import Path

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType


class CEC2020_p35(BenchmarkProblem):
    """
    CEC2020_35 problem 35
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 153
    num_constraints = 148

    def __init__(self):
        super().__init__(
            dim=153,
            num_objectives=1,
            num_constraints=148,
            #  X_opt= [[8.9093896456E-02] * 153],
            optimum=[0.079963854],
            bounds=[(-1.0, 1.0)] * 153,
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Load input data once
        # INPUT_DATA = './PFNBO_Experiments/TestProblems_Utils/CEC2020_powersystems/'
        # G = np.loadtxt(f'{INPUT_DATA}/FunctionPS2_G.txt')
        # B = np.loadtxt(f'{INPUT_DATA}/FunctionPS2_B.txt')
        # P = np.loadtxt(f'{INPUT_DATA}/FunctionPS2_P.txt')
        # Q = np.loadtxt(f'{INPUT_DATA}/FunctionPS2_Q.txt')

        script_dir = Path(__file__).parent
        path = script_dir / "input_data" / "FunctionPS2_G.txt"
        G = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS2_B.txt"
        B = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS2_P.txt"
        P = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS2_Q.txt"
        Q = np.loadtxt(path)

        # Complex admittance matrix
        Y = G + 1j * B

        # Initialize voltage vector (38 nodes)
        V = np.zeros((n_samples, 38), dtype=complex)
        V[:, 0] = 1  # Slack bus voltage

        # Initialize power vectors
        Pdg = np.zeros((n_samples, 38))
        Psp = np.zeros((n_samples, 38))
        Qsp = np.zeros((n_samples, 38))

        # Assign variables from x
        V[:, 1:38] = X[:, :37] + 1j * X[:, 37:74]  # Node voltages
        Psp[:, 1:38] = X[:, 74:111]  # Active power specified
        Qsp[:, 1:38] = X[:, 111:148]  # Reactive power specified
        Pdg[:, [33, 34, 35, 36, 37]] = X[:, 148:153]  # DG active power

        # Calculate currents
        I = V @ Y.T  # Matrix multiplication for each sample
        Ir = np.real(I)
        Im = np.imag(I)

        # Complex power injections
        spI = np.conj((Psp + 1j * Qsp) / V)
        spIr = np.real(spI)
        spIm = np.imag(spI)

        # Calculate power mismatches
        V_abs = np.abs(V)
        delP = Psp - Pdg + P[:, 0] * (V_abs / P[:, 4]) ** P[:, 5]
        delQ = Qsp + Q[:, 0] * (V_abs / Q[:, 4]) ** Q[:, 5]

        # Current mismatches
        delIr = Ir - spIr
        delIm = Im - spIm

        # Objective function: Real power at slack bus + sum of specified powers
        f = np.real(V[:, 0] * np.conj(I[:, 0])) + np.sum(Psp[:, 1:38], axis=1)

        # Equality constraints
        h = np.hstack([delIr[:, 1:38], delIm[:, 1:38], delP[:, 1:38], delQ[:, 1:38]])

        # No inequality constraints
        # g = np.zeros((n_samples, 0))

        if self.is_constrained:
            if "penalty_constrained" in self.flag:
                return (
                    None,
                    None,
                    -(
                        torch.from_numpy(f)
                        + torch.from_numpy((np.sum(abs(h), axis=1) - 1e-4) / 5e3)
                    ).unsqueeze(-1),
                )

            else:
                return (
                    torch.from_numpy(abs(h) - 1e-4),
                    None,
                    -torch.from_numpy(f).unsqueeze(-1),
                )
        else:
            return None, None, -torch.from_numpy(f).unsqueeze(-1)

            # if scaling:
            X = super().scale(X)

        # n = X.size(0)

        # gx = torch.zeros((n, self.num_constraints))

        # fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
        # fun.bounds[0, :].fill_(-5)
        # fun.bounds[1, :].fill_(10)

        # fx = fun(X)
        # fx = fx.reshape((n, 1))

        # gX[:, 0] = torch.sum(X,1)
        # gX[:, 1] = (torch.norm(X, p=2, dim=1)-5)

        # if self.is_constrained:
        #     return gx, fx
        # else:
        #     return None, fx
