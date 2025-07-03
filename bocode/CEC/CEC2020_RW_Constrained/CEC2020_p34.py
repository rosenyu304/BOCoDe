"""
https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation/
"""

from pathlib import Path

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType


class CEC2020_p34(BenchmarkProblem):
    """
    CEC2020_34 problem 34
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 118
    num_constraints = 108

    def __init__(self):
        super().__init__(
            dim=118,
            num_objectives=1,
            num_constraints=108,
            #  X_opt= [[0] * 118],
            optimum=[0.0],
            bounds=[(-1.0, 1.0)] * 118,
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        X = X.numpy()

        # Load data for Problem 34
        # INPUT_DATA = './PFNBO_Experiments/TestProblems_Utils/CEC2020_powersystems/'
        # G = np.loadtxt(f'{INPUT_DATA}/FunctionPS1_G.txt')
        # B = np.loadtxt(f'{INPUT_DATA}/FunctionPS1_B.txt')
        # P = np.loadtxt(f'{INPUT_DATA}/FunctionPS1_P.txt')
        # Q = np.loadtxt(f'{INPUT_DATA}/FunctionPS1_Q.txt')

        script_dir = Path(__file__).parent
        path = script_dir / "input_data" / "FunctionPS1_G.txt"
        G = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS1_B.txt"
        B = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS1_P.txt"
        P = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS1_Q.txt"
        Q = np.loadtxt(path)

        Y = G + 1j * B
        n_samples = X.shape[0]

        # Initialize base voltages (repeated for each sample)
        V_base = np.zeros((n_samples, 30), dtype=complex)
        V_base[:, 0] = 1
        V_base[:, 1] = np.exp(1j * 4 * np.pi / 3)
        V_base[:, 2] = np.exp(1j * 2 * np.pi / 3)

        # Set voltages from input x
        V_base[:, 3:30] = X[:, 0:27] + 1j * X[:, 27:54]

        # Initialize power arrays
        Pdg = np.zeros((n_samples, 30))
        Qdg = np.zeros((n_samples, 30))
        Psp = np.zeros((n_samples, 30))
        Qsp = np.zeros((n_samples, 30))

        # Set values from x
        Psp[:, 3:30] = X[:, 54:81]
        Qsp[:, 3:30] = X[:, 81:108]
        Pdg[:, [8, 15, 20, 23, 29]] = X[:, 108:113]
        Qdg[:, [8, 15, 20, 23, 29]] = X[:, 113:118]

        # Calculate currents (vectorized for all samples)
        I = V_base @ Y.T  # Matrix multiplication for all samples at once
        Ir = np.real(I)
        Im = np.imag(I)

        spI = np.conj((Psp + 1j * Qsp) / V_base)
        spIr = np.real(spI)
        spIm = np.imag(spI)

        # Calculate deltas
        delP = Psp - Pdg - P
        delQ = Qsp - Qdg - Q
        delIr = Ir - spIr
        delIm = Im - spIm

        # Objective function (vectorized)
        f = abs(I[:, 0] + I[:, 1] + I[:, 2]) + abs(
            I[:, 0]
            + np.exp(1j * 4 * np.pi / 3) * I[:, 1]
            + np.exp(1j * 2 * np.pi / 3) * I[:, 2]
        )

        # Combine all equality constraints
        h = np.concatenate(
            [delIr[:, 3:30], delIm[:, 3:30], delP[:, 3:30], delQ[:, 3:30]], axis=1
        )

        # No inequality constraints
        # g = np.zeros((n_samples, 1))

        if self.is_constrained:
            if "penalty_constrained" in self.flag:
                return (
                    None,
                    None,
                    -(
                        torch.from_numpy(f)
                        + torch.from_numpy((np.sum(abs(h), axis=1) - 1e-4) / 100)
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
