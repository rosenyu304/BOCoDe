"""
https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation/
"""

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType


class CEC2020_p37(BenchmarkProblem):
    """
    CEC2020_p37
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 126
    num_constraints = 116

    def __init__(self):
        super().__init__(
            dim=126,
            num_objectives=1,
            num_constraints=116,
            #  X_opt= [[0] * 116],
            optimum=[0.018593563],
            bounds=[(-1.0, 1.0)] * 116 + [(0.0, 1.0)] * 10,
        )

    def _evaluate_implementation(self, X, scaling=False):
        from pathlib import Path

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        # Load system data
        # INPUT_DATA = './PFNBO_Experiments/TestProblems_Utils/CEC2020_powersystems/'
        # G = np.loadtxt(f'{INPUT_DATA}/FunctionPS11_G.txt')
        # B = np.loadtxt(f'{INPUT_DATA}/FunctionPS11_B.txt')
        # P = np.loadtxt(f'{INPUT_DATA}/FunctionPS11_P.txt')
        # Q = np.loadtxt(f'{INPUT_DATA}/FunctionPS11_Q.txt')
        script_dir = Path(__file__).parent
        path = script_dir / "input_data" / "FunctionPS11_G.txt"
        G = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS11_B.txt"
        B = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS11_P.txt"
        P = np.loadtxt(path)
        path = script_dir / "input_data" / "FunctionPS11_Q.txt"
        Q = np.loadtxt(path)

        # Complex admittance matrix
        Y = G + 1j * B
        n_samples = X.shape[0]

        # Initialize voltages (30 buses)
        V = np.zeros((n_samples, 30), dtype=complex)
        V[:, 0] = 1  # Slack bus

        # Initialize power vectors
        Pg = np.zeros((n_samples, 30))
        Qg = np.zeros((n_samples, 30))
        Psp = np.zeros((n_samples, 30))
        Qsp = np.zeros((n_samples, 30))

        # Assign variables from decision vector X
        V[:, 1:30] = X[:, :29] + 1j * X[:, 29:58]  # Bus voltages
        Psp[:, 1:30] = X[:, 58:87]  # Specified active power
        Qsp[:, 1:30] = X[:, 87:116]  # Specified reactive power
        Pg[:, [1, 12, 21, 22, 26]] = X[
            :, 116:121
        ]  # Generator active power (0-based indexing)
        Qg[:, [1, 12, 21, 22, 26]] = X[:, 121:126]  # Generator reactive power

        # Calculate currents
        I = V @ Y.T
        Ir = np.real(I)
        Im = np.imag(I)

        # Power injections
        spI = np.conj((Psp + 1j * Qsp) / V)
        spIr = np.real(spI)
        spIm = np.imag(spI)

        # Mismatches
        delP = Psp - Pg + P
        delQ = Qsp - Qg + Q
        delIr = Ir - spIr
        delIm = Im - spIm

        # Calculate slack bus power
        Pg[:, 0] = np.real(V[:, 0] * np.conj(I[:, 0]))

        # Calculate objective based on problem
        # if prob_k == 37:  # Minimize active power loss
        # if '37' in self.flag:
        f = np.real(V[:, 0] * np.conj(I[:, 0])) + np.sum(Psp[:, 1:30], axis=1)
        if "penalty_constrained" in self.flag:
            f = abs(f)
        FACTOR = 25

        # Equality constraints
        h = np.concatenate(
            [delIr[:, 1:30], delIm[:, 1:30], delP[:, 1:30], delQ[:, 1:30]], axis=1
        )

        if self.is_constrained:
            if "penalty_constrained" in self.flag:
                return (
                    None,
                    None,
                    -(
                        torch.from_numpy(f)
                        + torch.from_numpy((np.sum(abs(h), axis=1) - 1e-4) / FACTOR)
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
