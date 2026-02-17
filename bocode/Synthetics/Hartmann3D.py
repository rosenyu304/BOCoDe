from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Hartmann3D(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/hart3.html
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Hartmann",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 3-Dim",
            "IMPORTS: None",
        ]

        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 1)] * 3,
            optimum=[[-3.86278]],
            x_opt=[[0.114614, 0.555649, 0.852547]],
            tags=tags,
        )

    def hart3(self, X):
        # Parameters
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        A = torch.tensor([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = 1e-4 * torch.tensor(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ],
            dtype=torch.float32,
        )

        outer = 0

        for ii in range(4):
            inner = 0

            for jj in range(3):
                xj = X[:, jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]

                # Compute the inner sum
                inner += Aij * (xj - Pij) ** 2

            # Update the outer sum
            new = alpha[ii] * torch.exp(-inner)
            outer += new

        # Return the negative of the outer sum
        y = -outer
        return y

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        return None, self.hart3(X).to(dtype=torch.float32).unsqueeze(-1)
