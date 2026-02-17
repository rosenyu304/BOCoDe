import math
from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Langermann(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/langermann.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Langermann",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2D",
            "IMPORTS: torch, math",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            optimum=[[None]],
            x_opt=[[None]],
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            tags=tags,
        )

        # Constants for the function
        self.m = 5
        self.c = torch.tensor([1.0, 2.0, 5.0, 2.0, 3.0])
        self.A = torch.tensor(
            [[3.0, 5.0], [5.0, 2.0], [2.0, 1.0], [1.0, 4.0], [7.0, 9.0]]
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = X[..., 0], X[..., 1]
        x = torch.stack([x1, x2], dim=-1)  # (..., 2)
        # Compute squared distance to each A_i row
        dist_sq = torch.sum((x.unsqueeze(-2) - self.A) ** 2, dim=-1)  # (..., m)
        term = torch.exp(-dist_sq / math.pi) * torch.cos(math.pi * dist_sq)
        fx = torch.sum(self.c * term, dim=-1)
        return None, -fx.unsqueeze(-1)  # Negate for maximization
