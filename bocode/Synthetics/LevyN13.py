import math
from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class LevyN13(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/levy13.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "LevyN13",
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
            optimum=[[0.0]],
            x_opt=[[1.0, 1.0]],
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = X[..., 0], X[..., 1]

        term1 = torch.sin(3 * math.pi * x1) ** 2
        term2 = (x1 - 1) ** 2 * (1 + torch.sin(3 * math.pi * x2) ** 2)
        term3 = (x2 - 1) ** 2 * (1 + torch.sin(2 * math.pi * x2) ** 2)

        fx = term1 + term2 + term3
        return None, -fx.unsqueeze(-1)  # Negate for maximization
