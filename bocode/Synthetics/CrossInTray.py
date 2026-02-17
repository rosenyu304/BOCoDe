import math
from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class CrossInTray(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/crossit.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "CrossInTray",
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
            optimum=[[-2.06261]],
            x_opt=[
                [1.3491, 1.3491],
                [1.3491, -1.3491],
                [-1.3491, 1.3491],
                [-1.3491, -1.3491],
            ],
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = X[..., 0], X[..., 1]
        term = torch.abs(
            torch.sin(x1)
            * torch.sin(x2)
            * torch.exp(torch.abs(100 - torch.sqrt(x1**2 + x2**2) / math.pi))
        )
        fx = -0.0001 * (term + 1) ** 0.1
        return None, -fx.unsqueeze(-1)  # Negate for maximization
