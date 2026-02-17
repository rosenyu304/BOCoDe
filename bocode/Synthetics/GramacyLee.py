import math
from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class GramacyLee(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/grlee12.html
    """

    available_dimensions = 1
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "GramacyLee",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 1D",
            "IMPORTS: torch, math",
        ]

        super().__init__(
            dim=1,
            num_objectives=1,
            num_constraints=0,
            optimum=[[None]],
            x_opt=[[None]],
            bounds=[(0.5, 2.5)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = X[..., 0]
        fx = torch.sin(10 * math.pi * x) / (2 * x) + (x - 1) ** 4
        return None, -fx.unsqueeze(-1)  # Negate for maximization
