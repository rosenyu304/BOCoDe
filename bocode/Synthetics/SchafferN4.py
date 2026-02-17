from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class SchafferN4(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/schaffer4.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "SchafferN4",
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
            bounds=[(-100.0, 100.0), (-100.0, 100.0)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = X[..., 0], X[..., 1]
        numerator = torch.cos(torch.sin(torch.abs(x1**2 - x2**2))) ** 2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        fx = 0.5 + numerator / denominator
        return None, -fx.unsqueeze(-1)  # Negate for maximization
