from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class SchafferN2(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/schaffer2.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "SchafferN2",
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
            x_opt=[[0.0, 0.0]],
            bounds=[(-100.0, 100.0), (-100.0, 100.0)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = X[..., 0], X[..., 1]
        numerator = torch.sin(x1**2 - x2**2) ** 2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        fx = 0.5 + numerator / denominator
        return None, -fx.unsqueeze(-1)  # Negate for maximization
