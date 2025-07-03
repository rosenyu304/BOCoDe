from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Bukin(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/bukin6.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Bukin",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2D",
            "IMPORTS: ",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            optimum=[[0]],
            x_opt=[[-10, 1]],
            bounds=[(-15.0, -5.0), (-3.0, 3.0)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        part1 = 100.0 * torch.sqrt(torch.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * torch.abs(X[..., 0] + 10.0)
        fx = -(part1 + part2)

        return None, fx.unsqueeze(-1)
