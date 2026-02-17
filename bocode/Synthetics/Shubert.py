from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Shubert(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/shubert.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Shubert",
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
            optimum=[[-186.7309]],  # global minimum value
            x_opt=[[None]],  # multiple minima, so we omit coordinates
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            tags=tags,
        )

        # constants for i = 1..5
        self.i_vals = torch.arange(1.0, 6.0)

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = X[..., 0], X[..., 1]

        # Compute sums along i for x1 and x2
        sum1 = torch.sum(
            self.i_vals * torch.cos((self.i_vals + 1) * x1.unsqueeze(-1) + self.i_vals),
            dim=-1,
        )
        sum2 = torch.sum(
            self.i_vals * torch.cos((self.i_vals + 1) * x2.unsqueeze(-1) + self.i_vals),
            dim=-1,
        )

        fx = sum1 * sum2
        return None, -fx.unsqueeze(-1)  # Negate for maximization
