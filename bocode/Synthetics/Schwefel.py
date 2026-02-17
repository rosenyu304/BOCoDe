from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Schwefel(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """

    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, dim: int = 2):
        tags = [
            "Schwefel",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: Arbitrary d",
            "IMPORTS: torch, math",
        ]

        super().__init__(
            dim=dim,
            num_objectives=1,
            num_constraints=0,
            optimum=[[0.0]],
            x_opt=[[420.9687] * dim],
            bounds=[(-500.0, 500.0)] * dim,
            tags=tags,
        )

        self.constant = 418.9829

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        absX = torch.abs(X)
        term = X * torch.sin(torch.sqrt(absX))
        fx = self.constant * X.shape[-1] - torch.sum(term, dim=-1)
        return None, -fx.unsqueeze(-1)  # Negate for maximization
