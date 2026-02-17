from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Cosine8(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/beale.html
    """

    available_dimensions = 8
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Cosine8",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 8-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=8,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1.0, 1.0)] * 8,
            optimum=[[-0.8]],
            x_opt=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import Cosine8 as Cosine8_imported

        fun = Cosine8_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)
