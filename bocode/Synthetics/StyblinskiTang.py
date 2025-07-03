from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class StyblinskiTang(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/stybtang.html
    """

    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, dim: int = 10):
        tags = [
            "StyblinskiTang",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: N-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=dim,
            num_objectives=1,
            num_constraints=0,
            optimum=[[39.166166 * dim]],
            x_opt=[[-2.903534] * dim],
            bounds=[(-5, 5)] * dim,
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from botorch.test_functions.synthetic import (
            StyblinskiTang as StyblinskiTang_imported,
        )

        fun = StyblinskiTang_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(1)
