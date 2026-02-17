from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class ThreeHumpCamel(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/camel3.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "ThreeHumpCamel",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-5, 5)] * 2,
            optimum=[[0]],
            x_opt=[[0, 0]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            ThreeHumpCamel as ThreeHumpCamel_imported,
        )

        fun = ThreeHumpCamel_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)
