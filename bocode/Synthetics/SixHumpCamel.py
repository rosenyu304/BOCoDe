from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class SixHumpCamel(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/camel6.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "SixHumpCamel",
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
            bounds=[(-3, 3)] * 2,
            optimum=[[1.0316], [1.0316]],
            x_opt=[[0.0898, -0.7126], [-0.0898, 0.7126]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            SixHumpCamel as SixHumpCamel_imported,
        )

        fun = SixHumpCamel_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)
