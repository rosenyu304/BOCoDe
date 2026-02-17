from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Hartmann6D(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/hart6.html
    """

    available_dimensions = 6
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Hartmann",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 6-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=6,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 1)] * 6,
            optimum=[[3.32237]],
            x_opt=[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import Hartmann as Hartmann_imported

        fun = Hartmann_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)
