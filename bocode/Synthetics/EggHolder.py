from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class EggHolder(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/egg.html
    and
    BoTorch: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "EggHolder",
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
            bounds=[(-512, 512)] * 2,
            optimum=[[959.6407]],
            x_opt=[[512, 404.2319]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import EggHolder as EggHolder_imported

        fun = EggHolder_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)
