from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class HolderTable(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/holder.html
    and
    BoTorch: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "HolderTable",
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
            bounds=[(-10, 10)] * 2,
            optimum=[[19.2085]],
            x_opt=[[8.05502, 9.66459]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import HolderTable as HolderTable_imported

        fun = HolderTable_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)
