from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class DixonPrice(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/dixonpr.html
    """

    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, dim: int = 2):
        tags = [
            "DixonPrice",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: N-Dim",
            "IMPORTS: BoTorch",
        ]

        x_opt = torch.tensor(
            [[2 ** (-(2**i - 2) / 2**i) for i in range(1, dim + 1)]],
            dtype=torch.float32,
        ).tolist()

        super().__init__(
            dim,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-10, 10)] * dim,
            optimum=[[0]],
            x_opt=x_opt,
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from botorch.test_functions.synthetic import DixonPrice as DixonPrice_imported

        fun = DixonPrice_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)
