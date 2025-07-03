from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Powell(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/powell.html
    """

    available_dimensions = (4, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, dim: int = 4):
        tags = [
            "Powell",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: N-Dim (at least 4)",
            "IMPORTS: BoTorch",
        ]

        if dim < 4:
            raise ValueError("Powell function is only defined for n >= 4")

        super().__init__(
            dim,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-4, 5)] * dim,
            optimum=[[0]],
            x_opt=[[0] * dim],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from botorch.test_functions.synthetic import Powell as Powell_imported

        fun = Powell_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(1)
