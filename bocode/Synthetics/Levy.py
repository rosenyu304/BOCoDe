from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Levy(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/levy.html
    """

    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, dim: int = 2):
        tags = [
            "Levy",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: N-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-10, 10)] * dim,
            optimum=[[0]],
            x_opt=[[1] * dim],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from botorch.test_functions.synthetic import Levy as Levy_imported

        fun = Levy_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(1)
