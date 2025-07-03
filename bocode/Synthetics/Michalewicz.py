from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Michalewicz(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/michal.html
    """

    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, dim: int = 2):
        tags = [
            "Michalewicz",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: N-Dim",
            "IMPORTS: BoTorch",
        ]

        import math

        opt = {2: [1.8013], 5: [4.687658], 10: [9.66015]}
        optimum = opt.get(dim)

        x_opts = {2: [[2.202905, 1.570796]]}
        x_opt = x_opts.get(dim)

        super().__init__(
            dim,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, math.pi)] * dim,
            optimum=optimum,
            x_opt=x_opt,
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from botorch.test_functions.synthetic import Michalewicz as Michalewicz_imported

        fun = Michalewicz_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(1)
