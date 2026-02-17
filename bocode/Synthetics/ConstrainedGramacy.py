from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class ConstrainedGramacy(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/camel3.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 2

    def __init__(self):
        tags = [
            "ConstrainedGramacy",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 2",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=2,
            bounds=[(0, 1)] * 2,
            optimum=[[-0.5998]],
            x_opt=[[0.1954, 0.4044]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            ConstrainedGramacy as ConstrainedGramacy_imported,
        )

        fun = ConstrainedGramacy_imported(negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)
