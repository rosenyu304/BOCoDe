from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Ackley(BenchmarkProblem):
    """
    Sources:
    (1) https://www.sfu.ca/~ssurjano/ackley.html
    (2) Eriksson D, Poloczek M (2021) Scalable constrained bayesian optimization.
    In: International Conference on Artificial Intelligence and Statistics, PMLR, pp 730–738
    """

    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 2

    def __init__(self, dim: int = 2):
        tags = [
            "Ackley",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: Constrained (2)",
            "SPACE: Continuous",
            "SCALABLE: N-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim,
            num_objectives=1,
            num_constraints=2,
            optimum=[[0]],
            x_opt=[[0] * dim],
            bounds=[(-5, 10)] * dim,
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from botorch.test_functions import Ackley as Ackley_imported

        n = X.size(0)

        gx = torch.zeros((n, self.num_constraints))

        fun = Ackley_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        gx[:, 0] = torch.sum(X, 1)
        gx[:, 1] = torch.norm(X, p=2, dim=1) - 5

        return gx, fun(X).unsqueeze(1)
