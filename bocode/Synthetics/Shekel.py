from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class BaseShekel(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/shekel.html
    """

    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, m: int, optimum):
        tags = [
            "Shekel",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 4-Dim",
            "IMPORTS: BoTorch",
        ]
        self.m = m
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 10)] * 4,
            optimum=optimum,
            x_opt=[[4.0, 4.0, 4.0, 4.0]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import Shekel as Shekel_imported

        fun = Shekel_imported(m=self.m, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class Shekelm5(BaseShekel):
    def __init__(self):
        super().__init__(m=5, optimum=[10.1532])


class Shekelm7(BaseShekel):
    def __init__(self):
        super().__init__(m=7, optimum=[10.4029])


class Shekelm10(BaseShekel):
    def __init__(self):
        super().__init__(m=10, optimum=[10.5364])


class Shekel(BaseShekel):
    def __init__(self, m=5, optimum=[10.1532]):
        super().__init__(m=m, optimum=optimum)
