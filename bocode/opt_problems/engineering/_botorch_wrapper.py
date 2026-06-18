"""Thin wrapper around BoTorch multi-objective test problems.

Sources:
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
"""

import torch

from ...base import BenchmarkProblem


class MultiObjBotorchProblem(BenchmarkProblem):
    """Adapter exposing a BoTorch ``MultiObjectiveTestProblem`` as a BenchmarkProblem.

    Subclasses set ``available_dimensions``, ``num_objectives``, and
    ``num_constraints`` as plain class attributes and pass the BoTorch problem
    class to ``__init__``; the BoTorch problem is only instantiated at
    construction time.
    """

    def __init__(
        self,
        botorch_problem,
        optimum: torch.Tensor | None = None,
        x_opt: torch.Tensor | None = None,
        dim: int | None = None,
    ) -> None:
        if dim is None:
            self.botorch_problem = botorch_problem()
            dim = self.botorch_problem.dim
        else:
            self.botorch_problem = botorch_problem(dim=dim)
        bounds = list(zip(*self.botorch_problem.bounds.numpy(), strict=False))
        num_obj = self.botorch_problem.num_objectives
        num_cons = getattr(self.botorch_problem, "num_constraints", 0)

        super().__init__(
            dim=dim,
            num_objectives=num_obj,
            num_constraints=num_cons,
            bounds=bounds,
            x_opt=x_opt,
            optimum=optimum,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling: bool = False
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        if self.num_constraints != 0:
            return self.botorch_problem.evaluate_slack_true(X), self.botorch_problem(X)

        return None, self.botorch_problem(X)
