from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class BotorchProblem(BenchmarkProblem):
    """
    Sources:
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020.
    http://arxiv.org/abs/1910.06403
    """

    input_type = DataType.CONTINUOUS

    def __init__(self, botorch_problem, optimum=None, x_opt=None, dim=None):
        if dim is None:
            self.botorch_problem = botorch_problem()
            dim = self.botorch_problem.dim
            self.fixedDim = True
        else:
            self.botorch_problem = botorch_problem(dim=dim)
            self.fixedDim = False
        bounds = list(zip(*self.botorch_problem.bounds.numpy()))
        num_obj = self.botorch_problem.num_objectives
        num_cons = (
            self.botorch_problem.num_constraints
            if hasattr(self.botorch_problem, "num_constraints")
            else 0
        )

        super().__init__(
            dim=dim,
            num_objectives=num_obj,
            num_constraints=num_cons,
            bounds=bounds,
            x_opt=x_opt,
            optimum=optimum,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        # if self.fixedDim:
        if self.num_constraints != 0:
            return self.botorch_problem.evaluate_slack_true(X), self.botorch_problem(
                X
            ).unsqueeze(-1)

        return None, self.botorch_problem(X).unsqueeze(-1)


class MultiObjBotorchProblem(BenchmarkProblem):
    input_type = DataType.CONTINUOUS

    def __init__(self, botorch_problem, optimum=None, x_opt=None, dim=None):
        if dim is None:
            self.botorch_problem = botorch_problem()
            dim = self.botorch_problem.dim
            self.fixedDim = True
        else:
            self.botorch_problem = botorch_problem(dim=dim)
            self.fixedDim = False
        bounds = list(zip(*self.botorch_problem.bounds.numpy()))
        num_obj = self.botorch_problem.num_objectives
        num_cons = (
            self.botorch_problem.num_constraints
            if hasattr(self.botorch_problem, "num_constraints")
            else 0
        )

        super().__init__(
            dim=dim,
            num_objectives=num_obj,
            num_constraints=num_cons,
            bounds=bounds,
            x_opt=x_opt,
            optimum=optimum,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        # if self.fixedDim:
        if self.num_constraints != 0:
            return self.botorch_problem.evaluate_slack_true(X), self.botorch_problem(X)

        return None, self.botorch_problem(X)
