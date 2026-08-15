"""Thin wrapper around BoTorch multi-objective test problems.

BoTorch's multi-objective test problems are written for **minimization** and
publish their hypervolume reference point (``_ref_point``) in that same
minimization frame. BoCoDe **maximizes**, so the BoTorch problem is constructed
with ``negate=True``: BoTorch then negates both the objectives *and* the
reference point, and the negated ``ref_point`` buffer can be used verbatim. This
matches what :mod:`bocode.opt_problems.synthetic._wrapper` already does for the
BoTorch synthetic multi-objective functions.

Constraints are handled explicitly too: BoTorch reports constraint *slack* that
is feasible when ``>= 0``, whereas BoCoDe's convention is feasible when ``<= 0``,
so the slack is negated. ``negate`` only affects the objectives in BoTorch, never
the slack.

Sources:
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
Reference points: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
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
            self.botorch_problem = botorch_problem(negate=True)
            dim = self.botorch_problem.dim
        else:
            self.botorch_problem = botorch_problem(dim=dim, negate=True)
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
            # Already in the maximization frame: BoTorch negates the reference
            # point together with the objectives when negate=True.
            ref_point=self.botorch_problem.ref_point.tolist(),
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling: bool = False
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        if self.num_constraints != 0:
            # BoTorch slack >= 0 is feasible; BoCoDe's convention is c <= 0.
            slack = self.botorch_problem.evaluate_slack_true(X)
            return -slack, self.botorch_problem(X)

        return None, self.botorch_problem(X)
