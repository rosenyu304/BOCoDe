"""MW3 constrained multi-objective synthetic test function (2 objectives, 2 constraints).

Ported from the paper (formulas verified numerically against pymoo 0.6.2):
Z. Ma and Y. Wang. Evolutionary Constrained Multiobjective Optimization: Test Suite
Construction and Performance Comparisons. IEEE TEVC, 23(6):972-986, 2019.

Uses the variable-linkage distance function g3. Bounds [0, 1]^dim. Objectives negated to
maximize (BoCoDe convention); constraints are feasible when <= 0.
"""

import torch

from ...base import BenchmarkProblem
from .MW1 import _la1


class MW3(BenchmarkProblem):
    """MW3 (2 objectives, 2 constraints; feasible <= 0), negated to maximize."""

    available_dimensions = (2, 100)
    num_objectives = 2
    num_constraints = 2

    def __init__(self, dim: int | None = None) -> None:
        # Default dim=2 (not the paper's 15): low dim keeps the feasible region findable
        # (~22.7% at d=2 vs ~0% at d>=10), matching how BoCoDe runs MW7 at dim=2.
        # Pass dim explicitly for the paper-standard 15.
        d = dim if dim is not None else 2
        super().__init__(
            dim=d,
            num_objectives=2,
            num_constraints=2,
            bounds=[(0.0, 1.0)] * d,
            ref_point=[-1.2, -1.2],
        )

    def _g3(self, X: torch.Tensor) -> torch.Tensor:
        m = self.num_objectives
        contrib = 2.0 * (X[:, m - 1 :] + (X[:, m - 2 : -1] - 0.5) ** 2 - 1.0) ** 2
        return 1.0 + contrib.sum(dim=1)

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        X = X.to(torch.double)
        g = self._g3(X)
        f0 = X[:, 0]
        f1 = g * (1.0 - f0 / g)
        theta = (2.0**0.5) * f1 - (2.0**0.5) * f0
        g0 = f0 + f1 - 1.05 - _la1(0.45, 0.75, 1.0, 6.0, theta)
        g1 = 0.85 - f0 - f1 + _la1(0.3, 0.75, 1.0, 2.0, theta)
        cons = torch.stack([g0, g1], dim=-1)  # <= 0 feasible
        obj = torch.stack([-f0, -f1], dim=-1)  # maximize
        return cons, obj
