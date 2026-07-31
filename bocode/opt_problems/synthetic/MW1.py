"""MW1 constrained multi-objective synthetic test function (2 objectives, 1 constraint).

Ported from the paper (formulas verified numerically against pymoo 0.6.2):
Z. Ma and Y. Wang. Evolutionary Constrained Multiobjective Optimization: Test Suite
Construction and Performance Comparisons. IEEE TEVC, 23(6):972-986, 2019.

Uses the biased distance function g1 (requires dim >= 3, since g1's variable exponent is
n = dim - n_obj). Bounds [0, 1]^dim. Objectives negated to maximize (BoCoDe convention);
constraint is feasible when <= 0.
"""

import torch

from ...base import BenchmarkProblem


def _safe_pow(base: torch.Tensor, exp: float) -> torch.Tensor:
    """np.power semantics for our exponents: odd C keeps sign, even D drops it."""
    if int(exp) % 2 == 1:
        return torch.sign(base) * base.abs() ** exp
    return base.abs() ** exp


def _la1(A: float, B: float, C: float, D: float, theta: torch.Tensor) -> torch.Tensor:
    """A * sin(B*pi*theta^C)^D  (MW local-adjustment term LA1)."""
    return A * _safe_pow(torch.sin(B * torch.pi * _safe_pow(theta, C)), D)


class MW1(BenchmarkProblem):
    """MW1 (2 objectives, 1 constraint; feasible <= 0), negated to maximize."""

    available_dimensions = (3, 100)
    num_objectives = 2
    num_constraints = 1

    def __init__(self, dim: int | None = None) -> None:
        # Default dim=3 (not the paper's 15): g1 needs dim>=3, and low dim keeps the
        # feasible region findable (~3.4% at d=3 vs ~0% at d>=6), matching how BoCoDe
        # runs MW7 at dim=2. Pass dim explicitly for the paper-standard 15.
        d = dim if dim is not None else 3
        super().__init__(
            dim=d,
            num_objectives=2,
            num_constraints=1,
            bounds=[(0.0, 1.0)] * d,
            ref_point=[-1.2, -1.2],
        )

    def _g1(self, X: torch.Tensor) -> torch.Tensor:
        d = self.dim
        n = d - self.num_objectives
        z = X[:, self.num_objectives - 1:] ** n
        i = torch.arange(self.num_objectives - 1, d, dtype=X.dtype, device=X.device)
        return 1.0 + (1.0 - torch.exp(-10.0 * (z - 0.5 - i / (2 * d)) ** 2)).sum(dim=1)

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        X = X.to(torch.double)
        g = self._g1(X)
        f0 = X[:, 0]
        f1 = g * (1.0 - 0.85 * f0 / g)
        theta = (2.0 ** 0.5) * f1 - (2.0 ** 0.5) * f0
        g0 = f0 + f1 - 1.0 - _la1(0.5, 2.0, 1.0, 8.0, theta)
        cons = g0.unsqueeze(-1)                       # <= 0 feasible
        obj = torch.stack([-f0, -f1], dim=-1)         # maximize
        return cons, obj
