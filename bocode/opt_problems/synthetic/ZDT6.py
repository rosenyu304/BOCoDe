"""ZDT6 multi-objective synthetic test function (10-D, 2 objectives; biased, non-uniform).

Ported from the paper (formulas verified numerically against pymoo 0.6.1.3):
E. Zitzler, K. Deb, and L. Thiele. Comparison of Multiobjective Evolutionary
Algorithms: Empirical Results. Evolutionary Computation, 8(2):173-195, 2000.
https://doi.org/10.1162/106365600568202

    f1 = 1 - exp(-4 x1) sin^6(6 pi x1)
    g  = 1 + 9 (sum_{i=2}^{n} x_i / (n - 1))^0.25
    f2 = g (1 - (f1 / g)^2)

x in [0, 1]^n. Two difficulties: the Pareto-optimal solutions are non-uniformly spread
along a non-convex front (density thins towards f1 = 1), and the density of solutions
is lowest near the front (the 0.25 power biases ``g`` away from its optimum g = 1 at
x2..xn = 0). Objectives negated to maximize; reference point [-11, -11] follows BoTorch's
ZDT convention.
"""

import torch

from ...base import BenchmarkProblem


class ZDT6(BenchmarkProblem):
    """ZDT6 (10-D inputs, 2 objectives; non-uniform concave front), negated to maximize."""

    available_dimensions = (2, 100)
    num_objectives = 2
    num_constraints = 0

    def __init__(self, dim: int | None = None) -> None:
        d = dim if dim is not None else 10  # paper-standard n = 10
        super().__init__(
            dim=d,
            num_objectives=2,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * d,
            ref_point=[-11.0, -11.0],
        )

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        X = X.to(torch.double)
        x1 = X[:, 0]
        f1 = 1.0 - torch.exp(-4.0 * x1) * torch.sin(6.0 * torch.pi * x1) ** 6
        g = 1.0 + 9.0 * (X[:, 1:].sum(dim=1) / (self.dim - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return None, torch.stack([-f1, -f2], dim=-1)
