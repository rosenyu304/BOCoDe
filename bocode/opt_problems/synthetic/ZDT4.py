"""ZDT4 multi-objective synthetic test function (10-D, 2 objectives; multimodal).

Ported from the paper (formulas verified numerically against pymoo 0.6.1.3):
E. Zitzler, K. Deb, and L. Thiele. Comparison of Multiobjective Evolutionary
Algorithms: Empirical Results. Evolutionary Computation, 8(2):173-195, 2000.
https://doi.org/10.1162/106365600568202

    f1 = x1
    g  = 1 + 10 (n - 1) + sum_{i=2}^{n} (x_i^2 - 10 cos(4 pi x_i))
    f2 = g (1 - sqrt(f1 / g))

x1 in [0, 1], x2..xn in [-5, 5]. The Rastrigin-type ``g`` has 21^(n-1) local Pareto
fronts (about 8e11 at n = 10), so a method that converges to the nearest front stalls
far from the global one (g = 1 at x2..xn = 0, front f2 = 1 - sqrt(f1)). Objectives are
negated to maximize (BoCoDe convention); the reference point [-11, -11] is BoTorch's ZDT
convention (``ZDT._ref_point = [11, 11]`` in the minimization frame).
Uniformly random points have g of about 90 and do not reach it: the hypervolume stays
0 until an algorithm escapes the worst local fronts (g < 11).
"""

import torch

from ...base import BenchmarkProblem


class ZDT4(BenchmarkProblem):
    """ZDT4 (10-D inputs, 2 objectives; 21^(n-1) local fronts), negated to maximize."""

    available_dimensions = (2, 100)
    num_objectives = 2
    num_constraints = 0

    def __init__(self, dim: int | None = None) -> None:
        d = dim if dim is not None else 10  # paper-standard n = 10
        super().__init__(
            dim=d,
            num_objectives=2,
            num_constraints=0,
            bounds=[(0.0, 1.0)] + [(-5.0, 5.0)] * (d - 1),
            ref_point=[-11.0, -11.0],
        )

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        X = X.to(torch.double)
        f1 = X[:, 0]
        rest = X[:, 1:]
        g = (
            1.0
            + 10.0 * (self.dim - 1)
            + (rest**2 - 10.0 * torch.cos(4.0 * torch.pi * rest)).sum(dim=1)
        )
        f2 = g * (1.0 - torch.sqrt(f1 / g))
        return None, torch.stack([-f1, -f2], dim=-1)
