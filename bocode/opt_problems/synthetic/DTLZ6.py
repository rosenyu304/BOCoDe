"""DTLZ6 multi-objective synthetic test function (12-D, 3 objectives; degenerate front).

Ported from the paper (formulas verified numerically against pymoo 0.6.1.3):
K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. Scalable Test Problems for
Evolutionary Multiobjective Optimization. In Evolutionary Multiobjective Optimization,
Springer, 2005, pp. 105-145. https://doi.org/10.1007/1-84628-137-7_6

DTLZ6 is DTLZ5 (a degenerate, curve-shaped Pareto front in M-dimensional objective
space) with the much harder distance function ``g = sum_{i in X_M} x_i^0.1``. The 0.1
power makes ``g`` almost flat away from its optimum ``x_M = 0`` and very steep near it,
so the true front is approached only from a thin sliver of the box and most
algorithms converge to a surface *above* the curve instead of to it:

    theta_1 = x_1 * pi / 2
    theta_i = pi / (4 (1 + g)) * (1 + 2 g x_i)              i = 2..M-1
    f_1 = (1 + g) cos(theta_1) ... cos(theta_{M-1})
    f_m = (1 + g) cos(theta_1) ... cos(theta_{M-m}) sin(theta_{M-m+1})
    f_M = (1 + g) sin(theta_1)

x in [0, 1]^n, k = n - M + 1 distance variables (paper standard k = 10, so n = 12 for
M = 3). Objectives negated to maximize. Reference point [-10]*M is BoTorch's convention
for DTLZ5, the structurally identical problem (``DTLZ5._ref_val = 10``).
"""

import math

import torch

from ...base import BenchmarkProblem


def dtlz_sphere(theta: torch.Tensor, g: torch.Tensor, m: int) -> torch.Tensor:
    """DTLZ2-family spherical mapping: ``theta`` holds angles already scaled to [0, pi/2]."""
    cos, sin = torch.cos(theta), torch.sin(theta)
    fs = []
    for i in range(m):
        f = (1.0 + g) * torch.prod(cos[:, : m - 1 - i], dim=1)
        if i > 0:
            f = f * sin[:, m - 1 - i]
        fs.append(f)
    return torch.stack(fs, dim=-1)


class DTLZ6(BenchmarkProblem):
    """DTLZ6 (12-D inputs, 3 objectives; degenerate curve front), negated to maximize."""

    available_dimensions = (3, 100)
    num_objectives = 3
    num_constraints = 0

    def __init__(
        self, dim: int | None = None, num_objectives: int | None = None
    ) -> None:
        m = num_objectives if num_objectives is not None else 3
        d = dim if dim is not None else m - 1 + 10
        if d < m:
            raise ValueError(f"DTLZ6 needs dim >= num_objectives, got dim={d}, M={m}")
        super().__init__(
            dim=d,
            num_objectives=m,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * d,
            ref_point=[-10.0] * m,
        )

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        X = X.to(torch.double)
        m = self.num_objectives
        g = (X[:, m - 1 :] ** 0.1).sum(dim=1)
        t = (1.0 + 2.0 * g.unsqueeze(-1) * X[:, : m - 1]) / (
            2.0 * (1.0 + g.unsqueeze(-1))
        )
        t = torch.cat([X[:, :1], t[:, 1:]], dim=1)
        f = dtlz_sphere(t * (math.pi / 2.0), g, m)
        return None, -f
