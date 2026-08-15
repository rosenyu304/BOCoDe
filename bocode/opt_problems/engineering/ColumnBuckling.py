"""Column buckling — Euler critical load of a column, mixed-variable.

The Euler critical buckling load of a slender column, a mixed continuous/categorical
engineering problem from the GP+ project (Bostanabad Research Group). The column
length is continuous; the material modulus, the effective-length factor, and the
section moment of inertia are categorical (a small catalog of standard choices).

Variables (order): ``L`` length [0.5, 1.5] (continuous), ``E`` Young's modulus
(categorical: 73.1, 200.0 GPa), ``K`` effective-length factor (categorical: 0.5,
0.7, 1.0, 2.0), ``I`` moment of inertia (categorical: 9.49, 12.1, 29.5).

Objective: the Euler critical load ``P = pi*E*I / (L*K)^2``. Unconstrained.

OPTIMIZATION DIRECTION: **the critical load is MAXIMIZED**, so the value is returned
as-is (BoCoDe maximizes). As with Borehole, the source does not settle this: GP+ uses
column buckling purely as a (multi-fidelity) *regression / emulation* testbed -- its
repository ships the problem as a dataset for GP fitting, with no optimization direction
attached. We keep MAXIMIZE because the quantity is a structural *capacity*: the critical
load is the load at which the column fails, so a larger value is a better column, and
every design variable here (material, section, end condition, length) is a design choice
that a designer makes to raise it. "Minimizing the buckling load" would mean searching
for the weakest possible column, which is not a design problem anyone poses.

TWO CAVEATS, both reported rather than silently changed:
1. ``P`` is monotone in all four variables (increasing in E and I, decreasing in L and
   K), so the optimum is a box CORNER (E=200, I=29.5, K=0.5, L=0.5) and this is an easy
   benchmark. This is a property of the problem, not a bug.
2. The textbook Euler formula is ``P = pi**2 * E * I / (K*L)**2``; this implementation
   (and GP+'s, as far as we could reconstruct it) uses ``pi``, not ``pi**2``. That is a
   constant positive factor, so it changes neither the argmax nor the ranking of any two
   designs -- only the reported magnitude. It is left as-is because we could not verify
   the exact GP+ expression (the current GP+ release no longer ships the analytic
   function), and "fixing" it would invalidate existing results for zero optimization
   benefit.

Sources:
A. Yousefpour, Z. Zanjani Foumani, M. Shishehbor, C. Mora, R. Bostanabad. GP+: a Python library for kernel-based learning via Gaussian processes. Advances in Engineering Software, 2024. https://github.com/Bostanabad-Research-Group/GP-Plus
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem

_E_LEVELS = [73.1, 200.0]
_K_LEVELS = [0.5, 0.7, 1.0, 2.0]
_I_LEVELS = [9.49, 12.1, 29.5]


class ColumnBuckling(BenchmarkProblem):
    """Maximize a column's Euler buckling load (4 vars; L continuous, E/K/I categorical).

    The load is a structural capacity, so it is maximized and returned unnegated; see
    the module docstring for why (the GP+ source uses this only for regression).
    ``is_discrete=False`` relaxes E, K, I to continuous ranges.
    """

    available_dimensions = 4
    num_objectives = 1
    num_constraints = 0

    def __init__(self, is_discrete: bool = True) -> None:
        bounds = [
            (0.5, 1.5),
            (min(_E_LEVELS), max(_E_LEVELS)),
            (min(_K_LEVELS), max(_K_LEVELS)),
            (min(_I_LEVELS), max(_I_LEVELS)),
        ]
        if is_discrete:
            self.variable_types = ["continuous", _E_LEVELS, _K_LEVELS, _I_LEVELS]
        else:
            self.variable_types = None
        super().__init__(dim=4, num_objectives=1, num_constraints=0, bounds=bounds)

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        L, E, K, I = x[:, 0], x[:, 1], x[:, 2], x[:, 3]  # noqa: E741

        P = torch.pi * E * I / (L * K) ** 2
        fx = torch.tensor(P, dtype=torch.float64).reshape(-1, 1)  # maximize load
        return None, fx
