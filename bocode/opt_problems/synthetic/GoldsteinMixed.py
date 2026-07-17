"""Goldstein-Price with one integer dimension (1 continuous + 1 integer).

The standard 2-D Goldstein-Price function with the second input restricted to the
integers ``{-2, -1, 0, 1, 2}``. This is the mixed-*integer* counterpart of
:mod:`~bocode.opt_problems.synthetic.GoldsteinLVGP`: the objective is
numerically identical, but ``x2`` is declared as an ordered integer variable
rather than an unordered categorical, so the two problems together isolate the
effect of the variable-type declaration on a solver.

Objective (minimize), the standard Goldstein-Price form::

    f(x1, x2) = [1 + (x1 + x2 + 1)^2 (19 - 14 x1 + 3 x1^2 - 14 x2 + 6 x1 x2 + 3 x2^2)]
              * [30 + (2 x1 - 3 x2)^2 (18 - 32 x1 + 12 x1^2 + 48 x2 - 36 x1 x2 + 27 x2^2)]

Variables: ``x1`` continuous in ``[-2, 2]``; ``x2`` integer in ``[-2, 2]``.

Reference optimum ``f* = 3.0`` (minimization sense) at ``x1 = 0``, ``x2 = -1``.
``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
A. A. Goldstein, J. F. Price. On descent from local minima. Mathematics of Computation 25(115):569-574, 1971.
J. Qian. Discretised mixed-variable benchmark suite, MIT MEng thesis (Fig. 3.2 benchmarks).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem


class GoldsteinMixed(BenchmarkProblem):
    """Goldstein-Price with an integer second dimension (2-D, mixed)."""

    available_dimensions = 2
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        self.variable_types = ["continuous", "integer"]
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            x_opt=[[0.0, -1.0]],
            optimum=[3.0],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        x1, x2 = x[:, 0], x[:, 1]
        term1 = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        f = term1 * term2
        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
