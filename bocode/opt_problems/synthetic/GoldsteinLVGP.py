"""Goldstein-Price with one qualitative factor (1 continuous + 1 categorical, K=5).

The mixed quantitative/qualitative Goldstein-Price benchmark used in the LVGP
paper. The second input ``x2`` is discretised to the five levels
``{-2, -1, 0, 1, 2}`` and presented as an *unordered* categorical. The per-level
optimum spread is enormous (3 at the best level versus ~7.7e4 at the worst), so
the qualitative choice is highly consequential and non-separable from the
continuous search.

Objective (minimize), the standard Goldstein-Price form::

    f(x1, x2) = [1 + (x1 + x2 + 1)^2 (19 - 14 x1 + 3 x1^2 - 14 x2 + 6 x1 x2 + 3 x2^2)]
              * [30 + (2 x1 - 3 x2)^2 (18 - 32 x1 + 12 x1^2 + 48 x2 - 36 x1 x2 + 27 x2^2)]

Variables: ``x1`` continuous in ``[-2, 2]``; ``x2`` categorical in
``{-2, -1, 0, 1, 2}``.

Reference optimum ``f* = 3.0`` (minimization sense) at ``x1 = 0``, ``x2 = -1``.
``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
Y. Zhang, D. W. Apley, W. Chen. Bayesian Optimization for Materials Design with Mixed Quantitative and Qualitative Variables. Scientific Reports 10:4924, 2020 (arXiv:1910.01688).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem


class GoldsteinLVGP(BenchmarkProblem):
    """Goldstein-Price with x2 recast as a 5-level unordered categorical (2-D, mixed)."""

    available_dimensions = 2
    num_objectives = 1
    num_constraints = 0

    #: the five qualitative levels of x2 (their physical values)
    LEVELS = [-2.0, -1.0, 0.0, 1.0, 2.0]

    def __init__(self) -> None:
        self.variable_types = ["continuous", list(self.LEVELS)]
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            x_opt=[[0.0, -1.0]],
            optimum=[3.0],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x1, x2 = X[:, 0], X[:, 1]
        term1 = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        f = term1 * term2
        fx = (-f).reshape(-1, 1).to(torch.float64)  # maximize -f
        return None, fx
