"""Styblinski-Tang with an additive categorical offset (4 continuous + 1 categorical, K=3).

The 4-D Styblinski-Tang function plus an unordered categorical "mode" that adds a
flat offset, so the three modes have distinct optimal values and the optimizer
must identify the best arm.

Objective (minimize)::

    f(x, c) = 0.5 * sum_i (x_i^4 - 16 x_i^2 + 5 x_i) + offset[c]

with ``offset = [0.0, 10.0, 25.0]``.

Variables: ``x0..x3`` continuous in ``[-5, 5]``; ``c`` categorical with K=3 levels
(nominal, so the decision variable carries the level *index*).

Reference optimum ``f* = -39.16616 * 4 = -156.66464`` (minimization sense) at
``x_i ~ -2.903534`` and the best mode ``c = 0``. ``evaluate()`` returns ``-f``
because BoCoDe maximizes.

Sources:
M. Styblinski, T.-S. Tang. Experiments in nonconvex optimization: Stochastic approximation with function smoothing and simulated annealing. Neural Networks 3(4):467-483, 1990.
Additive-offset categorical construction after the mixed-variable synthetic suites; see B. Ru et al., ICML 2020.
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem
from ._common import cat_index


class StyblinskiTangMixed(BenchmarkProblem):
    """4-D Styblinski-Tang plus a 3-level nominal categorical mode offset (5-D, mixed)."""

    available_dimensions = 5
    num_objectives = 1
    num_constraints = 0

    N_CONT = 4
    #: flat offset added by each level of the categorical
    OFFSETS = [0.0, 10.0, 25.0]

    def __init__(self) -> None:
        self.variable_types = ["continuous"] * self.N_CONT + [[0.0, 1.0, 2.0]]
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-5.0, 5.0)] * self.N_CONT + [(0.0, 2.0)],
            x_opt=[[-2.903534] * self.N_CONT + [0.0]],
            optimum=[-39.16616 * 4],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        cont = x[:, : self.N_CONT]
        c = cat_index(x[:, self.N_CONT], len(self.OFFSETS))
        offsets = torch.tensor(self.OFFSETS, dtype=torch.float64)

        f = 0.5 * (cont**4 - 16 * cont**2 + 5 * cont).sum(dim=1) + offsets[c]

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
