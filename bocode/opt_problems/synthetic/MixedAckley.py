"""Ackley with a categorical scale factor (4 continuous + 1 categorical, K=4).

The 4-D Ackley function whose continuous inputs are multiplied by a scale chosen
by an unordered categorical, so the level controls how rugged the landscape is
around the optimum.

Objective (minimize), with ``z = scale[c] * x`` and ``n = 4``::

    f(x, c) = -20 exp(-0.2 sqrt(sum z_i^2 / n)) - exp(sum cos(2 pi z_i) / n) + 20 + e

Variables: ``x0..x3`` continuous in ``[-32.768, 32.768]``; ``c`` categorical with
K=4 levels selecting ``scale`` in ``{1, 2, 4, 8}``. The categorical is nominal, so
the decision variable carries the level *index*.

Reference optimum ``f* = 0`` (minimization sense) at ``x = 0`` — reachable for
*any* level, since the scale only multiplies ``x``. The categorical therefore
changes the difficulty of the search, not the optimal value. ``evaluate()``
returns ``-f`` because BoCoDe maximizes.

Sources:
D. H. Ackley. A Connectionist Machine for Genetic Hillclimbing. Kluwer Academic Publishers, 1987 (the Ackley function).
Mixed-variable scale-categorical construction after the CoCaBO-style synthetic suites; see B. Ru et al., ICML 2020.
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem
from ._common import cat_index


class MixedAckley(BenchmarkProblem):
    """4-D Ackley with a 4-level nominal categorical scale factor (5-D, mixed)."""

    available_dimensions = 5
    num_objectives = 1
    num_constraints = 0

    N_CONT = 4
    #: scale factor selected by each level of the categorical
    SCALES = [1.0, 2.0, 4.0, 8.0]

    def __init__(self) -> None:
        self.variable_types = ["continuous"] * self.N_CONT + [[0.0, 1.0, 2.0, 3.0]]
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-32.768, 32.768)] * self.N_CONT + [(0.0, 3.0)],
            x_opt=[[0.0, 0.0, 0.0, 0.0, 0.0]],
            optimum=[0.0],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        cont = x[:, : self.N_CONT]
        c = cat_index(x[:, self.N_CONT], len(self.SCALES))
        scales = torch.tensor(self.SCALES, dtype=torch.float64)
        z = cont * scales[c].unsqueeze(1)

        n = self.N_CONT
        sum_sq = (z**2).sum(dim=1)
        sum_cos = torch.cos(2 * math.pi * z).sum(dim=1)
        f = (
            -20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq / n))
            - torch.exp(sum_cos / n)
            + 20.0
            + math.e
        )

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
