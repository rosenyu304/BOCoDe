"""Ackley-3 with two binary categorical sign flips (3 continuous + 2 categorical, K=2).

A 5-D mixed-variable Ackley: three continuous coordinates in ``[-1, 1]`` and two
binary categoricals that flip the sign of the first two coordinates.

Objective (minimize), with ``s_i = +1`` for level 0 and ``-1`` for level 1,
``z = (s_0 x_0, s_1 x_1, x_2)`` and ``n = 3``::

    f = -20 exp(-0.2 sqrt(sum z_i^2 / n)) - exp(sum cos(2 pi z_i) / n) + 20 + e

Variables: ``x0, x1, x2`` continuous in ``[-1, 1]``; ``cat0``, ``cat1``
categorical with K=2 levels each (nominal, so the decision variable carries the
level *index*).

Reference optimum ``f* = 0`` (minimization sense) at ``x = 0``, any levels.

.. note::
   **The categorical variables provably do not change the objective.** Ackley is
   *even* in each coordinate (it depends on ``z_i`` only through ``z_i^2`` and
   ``cos(2 pi z_i)``, both even functions), so flipping the sign of ``x_0`` or
   ``x_1`` leaves ``f`` exactly invariant. This problem is therefore effectively a
   3-D continuous Ackley with two inert categorical dimensions. That is a faithful
   port of the source definition, not a porting error — it is a useful *null*
   benchmark (does a mixed-variable solver waste budget on irrelevant categorical
   structure?) but it must not be read as a test of categorical modelling power.

``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
D. H. Ackley. A Connectionist Machine for Genetic Hillclimbing. Kluwer Academic Publishers, 1987 (the Ackley function).
Mixed-variable sign-flip construction after the mixed-variable BO benchmark suites; see B. Ru et al., ICML 2020.
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem
from ._common import cat_index


class Ackley5Mixed(BenchmarkProblem):
    """3-D Ackley with two binary categorical sign flips (5-D, mixed)."""

    available_dimensions = 5
    num_objectives = 1
    num_constraints = 0

    N_CONT = 3

    def __init__(self) -> None:
        self.variable_types = ["continuous"] * self.N_CONT + [[0.0, 1.0], [0.0, 1.0]]
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1.0, 1.0)] * self.N_CONT + [(0.0, 1.0), (0.0, 1.0)],
            x_opt=[[0.0, 0.0, 0.0, 0.0, 0.0]],
            optimum=[0.0],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        c0 = cat_index(x[:, 3], 2)
        c1 = cat_index(x[:, 4], 2)
        s0 = torch.where(c0 == 0, 1.0, -1.0).to(torch.float64)
        s1 = torch.where(c1 == 0, 1.0, -1.0).to(torch.float64)

        z = torch.stack([s0 * x[:, 0], s1 * x[:, 1], x[:, 2]], dim=1)

        n = self.N_CONT
        a, b, c = 20.0, 0.2, 2 * math.pi
        sum_sq = (z**2).sum(dim=1)
        sum_cos = torch.cos(c * z).sum(dim=1)
        f = (
            -a * torch.exp(-b * torch.sqrt(sum_sq / n))
            - torch.exp(sum_cos / n)
            + a
            + math.e
        )

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
