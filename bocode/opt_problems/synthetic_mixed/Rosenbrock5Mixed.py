"""Rosenbrock-3 with two categorical coordinate offsets (3 continuous + 2 categorical, K=4).

A 5-D mixed-variable Rosenbrock: three continuous coordinates in ``[-2, 2]`` and
two 4-level categoricals that add an offset to the first two coordinates. Because
the offsets shift the *inputs* of a strongly coupled valley (rather than adding a
flat penalty), the categorical choice interacts with the continuous search.

Objective (minimize), with ``offset = [0.0, 0.5, 1.0, 1.5]``,
``z = (x_0 + offset[c_0], x_1 + offset[c_1], x_2)``::

    f(z) = sum_{i=0..1} [ 100 (z_{i+1} - z_i^2)^2 + (1 - z_i)^2 ]

Variables: ``x0, x1, x2`` continuous in ``[-2, 2]``; ``cat0``, ``cat1``
categorical with K=4 levels each (nominal, so the decision variable carries the
level *index*).

Reference optimum ``f* = 0`` (minimization sense) at ``z = (1, 1, 1)``, which is
reachable for *every* combination of levels (the continuous bounds are wide enough
to absorb any offset). The categoricals therefore change the location of the
optimum, not its value. ``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
H. H. Rosenbrock. An automatic method for finding the greatest or least value of a function. The Computer Journal 3(3):175-184, 1960.
Mixed-variable offset-categorical construction after the mixed-variable BO benchmark suites; see B. Ru et al., ICML 2020.
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem
from ._common import cat_index


class Rosenbrock5Mixed(BenchmarkProblem):
    """3-D Rosenbrock with two 4-level categorical coordinate offsets (5-D, mixed)."""

    available_dimensions = 5
    num_objectives = 1
    num_constraints = 0

    N_CONT = 3
    #: offset added to x0 / x1 by each level of the corresponding categorical
    OFFSETS = [0.0, 0.5, 1.0, 1.5]

    def __init__(self) -> None:
        self.variable_types = ["continuous"] * self.N_CONT + [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-2.0, 2.0)] * self.N_CONT + [(0.0, 3.0), (0.0, 3.0)],
            x_opt=[[1.0, 1.0, 1.0, 0.0, 0.0]],
            optimum=[0.0],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        offsets = torch.tensor(self.OFFSETS, dtype=torch.float64)
        c0 = cat_index(x[:, 3], len(self.OFFSETS))
        c1 = cat_index(x[:, 4], len(self.OFFSETS))

        z = torch.stack([x[:, 0] + offsets[c0], x[:, 1] + offsets[c1], x[:, 2]], dim=1)

        f = (100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1.0 - z[:, :-1]) ** 2).sum(
            dim=1
        )

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
