"""Branin with an additive categorical offset (2 continuous + 1 categorical, K=3).

The standard 2-D Branin augmented with an unordered categorical that selects one
of three additive offsets, so different levels have different optimal values and
the optimizer must identify the best arm. The categorical is *separable* from the
continuous search (it only shifts the objective), which makes this the easy
counterpart to the non-separable CoCaBO functions.

Objective (minimize)::

    f(x1, x2, c) = a (x2 - b x1^2 + c1 x1 - r)^2 + s (1 - t) cos(x1) + s + offset[c]

with ``a = 1``, ``b = 5.1 / (4 pi^2)``, ``c1 = 5 / pi``, ``r = 6``, ``s = 10``,
``t = 1 / (8 pi)`` and ``offset = [0.0, 1.0, 2.5]``.

Variables: ``x1`` continuous in ``[-5, 10]``; ``x2`` continuous in ``[0, 15]``;
``c`` categorical with K=3 levels (nominal, so the decision variable carries the
level *index*).

Reference optimum ``f* = 0.397887`` (minimization sense) at the best level
``c = 0``, at any of Branin's three global optima. ``evaluate()`` returns ``-f``
because BoCoDe maximizes.

Sources:
F. H. Branin. Widely convergent method for finding multiple solutions of simultaneous nonlinear equations. IBM Journal of Research and Development 16(5):504-522, 1972 (the Branin function).
Additive-offset categorical construction after the mixed-variable synthetic suites; see B. Ru et al., ICML 2020.
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem
from ._common import cat_index


class BraninCategorical(BenchmarkProblem):
    """Branin plus a 3-level nominal categorical additive offset (3-D, mixed)."""

    available_dimensions = 3
    num_objectives = 1
    num_constraints = 0

    #: additive penalty applied by each level of the categorical
    OFFSETS = [0.0, 1.0, 2.5]

    def __init__(self) -> None:
        self.variable_types = ["continuous", "continuous", [0.0, 1.0, 2.0]]
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-5.0, 10.0), (0.0, 15.0), (0.0, 2.0)],
            x_opt=[[math.pi, 2.275, 0.0]],
            optimum=[0.397887],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        x1, x2 = x[:, 0], x[:, 1]
        c = cat_index(x[:, 2], len(self.OFFSETS))
        offsets = torch.tensor(self.OFFSETS, dtype=torch.float64)

        a = 1.0
        b = 5.1 / (4 * math.pi**2)
        cc = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * math.pi)

        branin = (
            a * (x2 - b * x1**2 + cc * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
        )
        f = branin + offsets[c]

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
