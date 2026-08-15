"""Rastrigin on a 20-D purely categorical grid (20 categorical, K=4 each).

The 20-D Rastrigin function restricted to a 4-level grid in *every* dimension,
giving a ``4^20 ~ 1.1e12`` purely-categorical search space. A companion to
``AckleyCat`` with the same grid construction but a strongly multimodal, *separable*
landscape (Ackley is non-separable), which isolates how much a categorical model
gains from modelling variable interactions.

Objective (minimize), with ``n = 20``::

    f(x) = 10 n + sum_{i=1..n} (x_i^2 - 10 cos(2 pi x_i))

Variables: ``x0..x19`` categorical over the four levels
``{0, 1.7067, 3.4133, 5.12}``. The levels are the physical values of ``x_i``, so
they are used directly by the objective.

The levels are the *one-sided* grid ``linspace(0, 5.12, 4)`` over the SFU Rastrigin
box ``[-5.12, 5.12]`` (see ``_common.onesided_levels``). Level 0 is the value ``0``,
so the global minimum ``f* = 0`` at ``x = 0`` is on the grid and attainable.

.. note::
   **The one-sided grid is what makes the categorical variables matter.** Rastrigin
   is *even* in every coordinate (it depends on ``x_i`` only through ``x_i^2`` and
   ``cos(2 pi x_i)``), so on the symmetric grid ``linspace(-5.12, 5.12, 4)`` level
   ``i`` and level ``3 - i`` contribute identically: only 2 of the 4 levels are
   distinguishable and the effective cardinality is halved. On the one-sided grid
   all four levels give distinct contributions.

``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
L. A. Rastrigin. Systems of extremal control. Nauka, Moscow, 1974 (the Rastrigin function).
S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments: Test Functions and Datasets — Rastrigin Function. https://www.sfu.ca/~ssurjano/rastr.html
K. Dreczkowski, A. Grosnit, H. Bou-Ammar. Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization. Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks), 2023. arXiv:2306.09803 (the categorical SFU task grids).
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem
from ._common import onesided_levels


class RastriginCat(BenchmarkProblem):
    """20-D Rastrigin on a 4-level categorical grid (purely discrete)."""

    available_dimensions = 20
    num_objectives = 1
    num_constraints = 0

    DIM = 20
    NUM_CATEGORIES = 4
    UB = 5.12  # the SFU Rastrigin box is [-UB, UB]

    def __init__(self) -> None:
        self.LEVELS = onesided_levels(self.UB, self.NUM_CATEGORIES)
        self.variable_types = [list(self.LEVELS)] * self.DIM
        super().__init__(
            dim=self.DIM,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, self.UB)] * self.DIM,
            x_opt=[[0.0] * self.DIM],  # every variable at level 0 (the value 0)
            optimum=[0.0],  # on-grid global minimum; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = self.enforce_variable_types(X.to(torch.float64))
        n = x.shape[1]
        f = 10.0 * n + (x**2 - 10.0 * torch.cos(2.0 * math.pi * x)).sum(dim=1)
        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
