"""Ackley on a 20-D purely categorical grid (20 categorical, K=4 each).

The 20-D Ackley function restricted to a 4-level grid in *every* dimension, giving a
``4^20 ~ 1.1e12`` purely-categorical search space. One of the standard
high-cardinality categorical benchmarks used to separate categorical specialists
(Casmopolitan, COMBO, BODi) from continuous-relaxation BO.

Objective (minimize), with ``a = 20``, ``b = 0.2``, ``c = 2 pi`` and ``n = 20``::

    f(x) = -a exp(-b sqrt(sum x_i^2 / n)) - exp(sum cos(c x_i) / n) + a + e

Variables: ``x0..x19`` categorical over the four levels
``{0, 10.9227, 21.8453, 32.768}``. The levels are the physical values of ``x_i``,
so they are used directly by the objective.

The levels are the *one-sided* grid ``linspace(0, 32.768, 4)`` over the SFU Ackley
box ``[-32.768, 32.768]`` (see ``_common.onesided_levels``). Level 0 is the value
``0``, so the global minimum ``f* = 0`` at ``x = 0`` is on the grid and attainable.

.. note::
   **The one-sided grid is what makes the categorical variables matter.** Ackley is
   *even* in every coordinate, so on the symmetric grid ``linspace(-32.768, 32.768,
   4)`` that MCBO builds by default, level ``i`` and level ``3 - i`` contribute
   identically: only 2 of the 4 levels are distinguishable and the effective
   cardinality is halved. On the one-sided grid all four levels give distinct
   contributions. See ``Ackley5Mixed`` for the same evenness pathology taken to its
   limit (fully inert categoricals).

``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
D. H. Ackley. A Connectionist Machine for Genetic Hillclimbing. Kluwer Academic Publishers, 1987 (the Ackley function).
S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments: Test Functions and Datasets — Ackley Function. https://www.sfu.ca/~ssurjano/ackley.html
K. Dreczkowski, A. Grosnit, H. Bou-Ammar. Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization. Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks), 2023. arXiv:2306.09803 (the categorical Ackley task and its 20-D / K=4 grid).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem
from ._common import onesided_levels, sfu_ackley


class AckleyCat(BenchmarkProblem):
    """20-D Ackley on a 4-level categorical grid (purely discrete)."""

    available_dimensions = 20
    num_objectives = 1
    num_constraints = 0

    DIM = 20
    NUM_CATEGORIES = 4
    UB = 32.768  # the SFU Ackley box is [-UB, UB]

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
        fx = (-sfu_ackley(x)).reshape(-1, 1)  # maximize -f
        return None, fx
