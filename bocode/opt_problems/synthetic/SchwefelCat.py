"""Schwefel on a 20-D purely categorical grid (20 categorical, K=4 each).

The 20-D Schwefel function restricted to a 4-level grid in *every* dimension, giving
a ``4^20 ~ 1.1e12`` purely-categorical search space. The hardest of the three
categorical SFU landscapes: Schwefel is *deceptive*, its global basin sitting far
from the centre of the box and next to the second-best basin, so a model that
extrapolates smoothly across levels is actively misled.

Objective (minimize), with ``n = 20``::

    f(x) = 418.9829 n - sum_{i=1..n} x_i sin(sqrt(|x_i|))

Variables: ``x0..x19`` categorical over the four levels
``{0, 166.6667, 333.3333, 500}``. The levels are the physical values of ``x_i``, so
they are used directly by the objective.

The levels are the *one-sided* grid ``linspace(0, 500, 4)`` over the SFU Schwefel box
``[-500, 500]``, matching ``AckleyCat`` and ``RastriginCat`` (see
``_common.onesided_levels``).

The unconstrained minimum (``x_i = 420.9687``, ``f = 0``) is *not* on the grid. The
function is separable, so the best on-grid value is exact: the per-coordinate term
``x sin(sqrt|x|)`` is maximised over the four levels at ``x = 500/3 = 166.6667``
(giving ``56.1424``), so ``f* = 418.9829 * 20 - 20 * 56.1424 = 7256.8105``
(minimization sense) at ``x_i = 166.6667`` in every dimension.

.. note::
   Unlike Ackley and Rastrigin, Schwefel's per-coordinate term ``x sin(sqrt|x|)`` is
   *odd* rather than even, so a symmetric grid would **not** be degenerate for it.
   The one-sided grid is used here for consistency with the rest of the categorical
   SFU family, which does require it.

``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
H.-P. Schwefel. Numerical Optimization of Computer Models. John Wiley & Sons, 1981 (the Schwefel function).
S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments: Test Functions and Datasets — Schwefel Function. https://www.sfu.ca/~ssurjano/schwef.html
K. Dreczkowski, A. Grosnit, H. Bou-Ammar. Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization. Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks), 2023. arXiv:2306.09803 (the categorical SFU task grids).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem
from ._common import onesided_levels


class SchwefelCat(BenchmarkProblem):
    """20-D Schwefel on a 4-level categorical grid (purely discrete)."""

    available_dimensions = 20
    num_objectives = 1
    num_constraints = 0

    DIM = 20
    NUM_CATEGORIES = 4
    UB = 500.0  # the SFU Schwefel box is [-UB, UB]

    def __init__(self) -> None:
        self.LEVELS = onesided_levels(self.UB, self.NUM_CATEGORIES)
        self.variable_types = [list(self.LEVELS)] * self.DIM
        super().__init__(
            dim=self.DIM,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, self.UB)] * self.DIM,
            x_opt=[[self.LEVELS[1]] * self.DIM],  # every variable at level 1 (500/3)
            optimum=[7256.810531],  # best on-grid value; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = self.enforce_variable_types(X.to(torch.float64))
        n = x.shape[1]
        f = 418.9829 * n - (x * torch.sin(torch.sqrt(torch.abs(x)))).sum(dim=1)
        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
