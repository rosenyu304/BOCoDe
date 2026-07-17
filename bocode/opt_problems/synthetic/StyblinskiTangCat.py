"""Styblinski-Tang on a 10-D purely categorical grid (10 categorical, K=5 each).

The 10-D Styblinski-Tang function restricted to a 5-level grid in *every*
dimension: each ``x_i`` is an unordered categorical over
``{-5, -2.5, 0, 2.5, 5}``. This is the only purely-discrete member of the
mixed-variable synthetic family, giving a 5^10 ~ 9.8e6 combinatorial search space.

Objective (minimize)::

    f(x) = 0.5 * sum_{i=1..10} (x_i^4 - 16 x_i^2 + 5 x_i)

Variables: ``x0..x9`` categorical in ``{-5, -2.5, 0, 2.5, 5}``. The levels are the
physical values of ``x_i``, so they are used directly by the objective.

The unconstrained minimum (``x_i ~ -2.903534``, ``f = -391.6617``) is *not* on the
grid: the best grid value is ``x_i = -2.5`` in every dimension, giving
``f* = -367.1875`` (minimization sense). ``evaluate()`` returns ``-f`` because
BoCoDe maximizes.

Sources:
M. Styblinski, T.-S. Tang. Experiments in nonconvex optimization: Stochastic approximation with function smoothing and simulated annealing. Neural Networks 3(4):467-483, 1990.
J. Qian. Discretised mixed-variable benchmark suite, MIT MEng thesis (Fig. 3.2 benchmarks).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem


class StyblinskiTangCat(BenchmarkProblem):
    """10-D Styblinski-Tang on a 5-level categorical grid (purely discrete)."""

    available_dimensions = 10
    num_objectives = 1
    num_constraints = 0

    DIM = 10
    #: the five levels available in every dimension (their physical values)
    LEVELS = [-5.0, -2.5, 0.0, 2.5, 5.0]

    def __init__(self) -> None:
        self.variable_types = [list(self.LEVELS)] * self.DIM
        super().__init__(
            dim=self.DIM,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-5.0, 5.0)] * self.DIM,
            x_opt=[[-2.5] * self.DIM],
            optimum=[-367.1875],  # best on-grid value; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        f = 0.5 * (x**4 - 16 * x**2 + 5 * x).sum(dim=1)
        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
