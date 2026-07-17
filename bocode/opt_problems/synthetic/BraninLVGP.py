"""Branin-Hoo with one qualitative factor (1 continuous + 1 categorical, K=4).

The mixed quantitative/qualitative Branin benchmark used in the LVGP paper. The
second Branin input ``x2`` is discretised to the four levels ``{0, 5, 10, 15}``
and presented as an *unordered* categorical, even though the levels carry a
hidden quantitative ordering — the canonical probe for whether a method recovers
that latent ordering instead of treating the levels as interchangeable.

Objective (minimize)::

    f(x1, x2) = a (x2 - b x1^2 + c x1 - r)^2 + s (1 - t) cos(x1) + s

with ``a = 1``, ``b = 5.1 / (4 pi^2)``, ``c = 5 / pi``, ``r = 6``, ``s = 10``,
``t = 1 / (8 pi)``.

Variables: ``x1`` continuous in ``[-5, 10]``; ``x2`` categorical in
``{0, 5, 10, 15}``. The best level is the interior level ``x2 = 10``.

Reference optimum ``f* = 2.79118`` (minimization sense) at ``x1 ~ -2.62``,
``x2 = 10``. ``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
Y. Zhang, D. W. Apley, W. Chen. Bayesian Optimization for Materials Design with Mixed Quantitative and Qualitative Variables. Scientific Reports 10:4924, 2020 (arXiv:1910.01688).
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem


class BraninLVGP(BenchmarkProblem):
    """Branin-Hoo with x2 recast as a 4-level unordered categorical (2-D, mixed)."""

    available_dimensions = 2
    num_objectives = 1
    num_constraints = 0

    #: the four qualitative levels of x2 (their physical values)
    LEVELS = [0.0, 5.0, 10.0, 15.0]

    def __init__(self) -> None:
        self.variable_types = ["continuous", list(self.LEVELS)]
        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-5.0, 10.0), (0.0, 15.0)],
            x_opt=[[-2.62, 10.0]],
            optimum=[2.79118],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x1, x2 = X[:, 0], X[:, 1]
        a = 1.0
        b = 5.1 / (4 * math.pi**2)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * math.pi)

        f = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
        fx = (-f).reshape(-1, 1).to(torch.float64)  # maximize -f
        return None, fx
