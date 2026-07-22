"""Constrained Ackley: n-D single-objective, 2 inequality constraints (negated to maximize).

Standard Ackley objective (a=20, b=0.2, c=2*pi) on the box [-5, 10]^n, negated so
BoCoDe maximizes it. Two constraints, feasible when ``g(x) <= 0``:
    g1(x) = sum_i x_i        <= 0
    g2(x) = ||x||_2 - 5      <= 0

Reference:
    Eriksson D, Poloczek M (2021) Scalable constrained Bayesian optimization. In:
    International Conference on Artificial Intelligence and Statistics, PMLR, pp 730-738.
    https://github.com/rosenyu304/BOEngineeringBenchmark
"""

import math

import torch

from ...base import BenchmarkProblem


def _ackley(X: torch.Tensor) -> torch.Tensor:
    """Standard Ackley (minimization form) evaluated row-wise; returns shape (n,)."""
    a, b, c = 20.0, 0.2, 2 * math.pi
    d = X.shape[1]
    sum_sq = torch.sum(X**2, dim=1)
    sum_cos = torch.sum(torch.cos(c * X), dim=1)
    term1 = -a * torch.exp(-b * torch.sqrt(sum_sq / d))
    term2 = -torch.exp(sum_cos / d)
    return term1 + term2 + a + math.e


class _ConstrainedAckleyBase(BenchmarkProblem):
    """Shared implementation; subclasses fix the dimension via ``available_dimensions``."""

    num_objectives = 1
    num_constraints = 2
    available_dimensions: int | None = None

    def __init__(self):
        dim = self.available_dimensions
        super().__init__(
            dim=dim,
            num_objectives=1,
            num_constraints=2,
            bounds=[(-5.0, 10.0)] * dim,
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        fx = -_ackley(X)  # negate: BoCoDe maximizes
        gx1 = torch.sum(X, dim=1)
        gx2 = torch.norm(X, p=2, dim=1) - 5.0
        gx = torch.stack([gx1, gx2], dim=1)
        return gx, fx.reshape(-1, 1)


class ConstrainedAckley2D(_ConstrainedAckleyBase):
    """Constrained Ackley in 2 dimensions."""

    available_dimensions = 2


class ConstrainedAckley6D(_ConstrainedAckleyBase):
    """Constrained Ackley in 6 dimensions."""

    available_dimensions = 6


class ConstrainedAckley10D(_ConstrainedAckleyBase):
    """Constrained Ackley in 10 dimensions."""

    available_dimensions = 10
