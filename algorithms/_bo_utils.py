"""Shared evaluation harness for the BoCoDe algorithm scripts.

Following the CleanRL philosophy, the optimization *logic* of each algorithm lives
in its own single file; this module only holds the shared plumbing that every
script needs: reproducible seeding, the two search-space adapters
(problem-optimization over the unit cube vs. dataset-optimization over a discrete
candidate pool), a GP-fitting helper, and a small results container.

Conventions
-----------
* BoCoDe maximizes the returned objective, so every adapter exposes a single
  objective ``Y`` (shape ``(n, 1)``) that is *maximized*.
* The continuous search space is the unit cube ``[0, 1]^d``; the adapter scales
  proposals to the problem's true bounds before evaluating.
* Constraints follow BoCoDe's convention (``c <= 0`` is feasible). Unconstrained
  algorithms fold them into a feasibility-aware objective via :func:`penalized`.

Sources:
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.double


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def penalized(values: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Fold constraint violations into the objective for unconstrained solvers.

    Subtracts the total positive violation (``sum(max(c, 0))``) from the objective
    so infeasible points are penalized. Feasible points are unchanged.
    """
    if constraints is None or constraints.numel() == 0:
        return values
    violation = constraints.clamp(min=0).sum(dim=1, keepdim=True)
    return values - violation


class ProblemObjective:
    """Continuous single-objective maximization over the unit cube ``[0, 1]^d``.

    Proposals in ``[0, 1]^d`` are scaled to the problem's bounds and evaluated.
    ``__call__`` returns the (penalized) objective to maximize; ``evaluate_raw``
    returns the unmodified ``(values, constraints)`` for constraint-aware solvers.
    """

    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim
        self.num_constraints = problem.num_constraints
        self.bounds = torch.stack(
            [torch.zeros(self.dim), torch.ones(self.dim)]
        ).to(DTYPE)

    def evaluate_raw(self, X_unit: torch.Tensor):
        X_unit = X_unit.clamp(0.0, 1.0).to(DTYPE)
        X = self.problem.scale(X_unit)
        values, constraints = self.problem.evaluate(X)
        return values[:, :1].to(DTYPE), (
            None if constraints is None else constraints.to(DTYPE)
        )

    def __call__(self, X_unit: torch.Tensor) -> torch.Tensor:
        values, constraints = self.evaluate_raw(X_unit)
        return penalized(values, constraints)


class DatasetObjective:
    """Discrete single-objective maximization over a fixed candidate pool.

    Wraps a :class:`bocode.opt_problems.materials._dataset_problem.MaterialsDatasetProblem`
    (or any problem exposing ``candidates`` and ``values``). Features are min-max
    normalized to ``[0, 1]`` so a GP sees a unit-cube input. ``select`` returns the
    objective for chosen candidate indices.
    """

    def __init__(self, dataset_problem):
        X = dataset_problem.candidates.to(DTYPE)
        self._lo = X.min(dim=0).values
        self._span = (X.max(dim=0).values - self._lo).clamp_min(1e-12)
        self.X = (X - self._lo) / self._span
        self.Y = dataset_problem.values.to(DTYPE).reshape(-1, 1)
        self.dim = self.X.shape[1]
        self.n_candidates = self.X.shape[0]

    def select(self, idx: torch.Tensor) -> torch.Tensor:
        return self.Y[idx]


def fit_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """Fit a SingleTaskGP with input normalization and output standardization."""
    model = SingleTaskGP(
        train_X=train_X.to(DTYPE),
        train_Y=train_Y.to(DTYPE),
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=train_Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


@dataclass
class Result:
    """Container for a single optimization run's trace."""

    algorithm: str
    problem: str
    seed: int
    best_history: list = field(default_factory=list)  # best objective after each eval

    @property
    def best(self) -> float:
        return self.best_history[-1] if self.best_history else float("nan")

    def log(self, current_best: float) -> None:
        self.best_history.append(float(current_best))
