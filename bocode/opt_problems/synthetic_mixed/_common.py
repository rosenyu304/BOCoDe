"""Shared helpers for the mixed-variable synthetic problems.

Private module (leading underscore): skipped by ``tools/generate_registry.py``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def onesided_levels(ub: float, k: int) -> list[float]:
    """The ``k`` levels of a *one-sided* grid spanning ``[0, ub]``.

    The categorical SFU problems (``AckleyCat``, ``RastriginCat``, ``SchwefelCat``,
    ``Ackley53``) discretise a landscape whose natural box is *symmetric*,
    ``[-ub, ub]``. Gridding that box directly — ``linspace(-ub, ub, k)``, as MCBO
    does — is degenerate for an **even** landscape: Ackley and Rastrigin depend on
    each coordinate only through ``x**2`` and ``cos(2*pi*x)``, so level ``i`` and
    level ``k-1-i`` score *identically*. Half the levels are then redundant (K=4
    behaves like K=2), and for a binary grid ``{-ub, +ub}`` the two levels are
    indistinguishable, leaving the categorical variables completely **inert** — the
    failure mode ``Ackley5Mixed`` documents as a deliberate null benchmark, which is
    exactly what a *high-cardinality categorical* benchmark must not be.

    Placing the levels on ``[0, ub]`` keeps every level distinct, and keeps the
    minimiser ``x = 0`` of the even landscapes on the grid as level 0, so the
    discrete optimum is both known and reachable.
    """
    return np.linspace(0.0, ub, k).tolist()


def sfu_ackley(x: torch.Tensor, a: float = 20.0, b: float = 0.2) -> torch.Tensor:
    """The SFU Ackley function (minimization sense; ``f = 0`` at ``x = 0``)."""
    n = x.shape[1]
    c = 2.0 * math.pi
    return (
        -a * torch.exp(-b * torch.sqrt((x**2).sum(dim=1) / n))
        - torch.exp(torch.cos(c * x).sum(dim=1) / n)
        + a
        + math.e
    )


def cat_index(col: torch.Tensor, k: int) -> torch.Tensor:
    """Round a column of level *indices* to a valid index in ``[0, k-1]``.

    Used by the problems whose categorical factor is genuinely nominal (a
    material, a sub-function selector), so the decision variable carries the
    level index rather than a physical level value.
    """
    return col.round().long().clamp(0, k - 1)


def cocabo_subfunctions(
    x1: torch.Tensor, x2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three scaled CoCaBO sub-functions (Ru et al. 2020): Rosenbrock, six-hump camel, Beale."""
    ros = (100.0 * (x2 - x1**2) ** 2 + (x1 - 1.0) ** 2) / 300.0
    six = (
        (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
    ) / 10.0
    beale = (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2**2) ** 2
        + (2.625 - x1 + x1 * x2**3) ** 2
    ) / 50.0
    return ros, six, beale
