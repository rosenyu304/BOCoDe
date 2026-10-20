"""MW2, MW4-MW6, MW8-MW14: the rest of the MW constrained multi-objective suite.

Ported from the paper (formulas verified numerically against pymoo 0.6.1.3, itself
based on the authors' C++ code):
Z. Ma and Y. Wang. Evolutionary Constrained Multiobjective Optimization: Test Suite
Construction and Performance Comparisons. IEEE TEVC, 23(6):972-986, 2019.
https://doi.org/10.1109/TEVC.2019.2896967

MW1, MW3 and MW7 live in their own modules; this module completes the suite. The MW
problems combine three distance functions (g1: biased/deceptive, g2: multimodal,
g3: variable linkage) with constraints that cut the front into small feasible pieces
(MW2, MW3, MW5, MW9, MW10, MW13), leave only a narrow feasible tunnel to it (MW5, MW6,
MW11, MW12), or make it many-objective (MW4, MW8, MW14 are scalable in M; default 3).

Default dimensions follow the convention of BoCoDe's MW1/MW3/MW7: a small ``dim`` that
keeps the feasible region findable by a random initial design (measured feasible
fractions, from 20,000 uniform points, in each class docstring; 0% at the paper's
``dim = 15`` for all but MW13), not the paper's 15. Pass ``dim=15`` for the
paper-standard size. Objectives are negated to maximize; constraints are feasible when
``<= 0``. Reference points are ``1.2 x`` the nadir of the true Pareto front (rounded
up to 0.1), the rule that gives MW1/MW3/MW7's ``[-1.2, -1.2]``.
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem
from .MW1 import _la1, _safe_pow

_SQRT2 = math.sqrt(2.0)


def _la2(A, B, C, D, theta):
    return A * _safe_pow(torch.sin(B * _safe_pow(theta, C)), D)


def _la3(A, B, C, D, theta):
    return A * _safe_pow(torch.cos(B * _safe_pow(theta, C)), D)


class _MW(BenchmarkProblem):
    """Shared MW distance functions and constructor."""

    available_dimensions = (3, 100)
    num_objectives = 2
    num_constraints = 1
    _default_dim = 3
    _upper = 1.0
    _ref = (1.2, 1.2)
    _scalable_m = False

    def __init__(
        self, dim: int | None = None, num_objectives: int | None = None
    ) -> None:
        m = type(self).num_objectives
        if num_objectives is not None:
            if not self._scalable_m and num_objectives != m:
                raise ValueError(f"{type(self).__name__} has exactly {m} objectives.")
            m = num_objectives
        d = dim if dim is not None else max(self._default_dim, m + 1)
        if d < m:
            raise ValueError(f"{type(self).__name__}: dim must be >= num_objectives.")
        # Scalable-M problems store (ref of f_1..f_{M-1}, ref of f_M).
        ref = (
            list(self._ref)
            if not self._scalable_m
            else [self._ref[0]] * (m - 1) + [self._ref[1]]
        )
        super().__init__(
            dim=d,
            num_objectives=m,
            num_constraints=type(self).num_constraints,
            bounds=[(0.0, self._upper)] * d,
            ref_point=[-r for r in ref],
        )

    def _g1(self, X):
        d, m = self.dim, self.num_objectives
        n = d - m
        z = X[:, m - 1 :] ** n
        i = torch.arange(m - 1, d, dtype=X.dtype, device=X.device)
        return 1.0 + (1.0 - torch.exp(-10.0 * (z - 0.5 - i / (2 * d)) ** 2)).sum(dim=1)

    def _g2(self, X):
        d, m = self.dim, self.num_objectives
        i = torch.arange(m - 1, d, dtype=X.dtype, device=X.device)
        z = 1.0 - torch.exp(-10.0 * (X[:, m - 1 :] - i / d) ** 2)
        contrib = (0.1 / d) * z * z + 1.5 - 1.5 * torch.cos(2.0 * torch.pi * z)
        return 1.0 + contrib.sum(dim=1)

    def _g3(self, X):
        m = self.num_objectives
        contrib = 2.0 * (X[:, m - 1 :] + (X[:, m - 2 : -1] - 0.5) ** 2 - 1.0) ** 2
        return 1.0 + contrib.sum(dim=1)

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        F, G = self._mw(X.to(torch.double))
        return G, -F  # constraints <= 0 feasible; objectives maximized

    def _mw(self, X):  # pragma: no cover - abstract
        raise NotImplementedError


def _atan(f1, f0):
    """arctan(f1 / f0) as in the reference code (f0 = 0 gives +inf -> pi/2)."""
    return torch.atan(f1 / f0)


class MW2(_MW):
    """MW2 (3-D default, 2 objectives, 1 constraint; feasible <= 0), negated to maximize.

    Linear front broken into feasible segments by a sin^8 ripple; multimodal g2.
    Random feasible fraction at the default dim: 6.1%.
    """

    def _mw(self, X):
        g = self._g2(X)
        f0 = X[:, 0]
        f1 = g * (1.0 - f0 / g)
        c = f0 + f1 - 1.0 - _la1(0.5, 3.0, 1.0, 8.0, _SQRT2 * f1 - _SQRT2 * f0)
        return torch.stack([f0, f1], -1), c.unsqueeze(-1)


class MW4(_MW):
    """MW4 (3 objectives by default, 1 constraint), negated to maximize.

    Many-objective (scalable ``num_objectives``) linear simplex front cut by a sin^8
    ripple, deceptive g1. Random feasible fraction at the default dim: 3.6%.
    """

    num_objectives = 3
    available_dimensions = (4, 100)
    _default_dim = 4
    _scalable_m = True
    _ref = (1.2, 1.2)

    def _mw(self, X):
        m = self.num_objectives
        g = self._g1(X)
        f = g.unsqueeze(-1).expand(-1, m).clone()
        f[:, 1:] = f[:, 1:] * torch.flip(X[:, : m - 1], dims=[1])
        cp = torch.cumprod(1.0 - X[:, : m - 1], dim=1)
        f[:, :-1] = f[:, :-1] * torch.flip(cp, dims=[1])
        c = (
            f.sum(dim=1)
            - 1.0
            - _la1(0.4, 2.5, 1.0, 8.0, f[:, -1] - f[:, :-1].sum(dim=1))
        )
        return f, c.unsqueeze(-1)


class MW5(_MW):
    """MW5 (3-D default, 2 objectives, 3 constraints), negated to maximize.

    The feasible front is a handful of isolated points/arcs on a circle, reachable only
    through a narrow feasible ring. Random feasible fraction at the default dim: 11.3%.
    """

    num_constraints = 3

    def _mw(self, X):
        g = self._g1(X)
        f0 = g * X[:, 0]
        f1 = g * torch.sqrt(1.0 - (f0 / g) ** 2)
        at = _atan(f1, f0)
        c0 = f0**2 + f1**2 - (1.7 - _la2(0.2, 2.0, 1.0, 1.0, at)) ** 2
        t = 0.5 * math.pi - 2.0 * torch.abs(at - 0.25 * math.pi)
        c1 = (1.0 + _la2(0.5, 6.0, 3.0, 1.0, t)) ** 2 - f0**2 - f1**2
        c2 = (1.0 - _la2(0.45, 6.0, 3.0, 1.0, t)) ** 2 - f0**2 - f1**2
        return torch.stack([f0, f1], -1), torch.stack([c0, c1, c2], -1)


class MW6(_MW):
    """MW6 (3-D default, 2 objectives, 1 constraint; x in [0, 1.1]^d), negated to maximize.

    Disconnected circular front, multimodal g2. Random feasible fraction at the default dim: 3.4%.
    """

    _upper = 1.1
    _ref = (1.4, 1.4)

    def _mw(self, X):
        g = self._g2(X)
        f0 = g * X[:, 0]
        f1 = g * torch.sqrt(1.21 - (f0 / g) ** 2)
        at = _atan(f1, f0)
        c = (
            f0**2 / (1.0 + _la3(0.15, 6.0, 4.0, 10.0, at)) ** 2
            + f1**2 / (1.0 + _la3(0.75, 6.0, 4.0, 10.0, at)) ** 2
            - 1.0
        )
        return torch.stack([f0, f1], -1), c.unsqueeze(-1)


class MW8(_MW):
    """MW8 (3 objectives by default, 1 constraint), negated to maximize.

    Many-objective spherical front with feasible bands; multimodal g2. Random feasible fraction at the default dim: 3.2%.
    """

    num_objectives = 3
    available_dimensions = (4, 100)
    _default_dim = 4
    _scalable_m = True
    _ref = (1.2, 1.2)

    def _mw(self, X):
        m = self.num_objectives
        g = self._g2(X)
        f = g.unsqueeze(-1).expand(-1, m).clone()
        f[:, 1:] = f[:, 1:] * torch.sin(
            0.5 * math.pi * torch.flip(X[:, : m - 1], dims=[1])
        )
        cp = torch.cumprod(torch.cos(0.5 * math.pi * X[:, : m - 1]), dim=1)
        f[:, :-1] = f[:, :-1] * torch.flip(cp, dims=[1])
        sq = (f**2).sum(dim=1)
        rr = 1.25 - _la2(0.5, 6.0, 1.0, 2.0, torch.asin(f[:, -1] / torch.sqrt(sq)))
        return f, (sq - rr * rr).unsqueeze(-1)


class MW9(_MW):
    """MW9 (3-D default, 2 objectives, 1 constraint), negated to maximize.

    Convex front only partly feasible; deceptive g1. Random feasible fraction at the default dim: 17.9%.
    """

    def _mw(self, X):
        g = self._g1(X)
        f0 = g * X[:, 0]
        f1 = g * (1.0 - (f0 / g) ** 0.6)
        t1 = (1.0 - 0.64 * f0 * f0 - f1) * (1.0 - 0.36 * f0 * f0 - f1)
        t2 = (1.35**2 - (f0 + 0.35) ** 2 - f1) * (1.15**2 - (f0 + 0.15) ** 2 - f1)
        return torch.stack([f0, f1], -1), torch.minimum(t1, t2).unsqueeze(-1)


class MW10(_MW):
    """MW10 (3-D default, 2 objectives, 3 constraints), negated to maximize.

    Disconnected concave front segments; multimodal g2; f1 = g x1^n concentrates
    solutions near f1 = 0. Random feasible fraction at the default dim: 1.4%.
    """

    num_constraints = 3
    _ref = (1.2, 1.4)

    def _mw(self, X):
        g = self._g2(X)
        f0 = g * X[:, 0] ** self.dim
        f1 = g * (1.0 - (f0 / g) ** 2)
        c0 = -1.0 * (2.0 - 4.0 * f0 * f0 - f1) * (2.0 - 8.0 * f0 * f0 - f1)
        c1 = (2.0 - 2.0 * f0 * f0 - f1) * (2.0 - 16.0 * f0 * f0 - f1)
        c2 = (1.0 - f0 * f0 - f1) * (1.2 - 1.2 * f0 * f0 - f1)
        return torch.stack([f0, f1], -1), torch.stack([c0, c1, c2], -1)


class MW11(_MW):
    """MW11 (3-D default, 2 objectives, 4 constraints; x in [0, sqrt(2)]^d), negated to maximize.

    Four feasible pockets of the front, separated by large infeasible regions; variable
    linkage g3. Random feasible fraction at the default dim: 8.8%.
    """

    num_constraints = 4
    _upper = _SQRT2
    _ref = (2.5, 2.5)

    def _mw(self, X):
        g = self._g3(X)
        f0 = g * X[:, 0]
        f1 = g * torch.sqrt(2.0 - (f0 / g) ** 2)
        c0 = -1.0 * (3.0 - f0 * f0 - f1) * (3.0 - 2.0 * f0 * f0 - f1)
        c1 = (3.0 - 0.625 * f0 * f0 - f1) * (3.0 - 7.0 * f0 * f0 - f1)
        c2 = -1.0 * (1.62 - 0.18 * f0 * f0 - f1) * (1.125 - 0.125 * f0 * f0 - f1)
        c3 = (2.07 - 0.23 * f0 * f0 - f1) * (0.63 - 0.07 * f0 * f0 - f1)
        return torch.stack([f0, f1], -1), torch.stack([c0, c1, c2, c3], -1)


class MW12(_MW):
    """MW12 (3-D default, 2 objectives, 2 constraints), negated to maximize.

    Front lies on the boundary of a thin wavy feasible band; deceptive g1. Random feasible fraction at the default dim: 20.3%.
    """

    num_constraints = 2
    _ref = (1.6, 1.3)

    def _mw(self, X):
        g = self._g1(X)
        f0 = g * X[:, 0]
        r = f0 / g
        f1 = g * (0.85 - 0.8 * r - 0.08 * torch.abs(torch.sin(3.2 * math.pi * r)))
        s = torch.sin
        c0 = (
            -1.0
            * (1.0 - 0.625 * f0 - f1 + 0.08 * s(2 * math.pi * (f1 - f0 / 1.6)))
            * (1.4 - 0.875 * f0 - f1 + 0.08 * s(2 * math.pi * (f1 / 1.4 - f0 / 1.6)))
        )
        c1 = (1.0 - 0.8 * f0 - f1 + 0.08 * s(2 * math.pi * (f1 - f0 / 1.5))) * (
            1.8 - 1.125 * f0 - f1 + 0.08 * s(2 * math.pi * (f1 / 1.8 - f0 / 1.6))
        )
        return torch.stack([f0, f1], -1), torch.stack([c0, c1], -1)


class MW13(_MW):
    """MW13 (3-D default, 2 objectives, 2 constraints; x in [0, 1.5]^d), negated to maximize.

    Disconnected front pieces on a thin feasible band; multimodal g2. Random feasible fraction at the default dim: 12.1%.
    """

    num_constraints = 2
    _upper = 1.5
    _ref = (1.9, 4.8)

    def _mw(self, X):
        g = self._g2(X)
        f0 = g * X[:, 0]
        r = f0 / g
        f1 = g * (5.0 - torch.exp(r) - torch.abs(0.5 * torch.sin(3 * math.pi * r)))
        sn = 0.5 * torch.sin(3 * math.pi * f0)
        c0 = (
            -1.0
            * (5.0 - (1.0 + f0 + 0.5 * f0 * f0) - sn - f1)
            * (5.0 - (1.0 + 0.7 * f0) - sn - f1)
        )
        c1 = (5.0 - torch.exp(f0) - sn - f1) * (5.0 - (1.0 + 0.4 * f0) - sn - f1)
        return torch.stack([f0, f1], -1), torch.stack([c0, c1], -1)


class MW14(_MW):
    """MW14 (3 objectives by default, 1 constraint; x in [0, 1.5]^d), negated to maximize.

    Many-objective front made of 2^(M-1) disconnected pieces; variable linkage g3.
    Random feasible fraction at the default dim: 6.8%.
    """

    num_objectives = 3
    available_dimensions = (4, 100)
    _default_dim = 4
    _scalable_m = True
    _upper = 1.5
    _ref = (1.8, 6.0)

    def _mw(self, X):
        m = self.num_objectives
        g = self._g3(X)
        fx = X[:, : m - 1]
        la = _la1(1.5, 1.1, 2.0, 1.0, fx)
        fm = g / (m - 1) * (6.0 - torch.exp(fx) - la).sum(dim=1)
        alpha = 6.1 - 1.0 - fx - 0.5 * fx * fx - la
        c = fm - alpha.sum(dim=1) / (m - 1)
        return torch.cat([fx, fm.unsqueeze(-1)], dim=1), c.unsqueeze(-1)
