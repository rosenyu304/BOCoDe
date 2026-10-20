"""DAS-CMOP1-DAS-CMOP9: difficulty-adjustable constrained multi-objective test problems.

Ported from the paper (formulas verified numerically against pymoo 0.6.1.3):
Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, and E. Goodman.
Difficulty Adjustable and Scalable Constrained Multiobjective Test Problem Toolkit.
Evolutionary Computation, 28(3):339-378, 2020. https://doi.org/10.1162/evco_a_00259
(arXiv:1612.07603)

Each problem has three independently tunable constraint difficulties, the triplet
``(eta, zeta, gamma)`` in [0, 1]^3:

* ``eta`` -- *diversity*: ``sin(20 pi x1) >= 2 eta - 1`` (and ``cos(20 pi x2)`` for the
  3-objective ones) chops the front into many disconnected feasible pieces;
* ``zeta`` -- *feasibility*: ``(e - g)(g - d) >= 0`` with ``d = 0.5``,
  ``e = d - ln(zeta)`` keeps the distance function in a band ``g in [d, e]``, so the
  unconstrained front (``g = 0``) is entirely infeasible and the feasible region
  shrinks as ``zeta`` grows;
* ``gamma`` -- *convergence*: ellipses (2-objective) / spheres (3-objective) of radius
  ``0.5 gamma`` block parts of the front, creating infeasible obstacles in front of it.

The default triplet is ``(0.5, 0.5, 0.5)`` (all three difficulties at medium level, the
``DIFFICULTIES[7]`` setting of the reference implementation and a common choice in the
constrained-EMO literature); pass ``difficulty=(eta, zeta, gamma)`` or an integer
1..16 (the reference implementation's table) to change it. ``n = 30`` variables in
``[0, 1]`` as in the paper. DAS-CMOP1-6 have 2 objectives and 11 constraints,
DAS-CMOP7-9 have 3 objectives and 7 constraints.

Distance functions: g1 (DAS-CMOP1-3) is ``sum_{j>=M} (x_j - sin(0.5 pi x_1))^2``; the
arXiv table writes the sum from ``j = 1`` but the paper's Eq. (3) and pymoo sum from
``j = M`` (so that ``g = 0`` is attainable); this port follows Eq. (3)/pymoo.
g2 (DAS-CMOP4-8) is a Rastrigin-type multimodal function, g3 (DAS-CMOP9) has variable
linkage on ``(x1 + x2)``.

Objectives negated to maximize; constraints are feasible when ``<= 0``. Reference
point: ``1.1 x`` the nadir ``(1.5, ..., 1.5)`` of the reference Pareto fronts that
pymoo ships for the default difficulty (identical for all nine problems), i.e.
``-(1.65, ..., 1.65)``. The same point is kept for other difficulty triplets: it stays
a fixed, algorithm-independent reference (so hypervolumes remain comparable), but it
is then not guaranteed to be exactly ``1.1 x`` that triplet's nadir.
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem

#: The reference implementation's 16 difficulty triplets (eta, zeta, gamma), 1-based.
DIFFICULTIES = [
    (0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.25), (0.25, 0.25, 0.25),
    (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5),
    (0.75, 0.0, 0.0), (0.0, 0.75, 0.0), (0.0, 0.0, 0.75), (0.75, 0.75, 0.75),
    (0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (0.0, 1.0, 0.5), (0.5, 1.0, 0.5),
]  # fmt: skip

_P2 = (0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0)
_Q2 = (1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5)
_S3 = 1.0 / math.sqrt(3.0)
_C3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (_S3, _S3, _S3))


class _DASCMOP(BenchmarkProblem):
    available_dimensions = (3, 100)
    num_objectives = 2
    num_constraints = 11
    _ref: tuple = ()

    def __init__(
        self,
        dim: int | None = None,
        difficulty: int | tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        if isinstance(difficulty, int):
            if not 1 <= difficulty <= len(DIFFICULTIES):
                raise ValueError(f"difficulty must be in 1..{len(DIFFICULTIES)}")
            difficulty = DIFFICULTIES[difficulty - 1]
        self.eta, self.zeta, self.gamma = (float(v) for v in difficulty)
        d = dim if dim is not None else 30
        m = type(self).num_objectives
        super().__init__(
            dim=d,
            num_objectives=m,
            num_constraints=type(self).num_constraints,
            bounds=[(0.0, 1.0)] * d,
            ref_point=[-r for r in self._ref],
        )

    # -- distance functions ------------------------------------------------------------
    def _g1(self, X):
        m = self.num_objectives
        return ((X[:, m - 1 :] - torch.sin(0.5 * math.pi * X[:, :1])) ** 2).sum(dim=1)

    def _g2(self, X):
        m = self.num_objectives
        z = X[:, m - 1 :] - 0.5
        return (self.dim - m + 1) + (z**2 - torch.cos(20.0 * math.pi * z)).sum(dim=1)

    def _g3(self, X):
        m, n = self.num_objectives, self.dim
        j = torch.arange(m, n + 1, dtype=X.dtype, device=X.device)
        tgt = torch.cos(0.25 * j / n * math.pi * (X[:, :1] + X[:, 1:2]))
        return ((X[:, m - 1 :] - tgt) ** 2).sum(dim=1)

    # -- shared constraint pieces (paper form c >= 0 feasible) -------------------------
    def _band(self, g):
        d = 0.5 if self.zeta != 0 else 0.0
        e = d - math.log(self.zeta) if self.zeta > 0 else 1e30
        if self.zeta == 1.0:
            return 1e-4 - torch.abs(e - g)
        return (e - g) * (g - d)

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        F, C = self._das(X.to(torch.double))
        return -C, -F  # paper c >= 0 feasible -> BoCoDe <= 0; objectives maximized

    def _das(self, X):  # pragma: no cover - abstract
        raise NotImplementedError


class _DAS2(_DASCMOP):
    """2-objective DAS-CMOP base (11 constraints)."""

    _ref = (1.65, 1.65)

    def _cons(self, X, f0, f1, g):
        b = 2.0 * self.eta - 1.0
        r = 0.5 * self.gamma
        th = -0.25 * math.pi
        p = torch.tensor(_P2, dtype=X.dtype, device=X.device)
        q = torch.tensor(_Q2, dtype=X.dtype, device=X.device)
        df0, df1 = f0.unsqueeze(-1) - p, f1.unsqueeze(-1) - q
        ell = (
            (df0 * math.cos(th) - df1 * math.sin(th)) ** 2 / 0.3
            + (df0 * math.sin(th) + df1 * math.cos(th)) ** 2 / 1.2
            - r
        )
        c0 = torch.sin(20.0 * math.pi * X[:, 0]) - b
        return torch.cat([c0.unsqueeze(-1), self._band(g).unsqueeze(-1), ell], dim=1)

    def _das(self, X):
        g = self._gfun(X)
        x1 = X[:, 0]
        f0 = x1 + g
        f1 = self._h(x1) + g
        return torch.stack([f0, f1], -1), self._cons(X, f0, f1, g)


class DASCMOP1(_DAS2):
    """DAS-CMOP1 (30-D, 2 objectives, 11 constraints; feasible <= 0), negated to maximize."""

    def _gfun(self, X):
        return self._g1(X)

    def _h(self, x1):
        return 1.0 - x1**2


class DASCMOP2(DASCMOP1):
    """DAS-CMOP2 (30-D, 2 objectives, 11 constraints; convex front), negated to maximize."""

    def _h(self, x1):
        return 1.0 - torch.sqrt(x1)


class DASCMOP3(DASCMOP1):
    """DAS-CMOP3 (30-D, 2 objectives, 11 constraints; disconnected front), negated to maximize."""

    def _h(self, x1):
        return 1.0 - torch.sqrt(x1) + 0.5 * torch.abs(torch.sin(5.0 * math.pi * x1))


class DASCMOP4(DASCMOP1):
    """DAS-CMOP4 (30-D, 2 objectives, 11 constraints; multimodal g), negated to maximize."""

    def _gfun(self, X):
        return self._g2(X)


class DASCMOP5(DASCMOP2):
    """DAS-CMOP5 (30-D, 2 objectives, 11 constraints; multimodal g), negated to maximize."""

    def _gfun(self, X):
        return self._g2(X)


class DASCMOP6(DASCMOP3):
    """DAS-CMOP6 (30-D, 2 objectives, 11 constraints; multimodal g), negated to maximize."""

    def _gfun(self, X):
        return self._g2(X)


class _DAS3(_DASCMOP):
    """3-objective DAS-CMOP base (7 constraints)."""

    num_objectives = 3
    num_constraints = 7
    _ref = (1.65, 1.65, 1.65)

    def _das(self, X):
        g = self._gfun(X)
        F = self._objs(X, g)
        b = 2.0 * self.eta - 1.0
        r = 0.5 * self.gamma
        c0 = torch.sin(20.0 * math.pi * X[:, 0]) - b
        c1 = torch.cos(20.0 * math.pi * X[:, 1]) - b
        ctr = torch.tensor(_C3, dtype=X.dtype, device=X.device)  # (4, 3)
        sph = ((F.unsqueeze(1) - ctr.unsqueeze(0)) ** 2).sum(dim=-1) - r**2
        C = torch.cat(
            [c0.unsqueeze(-1), c1.unsqueeze(-1), self._band(g).unsqueeze(-1), sph],
            dim=1,
        )
        return F, C


class DASCMOP7(_DAS3):
    """DAS-CMOP7 (30-D, 3 objectives, 7 constraints; linear front), negated to maximize."""

    def _gfun(self, X):
        return self._g2(X)

    def _objs(self, X, g):
        x1, x2 = X[:, 0], X[:, 1]
        return torch.stack([x1 * x2 + g, x2 * (1.0 - x1) + g, 1.0 - x2 + g], -1)


class DASCMOP8(DASCMOP7):
    """DAS-CMOP8 (30-D, 3 objectives, 7 constraints; spherical front), negated to maximize."""

    def _objs(self, X, g):
        c1, s1 = torch.cos(0.5 * math.pi * X[:, 0]), torch.sin(0.5 * math.pi * X[:, 0])
        c2, s2 = torch.cos(0.5 * math.pi * X[:, 1]), torch.sin(0.5 * math.pi * X[:, 1])
        return torch.stack([c1 * c2 + g, c1 * s2 + g, s1 + g], -1)


class DASCMOP9(DASCMOP8):
    """DAS-CMOP9 (30-D, 3 objectives, 7 constraints; linked g), negated to maximize."""

    def _gfun(self, X):
        return self._g3(X)
