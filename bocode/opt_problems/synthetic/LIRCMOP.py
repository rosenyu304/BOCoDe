"""LIR-CMOP1-LIR-CMOP14: constrained multi-objective problems with large infeasible regions.

Ported from the paper's Table 6 (formulas cross-checked numerically against an
independent transcription of the PlatEMO reference code):
Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. You, J. Mo, C. Wei, and E. Goodman.
An Improved Epsilon Constraint-handling Method in MOEA/D for CMOPs with Large
Infeasible Regions. Soft Computing, 23:12491-12510, 2019.
https://doi.org/10.1007/s00500-019-03794-x (arXiv:1707.08767)

Large infeasible regions block the way to the front, in two flavours: in LIR-CMOP1-4
the feasible region is a vanishingly small part of the box (a random sample of 2000
points at n = 30 or n = 10 contains no feasible point), while in LIR-CMOP5-14 most of
the box is feasible but the region *in front of* the Pareto front is not, so a method
that is guided by objectives alone is pulled into the infeasible obstacles:

* LIR-CMOP1-4: both distance functions must sit in the thin band ``g in [0.5, 0.51]``
  (two constraints), plus ``sin(20 pi x1) >= 0.5`` in LIR-CMOP3/4, which also makes
  the front disconnected;
* LIR-CMOP5-8: two or three rotated ellipses in objective space are infeasible and
  sit directly between the population and the front (convergence-hard);
* LIR-CMOP9-12: an ellipse obstacle plus a sinusoidal constraint that leaves only
  disconnected pieces of the front feasible;
* LIR-CMOP13-14 (3 objectives): ``|f|^2`` must lie in one or two thin spherical shells.

Index sets follow the paper: ``J1 = {3, 5, ...}`` (odd, 1-based) and
``J2 = {2, 4, ...}`` (even), ``n = 30`` variables in ``[0, 1]``. In LIR-CMOP5-12 the
targets are ``sin(0.5 j pi x1 / n)`` and ``cos(0.5 j pi x1 / n)``.

Note: jMetalPy 1.7's LIRCMOP classes deviate from the paper (``+ 7057`` instead of
``+ 0.7057`` in LIR-CMOP5, ``1.7957`` for ``1.7057`` in LIR-CMOP9, ``cos(0.5 pi + x2)``
for ``cos(0.5 pi x2)`` in LIR-CMOP13) and were *not* used as the reference.

Objectives are negated to maximize; constraints are feasible when ``<= 0`` (the paper
writes ``c(x) >= 0``, negated here). Reference points are ``1.1 x`` the nadir of the
true (constrained, non-dominated) Pareto front, rounded up to 0.05; the front is the
paper's analytic front (the ``GetOptimum`` construction of the PlatEMO reference code).
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem

_TH = -0.25 * math.pi
_AL = 0.25 * math.pi


def _ellipse(f0, f1, p, q, a, b, r=0.1):
    """Paper ``((f1-p)cos t - (f2-q)sin t)^2/a^2 + ((f1-p)sin t + (f2-q)cos t)^2/b^2 - r``."""
    u = (f0 - p) * math.cos(_TH) - (f1 - q) * math.sin(_TH)
    v = (f0 - p) * math.sin(_TH) + (f1 - q) * math.cos(_TH)
    return u**2 / a**2 + v**2 / b**2 - r


def _wave(f0, f1, shift):
    """Paper ``f1 sin a + f2 cos a - sin(4 pi (f1 cos a - f2 sin a)) - shift``."""
    return (
        f0 * math.sin(_AL)
        + f1 * math.cos(_AL)
        - torch.sin(4.0 * math.pi * (f0 * math.cos(_AL) - f1 * math.sin(_AL)))
        - shift
    )


class _LIRCMOP(BenchmarkProblem):
    available_dimensions = (3, 100)
    num_objectives = 2
    num_constraints = 2
    _ref: tuple = ()

    def __init__(self, dim: int | None = None) -> None:
        d = dim if dim is not None else 30
        super().__init__(
            dim=d,
            num_objectives=type(self).num_objectives,
            num_constraints=type(self).num_constraints,
            bounds=[(0.0, 1.0)] * d,
            ref_point=[-r for r in self._ref],
        )

    def _odd_even(self, X):
        """Columns of J1 = {3, 5, ...} and J2 = {2, 4, ...} (1-based), with their j."""
        n = self.dim
        j1 = torch.arange(3, n + 1, 2, dtype=X.dtype, device=X.device)
        j2 = torch.arange(2, n + 1, 2, dtype=X.dtype, device=X.device)
        return X[:, 2::2], j1, X[:, 1::2], j2

    def _g_fixed(self, X):
        """LIR-CMOP1-4: g1 = sum_J1 (x_j - sin(0.5 pi x1))^2, g2 with cos over J2."""
        xo, _, xe, _ = self._odd_even(X)
        x1 = X[:, :1]
        g1 = ((xo - torch.sin(0.5 * math.pi * x1)) ** 2).sum(dim=1)
        g2 = ((xe - torch.cos(0.5 * math.pi * x1)) ** 2).sum(dim=1)
        return g1, g2

    def _g_scaled(self, X):
        """LIR-CMOP5-12: targets sin/cos(0.5 j pi x1 / n)."""
        xo, j1, xe, j2 = self._odd_even(X)
        x1 = X[:, :1]
        n = self.dim
        g1 = ((xo - torch.sin(0.5 * j1 / n * math.pi * x1)) ** 2).sum(dim=1)
        g2 = ((xe - torch.cos(0.5 * j2 / n * math.pi * x1)) ** 2).sum(dim=1)
        return g1, g2

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        F, C = self._lir(X.to(torch.double))
        return -C, -F  # paper c >= 0 feasible -> BoCoDe <= 0; objectives maximized

    def _lir(self, X):  # pragma: no cover - abstract
        raise NotImplementedError


# -- LIR-CMOP1-4: thin g-bands ---------------------------------------------------------


class LIRCMOP1(_LIRCMOP):
    """LIR-CMOP1 (30-D, 2 objectives, 2 constraints; feasible <= 0), negated to maximize."""

    _ref = (1.65, 1.65)
    _sqrt = False
    _sin = False

    def _lir(self, X):
        g1, g2 = self._g_fixed(X)
        x1 = X[:, 0]
        f0 = x1 + g1
        f1 = (1.0 - torch.sqrt(x1) if self._sqrt else 1.0 - x1**2) + g2
        cs = [(0.51 - g1) * (g1 - 0.5), (0.51 - g2) * (g2 - 0.5)]
        if self._sin:
            cs.append(torch.sin(20.0 * math.pi * x1) - 0.5)
        return torch.stack([f0, f1], -1), torch.stack(cs, -1)


class LIRCMOP2(LIRCMOP1):
    """LIR-CMOP2 (30-D, 2 objectives, 2 constraints; convex front), negated to maximize."""

    _sqrt = True


class LIRCMOP3(LIRCMOP1):
    """LIR-CMOP3 (30-D, 2 objectives, 3 constraints; disconnected front), negated to maximize."""

    _ref = (1.6, 1.65)

    num_constraints = 3
    _sin = True


class LIRCMOP4(LIRCMOP3):
    """LIR-CMOP4 (30-D, 2 objectives, 3 constraints; disconnected convex front), negated to maximize."""

    _ref = (1.6, 1.55)

    _sqrt = True


# -- LIR-CMOP5-8: ellipse obstacles in objective space ---------------------------------


class LIRCMOP5(_LIRCMOP):
    """LIR-CMOP5 (30-D, 2 objectives, 2 constraints; ellipse obstacles), negated to maximize."""

    _ref = (1.9, 1.9)
    _sqrt = True
    _ell = ((1.6, 1.6, 2.0, 4.0), (2.5, 2.5, 2.0, 8.0))

    def _lir(self, X):
        g1, g2 = self._g_scaled(X)
        x1 = X[:, 0]
        f0 = x1 + 10.0 * g1 + 0.7057
        f1 = (1.0 - torch.sqrt(x1) if self._sqrt else 1.0 - x1**2) + 10.0 * g2 + 0.7057
        cs = [_ellipse(f0, f1, p, q, a, b) for p, q, a, b in self._ell]
        return torch.stack([f0, f1], -1), torch.stack(cs, -1)


class LIRCMOP6(LIRCMOP5):
    """LIR-CMOP6 (30-D, 2 objectives, 2 constraints; concave front), negated to maximize."""

    _sqrt = False
    _ell = ((1.8, 1.8, 2.0, 8.0), (2.8, 2.8, 2.0, 8.0))


class LIRCMOP7(LIRCMOP5):
    """LIR-CMOP7 (30-D, 2 objectives, 3 constraints; three ellipses), negated to maximize."""

    _ref = (2.65, 2.65)

    num_constraints = 3
    _ell = ((1.2, 1.2, 2.0, 6.0), (2.25, 2.25, 2.5, 12.0), (3.5, 3.5, 2.5, 10.0))


class LIRCMOP8(LIRCMOP7):
    """LIR-CMOP8 (30-D, 2 objectives, 3 constraints; concave, three ellipses), negated to maximize."""

    _sqrt = False


# -- LIR-CMOP9-12: ellipse + sinusoidal front-cutting constraint -----------------------


class LIRCMOP9(_LIRCMOP):
    """LIR-CMOP9 (30-D, 2 objectives, 2 constraints; disconnected front), negated to maximize."""

    _ref = (2.05, 2.45)
    _sqrt = False
    _ell = (1.4, 1.4, 1.5, 6.0)
    _shift = 2.0

    def _lir(self, X):
        g1, g2 = self._g_scaled(X)
        x1 = X[:, 0]
        f0 = 1.7057 * x1 * (10.0 * g1 + 1.0)
        h = 1.0 - torch.sqrt(x1) if self._sqrt else 1.0 - x1**2
        f1 = 1.7057 * h * (10.0 * g2 + 1.0)
        c0 = _ellipse(f0, f1, *self._ell)
        c1 = _wave(f0, f1, self._shift)
        return torch.stack([f0, f1], -1), torch.stack([c0, c1], -1)


class LIRCMOP10(LIRCMOP9):
    """LIR-CMOP10 (30-D, 2 objectives, 2 constraints; convex, disconnected), negated to maximize."""

    _ref = (1.95, 1.9)

    _sqrt = True
    _ell = (1.1, 1.2, 2.0, 4.0)
    _shift = 1.0


class LIRCMOP11(LIRCMOP10):
    """LIR-CMOP11 (30-D, 2 objectives, 2 constraints; convex, disconnected), negated to maximize."""

    _ref = (2.1, 2.45)

    _ell = (1.2, 1.2, 1.5, 5.0)
    _shift = 2.1


class LIRCMOP12(LIRCMOP9):
    """LIR-CMOP12 (30-D, 2 objectives, 2 constraints; concave, disconnected), negated to maximize."""

    _ref = (2.85, 2.5)

    _ell = (1.6, 1.6, 1.5, 6.0)
    _shift = 2.5


# -- LIR-CMOP13-14: thin spherical shells (3 objectives) -------------------------------


class LIRCMOP13(_LIRCMOP):
    """LIR-CMOP13 (30-D, 3 objectives, 2 constraints; spherical shells), negated to maximize."""

    num_objectives = 3
    _ref = (1.9, 1.9, 1.9)
    _shells = ((9.0, 4.0), (3.61, 3.24))

    def _lir(self, X):
        g = 10.0 * ((X[:, 2:] - 0.5) ** 2).sum(dim=1)
        a, b = 0.5 * math.pi * X[:, 0], 0.5 * math.pi * X[:, 1]
        r = 1.7057 + g
        F = torch.stack(
            [
                r * torch.cos(a) * torch.cos(b),
                r * torch.cos(a) * torch.sin(b),
                r * torch.sin(a),
            ],
            -1,
        )
        s = (F**2).sum(dim=1)
        cs = [(s - hi) * (s - lo) for hi, lo in self._shells]
        return F, torch.stack(cs, -1)


class LIRCMOP14(LIRCMOP13):
    """LIR-CMOP14 (30-D, 3 objectives, 3 constraints; three spherical shells), negated to maximize."""

    num_constraints = 3
    _ref = (1.95, 1.95, 1.95)
    _shells = ((9.0, 4.0), (3.61, 3.24), (3.0625, 2.56))
