"""WFG1-WFG9 scalable multi-objective synthetic test functions (Walking Fish Group toolkit).

Ported from the paper (formulas verified numerically against pymoo 0.6.1.3, which is
Apache-2.0 licensed; no code is copied):
S. Huband, P. Hingston, L. Barone, and L. While. A Review of Multiobjective Test
Problems and a Scalable Test Problem Toolkit. IEEE Transactions on Evolutionary
Computation, 10(5):477-506, 2006. https://doi.org/10.1109/TEVC.2005.861417

The WFG suite was built to cover the properties the ZDT/DTLZ problems miss. Each
problem chains *transformations* of the decision vector (shifts, biases, reductions)
into a *shape* function:

======  ====================================================================
WFG1    flat region + polynomial bias, mixed convex/concave front
WFG2    non-separable, disconnected convex front
WFG3    non-separable, degenerate (linear, lower-dimensional) front
WFG4    highly multimodal (many local fronts), concave
WFG5    deceptive, concave
WFG6    non-separable (hard reduction), concave
WFG7    parameter-dependent bias, concave
WFG8    parameter-dependent bias + non-separable, concave
WFG9    parameter-dependent bias + deceptive + multimodal + non-separable
======  ====================================================================

Variables: ``k`` position-related and ``l`` distance-related, ``n = k + l``; the domain
is ``x_i in [0, 2i]`` (1-based ``i``), so the box is deliberately non-uniformly scaled.
Defaults are the paper's standard setting ``M = 3`` objectives, ``k = 2 (M - 1) = 4``,
``l = 20`` (``n = 24``). ``l`` must be even for WFG2/WFG3. Pass ``dim`` and/or
``num_objectives`` to rescale (``k`` stays ``2 (M - 1)``, or 4 when ``M = 2``).

Every Pareto-optimal point has its distance variables at ``x_i = 0.35 * 2i`` (WFG1-7;
WFG8/9 have position-dependent optima) and lies on ``f_m = 2m * h_m``. Objectives lie
in ``[0, 2m + 1]`` (minimization frame) and are negated to maximize (BoCoDe
convention). The reference point is ``1.1 x`` the Pareto-front nadir ``(2, 4, ..., 2M)``,
i.e. ``-(2.2, 4.4, ..., 2.2 M)`` in the maximization frame, the usual WFG choice.
It is tight for WFG1 (whose flat-region bias keeps uniformly random points away from
the front, so a random sample scores a hypervolume of 0) and loose for WFG2-WFG9.
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem

_EPS = 1.0e-10


def _c01(x: torch.Tensor) -> torch.Tensor:
    """Snap round-off just outside [0, 1] back into it (WFG ``correct_to_01``)."""
    x = torch.where((x < 0) & (x >= -_EPS), torch.zeros_like(x), x)
    return torch.where((x > 1) & (x <= 1 + _EPS), torch.ones_like(x), x)


# -- transformations (Huband et al. 2006, Table II) -----------------------------------


def _s_linear(y, a=0.35):
    return _c01(torch.abs(y - a) / torch.abs(torch.floor(a - y) + a))


def _s_deceptive(y, a=0.35, b=0.001, c=0.05):
    t1 = torch.floor(y - a + b) * (1.0 - c + (a - b) / b) / (a - b)
    t2 = torch.floor(a + b - y) * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b)
    return _c01(1.0 + (torch.abs(y - a) - b) * (t1 + t2 + 1.0 / b))


def _s_multi(y, a, b, c):
    t1 = torch.abs(y - c) / (2.0 * (torch.floor(c - y) + c))
    t2 = (4.0 * a + 2.0) * math.pi * (0.5 - t1)
    return _c01((1.0 + torch.cos(t2) + 4.0 * b * t1**2) / (b + 2.0))


def _b_flat(y, a, b, c):
    zero = torch.zeros_like(y)
    r = (
        a
        + torch.minimum(zero, torch.floor(y - b)) * (a * (b - y) / b)
        - torch.minimum(zero, torch.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
    )
    return _c01(r)


def _b_poly(y, alpha):
    return _c01(y**alpha)


def _b_param(y, u, a=0.98 / 49.98, b=0.02, c=50.0):
    v = a - (1.0 - 2.0 * u) * torch.abs(torch.floor(0.5 - u) + a)
    return _c01(y ** (b + (c - b) * v))


def _r_sum(y, w):
    return _c01((y * w).sum(dim=1) / w.sum())


def _r_mean(y):
    return _c01(y.mean(dim=1))


def _r_nonsep(y, a: int):
    m = y.shape[1]
    num = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
    for j in range(m):
        num = num + y[:, j]
        for k in range(a - 1):
            num = num + torch.abs(y[:, j] - y[:, (1 + j + k) % m])
    half = math.ceil(a / 2.0)
    denom = m * half * (1.0 + 2.0 * a - 2.0 * half) / a
    return _c01(num / denom)


# -- shapes (Huband et al. 2006, Table I); m is 1-based ---------------------------------


def _h_concave(x, m):
    M = x.shape[1]
    if m == 1:
        r = torch.prod(torch.sin(0.5 * math.pi * x), dim=1)
    elif m <= M:
        r = torch.prod(torch.sin(0.5 * math.pi * x[:, : M - m + 1]), dim=1)
        r = r * torch.cos(0.5 * math.pi * x[:, M - m + 1])
    else:
        r = torch.cos(0.5 * math.pi * x[:, 0])
    return _c01(r)


def _h_convex(x, m):
    M = x.shape[1]
    if m == 1:
        r = torch.prod(1.0 - torch.cos(0.5 * math.pi * x), dim=1)
    elif m <= M:
        r = torch.prod(1.0 - torch.cos(0.5 * math.pi * x[:, : M - m + 1]), dim=1)
        r = r * (1.0 - torch.sin(0.5 * math.pi * x[:, M - m + 1]))
    else:
        r = 1.0 - torch.sin(0.5 * math.pi * x[:, 0])
    return _c01(r)


def _h_linear(x, m):
    M = x.shape[1]
    if m == 1:
        r = torch.prod(x, dim=1)
    elif m <= M:
        r = torch.prod(x[:, : M - m + 1], dim=1) * (1.0 - x[:, M - m + 1])
    else:
        r = 1.0 - x[:, 0]
    return _c01(r)


def _h_mixed(x, a=5.0, alpha=1.0):
    aux = 2.0 * a * math.pi
    return _c01((1.0 - x - torch.cos(aux * x + 0.5 * math.pi) / aux) ** alpha)


def _h_disc(x, alpha=1.0, beta=1.0, a=5.0):
    return _c01(1.0 - x**alpha * torch.cos(a * math.pi * x**beta) ** 2)


class _WFG(BenchmarkProblem):
    """Shared WFG machinery: bounds, the position/distance split and the final mapping."""

    available_dimensions = (3, 100)
    num_objectives = 3
    num_constraints = 0
    _degenerate = False  # WFG3: A_2..A_{M-1} = 0
    _even_l = False  # WFG2/WFG3 need an even number of distance variables

    def __init__(
        self, dim: int | None = None, num_objectives: int | None = None
    ) -> None:
        m = num_objectives if num_objectives is not None else 3
        if m < 2:
            raise ValueError("WFG problems need at least 2 objectives.")
        k = 4 if m == 2 else 2 * (m - 1)
        d = dim if dim is not None else k + 20
        l_ = d - k
        if l_ < 1:
            raise ValueError(f"{type(self).__name__}: dim must exceed k={k}, got {d}.")
        if self._even_l and l_ % 2:
            raise ValueError(f"{type(self).__name__}: dim - k must be even, got {l_}.")
        self.k, self.l = k, l_
        super().__init__(
            dim=d,
            num_objectives=m,
            num_constraints=0,
            bounds=[(0.0, 2.0 * (i + 1)) for i in range(d)],
            ref_point=[-1.1 * 2.0 * (i + 1) for i in range(m)],
        )

    def _position_reduce(self, y, reducer):
        """Reduce the k position variables in M-1 groups and the l distance ones to one."""
        m, k = self.num_objectives, self.k
        gap = k // (m - 1)
        cols = [reducer(y[:, i * gap : (i + 1) * gap], i) for i in range(m - 1)]
        cols.append(reducer(y[:, k:], None))
        return torch.stack(cols, dim=-1)

    def _finish(self, t, shapes):
        """Degeneracy constants A, then f_m = x_M + S_m h_m with S_m = 2m."""
        m = self.num_objectives
        A = [1.0] + [0.0 if self._degenerate else 1.0] * (m - 2)
        tm = t[:, -1]
        x = [torch.clamp(tm, min=A[i]) * (t[:, i] - 0.5) + 0.5 for i in range(m - 1)]
        x = torch.stack(x + [tm], dim=-1)
        h = shapes(x[:, :-1])
        S = torch.tensor(
            [2.0 * (i + 1) for i in range(m)], dtype=x.dtype, device=x.device
        )
        return x[:, -1:] + S * h

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = self.scale(X)
        X = X.to(torch.double)
        z = X / torch.tensor(
            [2.0 * (i + 1) for i in range(self.dim)], dtype=X.dtype, device=X.device
        )
        return None, -self._wfg(z)

    def _concave(self, x):
        return torch.stack(
            [_h_concave(x, i + 1) for i in range(self.num_objectives)], dim=-1
        )

    def _wfg(self, y):  # pragma: no cover - abstract
        raise NotImplementedError


def _t_shift_distance(y, k):
    """t1 shared by WFG1/2/3/6/7/8: linear shift of the distance variables."""
    return torch.cat([y[:, :k], _s_linear(y[:, k:], 0.35)], dim=1)


def _t_nonsep_pairs(y, k):
    """WFG2/WFG3 t2: keep position vars, pairwise non-separable reduction of distance vars."""
    l_ = y.shape[1] - k
    cols = [y[:, i] for i in range(k)]
    for i in range(l_ // 2):
        cols.append(_r_nonsep(y[:, k + 2 * i : k + 2 * i + 2], 2))
    return torch.stack(cols, dim=-1)


class WFG1(_WFG):
    """WFG1 (24-D, 3 objectives; flat + polynomial bias, mixed front), negated to maximize."""

    def _wfg(self, y):
        k, n, m = self.k, self.dim, self.num_objectives
        y = _t_shift_distance(y, k)
        y = torch.cat([y[:, :k], _b_flat(y[:, k:], 0.8, 0.75, 0.85)], dim=1)
        y = _b_poly(y, 0.02)
        w = torch.arange(2, 2 * n + 1, 2, dtype=y.dtype, device=y.device)
        gap = k // (m - 1)
        cols = [
            _r_sum(y[:, i * gap : (i + 1) * gap], w[i * gap : (i + 1) * gap])
            for i in range(m - 1)
        ]
        cols.append(_r_sum(y[:, k:], w[k:]))
        t = torch.stack(cols, dim=-1)

        def shapes(x):
            h = [_h_convex(x, i + 1) for i in range(m - 1)]
            return torch.stack(h + [_h_mixed(x[:, 0], a=5.0, alpha=1.0)], dim=-1)

        return self._finish(t, shapes)


class WFG2(_WFG):
    """WFG2 (24-D, 3 objectives; non-separable, disconnected front), negated to maximize."""

    _even_l = True

    def _wfg(self, y):
        m = self.num_objectives
        y = _t_nonsep_pairs(_t_shift_distance(y, self.k), self.k)
        t = self._position_reduce(y, lambda s, _i: _r_mean(s))

        def shapes(x):
            h = [_h_convex(x, i + 1) for i in range(m - 1)]
            return torch.stack(
                h + [_h_disc(x[:, 0], alpha=1.0, beta=1.0, a=5.0)], dim=-1
            )

        return self._finish(t, shapes)


class WFG3(_WFG):
    """WFG3 (24-D, 3 objectives; non-separable, degenerate linear front), negated to maximize."""

    _even_l = True
    _degenerate = True

    def _wfg(self, y):
        m = self.num_objectives
        y = _t_nonsep_pairs(_t_shift_distance(y, self.k), self.k)
        t = self._position_reduce(y, lambda s, _i: _r_mean(s))
        return self._finish(
            t, lambda x: torch.stack([_h_linear(x, i + 1) for i in range(m)], dim=-1)
        )


class WFG4(_WFG):
    """WFG4 (24-D, 3 objectives; highly multimodal, concave front), negated to maximize."""

    def _wfg(self, y):
        y = _s_multi(y, 30.0, 10.0, 0.35)
        t = self._position_reduce(y, lambda s, _i: _r_mean(s))
        return self._finish(t, self._concave)


class WFG5(_WFG):
    """WFG5 (24-D, 3 objectives; deceptive, concave front), negated to maximize."""

    def _wfg(self, y):
        y = _s_deceptive(y, 0.35, 0.001, 0.05)
        t = self._position_reduce(y, lambda s, _i: _r_mean(s))
        return self._finish(t, self._concave)


class WFG6(_WFG):
    """WFG6 (24-D, 3 objectives; non-separable reduction, concave front), negated to maximize."""

    def _wfg(self, y):
        y = _t_shift_distance(y, self.k)
        gap = self.k // (self.num_objectives - 1)
        t = self._position_reduce(
            y, lambda s, i: _r_nonsep(s, gap if i is not None else self.l)
        )
        return self._finish(t, self._concave)


class WFG7(_WFG):
    """WFG7 (24-D, 3 objectives; parameter-dependent bias, concave front), negated to maximize."""

    def _wfg(self, y):
        k = self.k
        cols = [_b_param(y[:, i], _r_mean(y[:, i + 1 :])) for i in range(k)]
        y = torch.cat([torch.stack(cols, dim=-1), y[:, k:]], dim=1)
        y = _t_shift_distance(y, k)
        t = self._position_reduce(y, lambda s, _i: _r_mean(s))
        return self._finish(t, self._concave)


class WFG8(_WFG):
    """WFG8 (24-D, 3 objectives; parameter-dependent bias, non-separable), negated to maximize."""

    def _wfg(self, y):
        k, n = self.k, self.dim
        cols = [
            _b_param(y[:, i], _r_mean(y[:, :i]), a=0.98 / 49.98, b=0.02, c=50.0)
            for i in range(k, n)
        ]
        y = torch.cat([y[:, :k], torch.stack(cols, dim=-1)], dim=1)
        y = _t_shift_distance(y, k)
        t = self._position_reduce(y, lambda s, _i: _r_mean(s))
        return self._finish(t, self._concave)


class WFG9(_WFG):
    """WFG9 (24-D, 3 objectives; deceptive + multimodal + non-separable), negated to maximize."""

    def _wfg(self, y):
        k, n = self.k, self.dim
        cols = [_b_param(y[:, i], _r_mean(y[:, i + 1 :])) for i in range(n - 1)]
        y = torch.cat([torch.stack(cols, dim=-1), y[:, n - 1 :]], dim=1)
        y = torch.cat(
            [
                _s_deceptive(y[:, :k], 0.35, 0.001, 0.05),
                _s_multi(y[:, k:], 30.0, 95.0, 0.35),
            ],
            dim=1,
        )
        gap = k // (self.num_objectives - 1)
        t = self._position_reduce(
            y, lambda s, i: _r_nonsep(s, gap if i is not None else self.l)
        )
        return self._finish(t, self._concave)
