"""Thin wrappers exposing BoTorch synthetic test functions as BenchmarkProblems.

These are classic *synthetic* optimization test functions (Ackley, Branin, DTLZ1,
…), kept separate from BoCoDe's real-world problem suite. They are used for the
example notebook and for testing algorithms on cheap, well-understood landscapes.
They are accessible as ``bocode.synthetic.<Name>`` and via
``bocode.get_problem("<Name>")`` (so the algorithm CLIs can target them), but they
are deliberately **excluded** from ``bocode.list_problems()`` and
``CATEGORIZATION.md``, which describe only the real-world problems.

BoTorch test functions are written for minimization; here they are wrapped with
``negate=True`` so the returned objective is *maximized*, matching BoCoDe's
convention.

Sources:
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem


def _scale_clamped(problem: BenchmarkProblem, X: torch.Tensor) -> torch.Tensor:
    """Scale ``[0, 1]^d`` input to the problem bounds, inset to stay strictly inside.

    BoTorch test functions validate that inputs lie within their bounds; a float
    cast at the boundary can overshoot, so we inset by a tiny relative epsilon.
    """
    X = problem.scale(X.clamp(0.0, 1.0))
    bounds = problem.torch_bounds.to(X)
    lo, hi = bounds[:, 0], bounds[:, 1]
    eps = 1e-6 * (hi - lo).clamp_min(1e-12)
    return torch.maximum(torch.minimum(X, hi - eps), lo + eps)


class SingleObjSyntheticProblem(BenchmarkProblem):
    """Wrap a single-objective BoTorch synthetic function (negated to maximize)."""

    num_objectives = 1
    num_constraints = 0

    #: BoTorch ``SyntheticTestFunction`` subclass and its constructor kwargs.
    botorch_cls = None
    botorch_kwargs: dict = {}

    def __init__(self, dim: int | None = None) -> None:
        # ``dim`` overrides the default for scalable functions (Ackley, Levy, …);
        # ignored by fixed-dimension functions whose constructor takes no ``dim``.
        kwargs = dict(self.botorch_kwargs)
        if dim is not None:
            kwargs["dim"] = dim
        self._fn = self.botorch_cls(negate=True, **kwargs)
        bounds = list(zip(*self._fn.bounds.numpy(), strict=True))
        super().__init__(
            dim=self._fn.dim,
            num_objectives=1,
            num_constraints=0,
            bounds=bounds,
        )

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = _scale_clamped(self, X)
        return None, self._fn(X.to(torch.double)).unsqueeze(-1)


class ShiftedSingleObjSyntheticProblem(SingleObjSyntheticProblem):
    """A single-objective synthetic whose optimum is moved off the box center.

    Six of the centered synthetics (Ackley, Griewank, Rastrigin, DixonPrice, Levy,
    Powell) have their global optimum *exactly at the center* of the search box, so
    any method with a center-biased initial design -- or a linear embedding whose
    origin maps to the box center -- gets the optimum for free. This is a known
    high-dimensional-BO pitfall; the Vanilla-BO paper (Hvarfner et al., 2024) shifts
    its test functions for exactly this reason.

    The shift is a fixed input OFFSET, applied before evaluating::

        f_shifted(x) = f(x - delta)

    so the optimum moves from ``x*`` to ``x* + delta`` and the optimal VALUE is
    unchanged. ``delta`` places the optimum at a fixed fraction of each dimension's
    range -- the 2/3 point along even-indexed dims and the 1/3 point along odd-indexed
    dims ("2/3 to the right, 1/3 down")::

        R_j      = hi_j - lo_j
        target_j = lo_j + (2/3) * R_j        for even j (0, 2, 4, ...)
        target_j = lo_j + (1/3) * R_j        for odd  j (1, 3, 5, ...)
        delta_j  = target_j - x*_j

    For a problem whose optimum is already centered (``x*_j = lo_j + R_j / 2``) this is
    exactly ``delta_j = +(2/3 - 1/2) * R_j`` on even dims and ``-(1/2 - 1/3) * R_j`` on
    odd dims. Anchoring on ``x*`` rather than assuming a centered optimum keeps the
    shifted optimum strictly inside the bounds (and away from both the center and the
    corners) for the problems whose optimum is *not* centered -- Rosenbrock,
    StyblinskiTang, Hartmann, Branin, SixHumpCamel.

    The search bounds are unchanged. The underlying BoTorch function is therefore
    evaluated on the translated box ``[lo - delta, hi - delta]``, which reaches outside
    the bounds BoTorch validates against, so those validation bounds are widened here.
    The analytic formulas are defined everywhere, and for every problem shifted in this
    module the global optimum value is still attained on the translated box (verified by
    random search in ``tests/test_shifted_synthetics.py``). EggHolder and HolderTable are
    deliberately NOT shifted: their optima sit on the boundary of the box and their
    formulas are unbounded below outside it, so no translation can move the optimum
    inside without breaking the reference optimal value.

    Problems with several global optima (Branin, SixHumpCamel) are anchored on the first
    one; the others translate by the same delta and may fall outside the box.

    Subclasses inherit ``botorch_cls`` / ``botorch_kwargs`` / ``available_dimensions``
    from their centered counterpart, e.g.
    ``class AckleyShifted(Ackley, ShiftedSingleObjSyntheticProblem)``.
    """

    def __init__(self, dim: int | None = None) -> None:
        super().__init__(dim=dim)
        bounds = self.torch_bounds.to(torch.double)
        lo, hi = bounds[:, 0], bounds[:, 1]
        x_opt = torch.tensor(self._fn._optimizers[0], dtype=torch.double)
        fractions = torch.tensor(
            [2 / 3 if j % 2 == 0 else 1 / 3 for j in range(self.dim)],
            dtype=torch.double,
        )
        target = lo + fractions * (hi - lo)
        if not bool(((target > lo) & (target < hi)).all()):
            raise ValueError(
                f"{type(self).__name__}: shifted optimum is not inside the bounds."
            )
        self._delta = target - x_opt
        # The translated box reaches outside the bounds BoTorch validates its inputs
        # against; widen them (with a small pad against float32 rounding at the edge).
        pad = 1e-6 * (hi - lo)
        self._fn.bounds = torch.stack(
            [
                torch.minimum(lo - self._delta, lo) - pad,
                torch.maximum(hi - self._delta, hi) + pad,
            ]
        ).to(self._fn.bounds)
        self.x_opt = target
        # BoCoDe convention: ``optimum`` is the literature f* in the MINIMIZATION frame,
        # so ``evaluate(x_opt) == -optimum`` (the wrapper negates to maximize).
        self.optimum = self._fn._optimal_value

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = _scale_clamped(self, X)
        X = X.to(torch.double) - self._delta.to(X.device)
        return None, self._fn(X).unsqueeze(-1)


class ConstrainedSingleObjSyntheticProblem(BenchmarkProblem):
    """Wrap a constrained single-objective BoTorch synthetic function.

    The objective is negated to maximize. BoTorch reports constraint *slack* that is
    feasible when ``>= 0``; here it is negated to BoCoDe's convention (feasible when
    ``<= 0``). Subclasses set ``num_constraints`` (a class attribute, so the algorithm
    harness can categorize the problem without instantiating it).
    """

    num_objectives = 1
    num_constraints = 0

    botorch_cls = None
    botorch_kwargs: dict = {}

    def __init__(self) -> None:
        self._fn = self.botorch_cls(negate=True, **self.botorch_kwargs)
        bounds = list(zip(*self._fn.bounds.numpy(), strict=True))
        super().__init__(
            dim=self._fn.dim,
            num_objectives=1,
            num_constraints=self.num_constraints,
            bounds=bounds,
        )

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = _scale_clamped(self, X)
        Xd = X.to(torch.double)
        obj = self._fn(Xd).unsqueeze(-1)
        gx = -self._fn.evaluate_slack(Xd)  # BoTorch slack >= 0 feasible -> gx <= 0
        return gx, obj


class MultiObjSyntheticProblem(BenchmarkProblem):
    """Wrap a multi-objective BoTorch synthetic function (negated to maximize)."""

    num_constraints = 0

    botorch_cls = None
    botorch_kwargs: dict = {}

    def __init__(self, dim: int | None = None) -> None:
        kwargs = dict(self.botorch_kwargs)
        if dim is not None:
            kwargs["dim"] = dim
        self._fn = self.botorch_cls(negate=True, **kwargs)
        bounds = list(zip(*self._fn.bounds.numpy(), strict=True))
        ref = getattr(self._fn, "ref_point", None)
        # ref_point is in the (negated) maximization frame already when negate=True.
        super().__init__(
            dim=self._fn.dim,
            num_objectives=self._fn.num_objectives,
            num_constraints=0,
            bounds=bounds,
            ref_point=None if ref is None else ref.tolist(),
        )

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = _scale_clamped(self, X)
        return None, self._fn(X.to(torch.double))


class ConstrainedMultiObjSyntheticProblem(BenchmarkProblem):
    """Wrap a constrained multi-objective BoTorch synthetic function.

    Objectives are negated to maximize. BoTorch reports constraint *slack* that is
    feasible when ``>= 0``; here it is negated to BoCoDe's convention (feasible when
    ``<= 0``). Subclasses set ``num_objectives`` and ``num_constraints`` (class
    attributes, so the algorithm harness can categorize the problem without
    instantiating it).
    """

    num_objectives = 2
    num_constraints = 0

    botorch_cls = None
    botorch_kwargs: dict = {}

    def __init__(self, dim: int | None = None) -> None:
        kwargs = dict(self.botorch_kwargs)
        if dim is not None:
            kwargs["dim"] = dim
        self._fn = self.botorch_cls(negate=True, **kwargs)
        bounds = list(zip(*self._fn.bounds.numpy(), strict=True))
        ref = getattr(self._fn, "ref_point", None)
        # ref_point is in the (negated) maximization frame already when negate=True.
        super().__init__(
            dim=self._fn.dim,
            num_objectives=self._fn.num_objectives,
            num_constraints=self.num_constraints,
            bounds=bounds,
            ref_point=None if ref is None else ref.tolist(),
        )

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = _scale_clamped(self, X)
        Xd = X.to(torch.double)
        obj = self._fn(Xd)
        gx = -self._fn.evaluate_slack(Xd)  # BoTorch slack >= 0 feasible -> gx <= 0
        return gx, obj
