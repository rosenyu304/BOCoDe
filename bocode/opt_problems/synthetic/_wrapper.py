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

from ...base import BenchmarkProblem, DataType


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

    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    #: BoTorch ``SyntheticTestFunction`` subclass and its constructor kwargs.
    botorch_cls = None
    botorch_kwargs: dict = {}

    def __init__(self) -> None:
        self._fn = self.botorch_cls(negate=True, **self.botorch_kwargs)
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


class MultiObjSyntheticProblem(BenchmarkProblem):
    """Wrap a multi-objective BoTorch synthetic function (negated to maximize)."""

    input_type = DataType.CONTINUOUS
    num_constraints = 0

    botorch_cls = None
    botorch_kwargs: dict = {}

    def __init__(self) -> None:
        self._fn = self.botorch_cls(negate=True, **self.botorch_kwargs)
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
