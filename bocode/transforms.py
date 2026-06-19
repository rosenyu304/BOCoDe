"""Problem transforms: derive a new runnable problem from an existing one.

Three wrappers turn a base :class:`~bocode.base.BenchmarkProblem` into a different
category so the same problem can be studied under different settings:

- :class:`PenalizedProblem` — constrained -> unconstrained single-objective, by adding
  a (scale-normalized) penalty on the total constraint violation to the objective.
- :class:`ScalarizedProblem` — multi-objective -> single-objective, by a
  (scale-normalized) weighted sum of the objectives.
- :class:`ContinuousRelaxation` — mixed-variable -> continuous, by dropping the
  integer/categorical restrictions (no snapping).

Each is itself a ``BenchmarkProblem`` (so every algorithm and the experiment runner can
consume it) and they compose, e.g.
``ScalarizedProblem(PenalizedProblem(base))``.

**Normalization.** Naively combining an objective with a constraint violation (or with
another objective) is dominated by whichever term has the larger magnitude. So at
construction each wrapper draws a Latin-Hypercube sample (``base.sample``) and rescales
each term to a comparable ``[0, 1]`` range before combining with equal weights — the
standard remedy in the constrained/multi-objective literature:

- weighted-sum scalarization with per-objective normalization: R. T. Marler & J. S.
  Arora, "Survey of multi-objective optimization methods for engineering," *Structural
  and Multidisciplinary Optimization* 26(6):369-395, 2004 (and "The weighted sum method
  for multi-objective optimization: new insights," *SMO* 41(6):853-862, 2010).
- normalized objective + constraint-violation penalty: B. Tessema & G. G. Yen, "An
  adaptive penalty formulation for constrained evolutionary optimization," *IEEE Trans.
  Systems, Man, and Cybernetics - A* 39(3):565-578, 2009; consistent with the CEC2020
  real-world suite's normalized mean violation (A. Kumar et al., *Swarm and Evolutionary
  Computation* 56:100693, 2020).
"""

from __future__ import annotations

import torch

from .base import BenchmarkProblem

_EPS = 1e-12


def _finite_range(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column (lo, hi) over the finite rows of ``t`` (shape (n, k))."""
    mask = torch.isfinite(t).all(dim=1)
    t = t[mask] if mask.any() else t
    return t.min(dim=0).values, t.max(dim=0).values


class _Wrapper(BenchmarkProblem):
    """Shared plumbing: hold a base problem and call it via its public evaluate()."""

    # Non-None class attrs so BenchmarkProblem.__new__'s validity check passes; each
    # instance overrides num_objectives/num_constraints via super().__init__.
    available_dimensions = 0
    num_objectives = 1
    num_constraints = 0

    def _base_eval(self, X: torch.Tensor):
        # base.evaluate returns (values, constraints); _evaluate_implementation must
        # return (constraints, values), so callers re-flip as needed.
        values, constraints = self.base.evaluate(X)
        return values.to(torch.double), (
            None if constraints is None else constraints.to(torch.double)
        )


class PenalizedProblem(_Wrapper):
    """Constrained single-objective problem made unconstrained via a normalized penalty.

    Value (maximized) = ``obj_norm - weight * violation_norm``, where ``obj`` and the
    total violation ``sum(max(0, g_i))`` are each min-max normalized from a
    construction-time sample. With ``weight=1`` the objective and the violation carry
    equal weight on a common scale (Tessema & Yen, 2009).
    """

    def __init__(
        self,
        base: BenchmarkProblem,
        weight: float = 1.0,
        n_sample: int = 256,
        seed: int = 0,
    ) -> None:
        if base.num_constraints == 0:
            raise ValueError("PenalizedProblem needs a constrained base problem.")
        if base.num_objectives != 1:
            raise ValueError("Scalarize multi-objective problems before penalizing.")
        self.base = base
        self.weight = float(weight)
        self.variable_types = base.variable_types

        X = base.sample(n_sample, seed=seed)
        values, constraints = base.evaluate(X)
        obj = values[:, :1]
        viol = constraints.clamp(min=0).sum(dim=1, keepdim=True)
        self._obj_lo, self._obj_hi = _finite_range(obj)
        self._viol_hi = _finite_range(viol)[1]

        super().__init__(
            dim=base.dim,
            num_objectives=1,
            num_constraints=0,
            bounds=base.bounds,
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        values, constraints = self._base_eval(X)
        obj = values[:, :1]
        viol = constraints.clamp(min=0).sum(dim=1, keepdim=True)
        obj_n = (obj - self._obj_lo) / (self._obj_hi - self._obj_lo + _EPS)
        viol_n = viol / (self._viol_hi + _EPS)
        return None, obj_n - self.weight * viol_n


class ScalarizedProblem(_Wrapper):
    """Multi-objective problem reduced to single-objective by a normalized weighted sum.

    Each objective is min-max normalized from a construction-time sample, then combined
    with ``weights`` (default equal, ``1/m``) (Marler & Arora, 2004/2010). Any
    constraints are preserved.
    """

    def __init__(
        self, base: BenchmarkProblem, weights=None, n_sample: int = 256, seed: int = 0
    ) -> None:
        if base.num_objectives < 2:
            raise ValueError("ScalarizedProblem needs a multi-objective base problem.")
        self.base = base
        m = base.num_objectives
        if weights is None:
            w = torch.full((m,), 1.0 / m, dtype=torch.double)
        else:
            w = torch.as_tensor(weights, dtype=torch.double)
            w = w / w.sum()
        self.weights = w
        self.variable_types = base.variable_types

        X = base.sample(n_sample, seed=seed)
        values, _ = base.evaluate(X)
        self._lo, self._hi = _finite_range(values)

        super().__init__(
            dim=base.dim,
            num_objectives=1,
            num_constraints=base.num_constraints,
            bounds=base.bounds,
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        values, constraints = self._base_eval(X)
        vn = (values - self._lo) / (self._hi - self._lo + _EPS)
        scalar = (vn * self.weights).sum(dim=1, keepdim=True)
        return constraints, scalar


class ContinuousRelaxation(_Wrapper):
    """Mixed-variable problem relaxed to fully continuous (integer/categorical dropped).

    Drops ``variable_types`` so no snapping/rounding is applied; the objective is
    evaluated on the continuous box. Not meaningful for discrete candidate-pool
    (dataset) problems, which are rejected.
    """

    def __init__(self, base: BenchmarkProblem) -> None:
        if not base.is_mixed_variable:
            raise ValueError(
                "ContinuousRelaxation needs a mixed-variable base problem."
            )
        if hasattr(base, "candidates"):
            raise ValueError(
                "Dataset/candidate-pool problems have no continuous relaxation."
            )
        self.base = base
        self.variable_types = None  # fully continuous
        super().__init__(
            dim=base.dim,
            num_objectives=base.num_objectives,
            num_constraints=base.num_constraints,
            bounds=base.bounds,
            ref_point=base.ref_point,
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        values, constraints = self._base_eval(X)
        return constraints, values
