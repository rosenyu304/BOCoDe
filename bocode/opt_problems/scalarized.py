"""Penalty-scalarization wrappers.

Each wrapper turns an underlying *constrained* (and possibly multi-objective)
BoCoDe problem into a single-objective, unconstrained, continuous problem by
(1) collapsing all objectives to their equal-weight mean and (2) folding the
total positive constraint violation into that scalar as a penalty (the same
``sum(max(c, 0))`` convention as ``algorithms._bo_utils.penalized``).

The wrapper reports ``num_objectives=1`` and ``num_constraints=0`` so the
single-objective, continuous, unconstrained solver paths (and the plotting
categorization) treat these as SO-continuous-unconstrained problems.

The 42 named subclasses at the bottom of this module (``<BaseName>_PS``) each
hardwire one base problem. The base class is resolved *lazily* at instantiation
time via ``bocode.get_problem`` so importing this module never pulls in the
optional dependencies (modact, mazda, ...) of bases that are not used.
"""

from __future__ import annotations

import torch

import bocode

from ..base import BenchmarkProblem


# Fixed per-objective scales for MULTI-objective base problems, used to
# normalize each objective before the equal-weight mean so no single
# large-magnitude objective dominates. Each value is the per-objective std of
# the base problem's values over a fixed sample (torch.rand, seed=0, in the
# base's bounds; 128 points, 32 for the slow simulator problems Mazda/Mazda_SCA),
# floored at 1e-8. These are deterministic problem constants, NOT computed from
# the data a run observes. Single-objective (and any un-baked) problems default
# to all-ones, leaving their scalar identical to the plain equal-weight mean.
_OBJ_SCALE = {
    "Mazda": [0.043776470595867735, 0.9755064854862865, 0.02844620171752701, 0.026475858114910756, 0.02404480081325757],
    "Mazda_SCA": [0.03616851590877216, 2.462419148828268, 0.025541939770907903, 0.025417065644670056],
    "CRE21": [148.2503457009819, 3.5849138510621255],
    "CS3": [0.18645227828290067, 196.45089545573978],
    "CS4": [0.18645227828290067, 196.45089545573978],
    "CT3": [0.18645227828290067, 0.43459715481499955],
    "CT4": [0.18645227828290067, 0.43459715481499955],
    "CTS3": [0.18645227828290067, 0.43459715481499955, 196.45089545573978],
    "CTS4": [0.18645227828290067, 0.43459715481499955, 196.45089545573978],
    "CTSE4": [0.18645227828290067, 0.43459715481499955, 196.45089545573978, 0.02001292255342247],
    "CTSEI3": [0.18645227828290067, 0.43459715481499955, 196.45089545573978, 0.02001292255342247, 23.067607251410823],
    "CTSEI4": [0.18645227828290067, 0.43459715481499955, 196.45089545573978, 0.02001292255342247, 23.067607251410823],
}


class PenaltyScalarizedProblem(BenchmarkProblem):
    """Present a constrained (multi-)objective base problem as SO-unconstrained.

    Subclasses set ``_base_name`` to the registry name of the base problem.
    """

    # Class-level attributes required by ``BenchmarkProblem.__new__``'s
    # "fully implemented" check (they must be non-None). ``dim`` is set per
    # instance in ``__init__`` from the resolved base problem.
    _base_name: str | None = None
    available_dimensions = 0
    num_objectives = 1
    num_constraints = 0

    tags = {"single_objective", "continuous", "unconstrained", "scalarized"}

    def __init__(self) -> None:
        base = bocode.get_problem(self._base_name)()
        self._base = base
        # Fixed per-objective scale used to normalize each objective before the
        # equal-weight mean. Defaults to all-ones (single-objective and un-baked
        # problems are unchanged); baked MO problems get their precomputed stds.
        self._obj_scale = torch.tensor(
            _OBJ_SCALE.get(self._base_name, [1.0] * base.num_objectives),
            dtype=torch.float64,
        )
        super().__init__(
            dim=base.dim,
            num_objectives=1,
            num_constraints=0,
            bounds=base.bounds,
            ref_point=None,
        )
        # Mirror the base's per-dimension variable types so the wrapper reports
        # the same dim/type layout (no-op for fully-continuous bases).
        base_vt = getattr(base, "variable_types", None)
        if base_vt is not None:
            self.variable_types = list(base_vt)

    def evaluate(self, X):
        # Delegate to the base's PUBLIC evaluate() to get its normalized /
        # validated output: values (n, m) in the maximization frame, constraints
        # (n, k) feasible when <= 0.
        base_values, base_cons = self._base.evaluate(X)
        # Normalize each objective by its fixed per-objective scale before the
        # equal-weight mean so no single large-magnitude objective dominates.
        scaled = base_values / self._obj_scale.to(base_values)
        scalar = scaled.mean(dim=1, keepdim=True)  # equal-weight mean
        if base_cons is not None and base_cons.numel() > 0:
            scalar = scalar - base_cons.clamp(min=0).sum(dim=1, keepdim=True)
        return scalar, None


# The 42 base problems to wrap (registry names). The scalarized name is
# "<name>_PS".
_BASE_NAMES = [
    # 12 multi-objective-constrained
    "CRE21",
    "CS3",
    "CS4",
    "CT3",
    "CT4",
    "CTS3",
    "CTS4",
    "CTSE4",
    "CTSEI3",
    "CTSEI4",
    "Mazda",
    "Mazda_SCA",
    # 30 single-objective-constrained
    "HeatExchanger",
    "CEC2020_p1",
    "CEC2020_p2",
    "CEC2020_p3",
    "CEC2020_p4",
    "CEC2020_p5",
    "CEC2020_p6",
    "CEC2020_p7",
    "CEC2020_p11",
    "CEC2020_p34",
    "CEC2020_p35",
    "CEC2020_p36",
    "CEC2020_p37",
    "CEC2020_p38",
    "CEC2020_p39",
    "CEC2020_p40",
    "CEC2020_p41",
    "CEC2020_p42",
    "CEC2020_p43",
    "CEC2020_p45",
    "CEC2020_p46",
    "CEC2020_p47",
    "CEC2020_p48",
    "CEC2020_p49",
    "CEC2020_p50",
    "CEC2020_p51",
    "CEC2020_p52",
    "CEC2020_p53",
    "CEC2020_p54",
    "CEC2020_p55",
    "CEC2020_p56",
    "CEC2020_p57",
]


def _make_scalarized(base_name: str) -> type[PenaltyScalarizedProblem]:
    """Build a ``PenaltyScalarizedProblem`` subclass hardwired to ``base_name``."""
    return type(
        f"{base_name}_PS",
        (PenaltyScalarizedProblem,),
        {"_base_name": base_name},
    )


# Create each subclass as a real, importable module-level attribute so the
# registry (which does ``getattr(module, "<name>_PS")``) can find it.
for _name in _BASE_NAMES:
    globals()[f"{_name}_PS"] = _make_scalarized(_name)

del _name
