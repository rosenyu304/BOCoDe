"""SRN constrained multi-objective synthetic test function (2-D, 2 objectives, 2 constraints).

Sources:
N. Srinivas and K. Deb. Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms. Evolutionary Computation, 2(3):221-248, 1994.
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import ConstrainedMultiObjSyntheticProblem


class SRN(ConstrainedMultiObjSyntheticProblem):
    """Srinivas and Deb (2-D, 2 objectives, 2 constraints; feasible <= 0), negated to maximize."""

    available_dimensions = 2
    num_objectives = 2
    num_constraints = 2
    botorch_cls = _mo.SRN
    botorch_kwargs: dict = {}

    def __init__(self, dim: int | None = None) -> None:
        super().__init__(dim=dim)
        # BoTorch's SRN ``ref_point`` is [0, 0] (the ideal point). In the negated
        # maximization frame the feasible front never dominates [0, 0] (objective 1
        # is always negative), so the hypervolume is identically 0. Use the nadir of
        # the feasible objective region instead (feasible obj mins are ~[-34.8,
        # -4.0]), matching the working problems' convention, so HV is positive and
        # comparable.
        self.ref_point = [-45.0, -12.0]
