"""BNH constrained multi-objective synthetic test function (2-D, 2 objectives, 2 constraints).

Sources:
T. T. Binh and U. Korn. MOBES: A Multiobjective Evolution Strategy for Constrained Optimization Problems. In Proceedings of the Third International Conference on Genetic Algorithms (Mendel), pages 176-182, 1997.
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import ConstrainedMultiObjSyntheticProblem


class BNH(ConstrainedMultiObjSyntheticProblem):
    """Binh and Korn (2-D, 2 objectives, 2 constraints; feasible <= 0), negated to maximize."""

    available_dimensions = 2
    num_objectives = 2
    num_constraints = 2
    botorch_cls = _mo.BNH
    botorch_kwargs: dict = {}

    def __init__(self, dim: int | None = None) -> None:
        super().__init__(dim=dim)
        # BoTorch's BNH ``ref_point`` is [0, 0] (the ideal point). In the negated
        # maximization frame every feasible objective is <= 0, so the feasible
        # front never dominates [0, 0] and the hypervolume is identically 0. Use
        # the nadir of the feasible objective region instead (feasible obj mins are
        # ~[-135.8, -49.9]), matching the working problems' convention (e.g.
        # ConstrainedBraninCurrin's [-80, -12]), so HV is positive and comparable.
        self.ref_point = [-140.0, -55.0]
