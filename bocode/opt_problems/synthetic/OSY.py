"""OSY constrained multi-objective synthetic test function (6-D, 2 objectives, 6 constraints).

Sources:
A. Osyczka and S. Kundu. A New Method to Solve Generalized Multicriteria Optimization Problems Using the Simple Genetic Algorithm. Structural Optimization, 10(2):94-99, 1995.
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import ConstrainedMultiObjSyntheticProblem


class OSY(ConstrainedMultiObjSyntheticProblem):
    """Osyczka and Kundu (6-D, 2 objectives, 6 constraints; feasible <= 0), negated to maximize."""

    available_dimensions = 6
    num_objectives = 2
    num_constraints = 6
    botorch_cls = _mo.OSY
    botorch_kwargs: dict = {}
