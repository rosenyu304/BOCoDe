"""CONSTR constrained multi-objective synthetic test function (2-D, 2 objectives, 2 constraints).

Sources:
K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2):182-197, 2002.
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import ConstrainedMultiObjSyntheticProblem


class CONSTR(ConstrainedMultiObjSyntheticProblem):
    """CONSTR (2-D, 2 objectives, 2 constraints; feasible <= 0), negated to maximize."""

    available_dimensions = 2
    num_objectives = 2
    num_constraints = 2
    botorch_cls = _mo.CONSTR
    botorch_kwargs: dict = {}
