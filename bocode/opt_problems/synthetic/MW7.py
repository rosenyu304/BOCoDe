"""MW7 constrained multi-objective synthetic test function (2-D, 2 objectives, 2 constraints).

Sources:
Z. Ma and Y. Wang. Evolutionary Constrained Multiobjective Optimization: Test Suite Construction and Performance Comparisons. IEEE Transactions on Evolutionary Computation, 23(6):972-986, 2019.
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import ConstrainedMultiObjSyntheticProblem


class MW7(ConstrainedMultiObjSyntheticProblem):
    """MW7 (2-D, 2 objectives, 2 constraints; feasible <= 0), negated to maximize."""

    available_dimensions = (2, 100)
    num_objectives = 2
    num_constraints = 2
    botorch_cls = _mo.MW7
    botorch_kwargs = {"dim": 2}
