"""C2-DTLZ2 constrained multi-objective synthetic test function (12-D, 2 objectives, 1 constraint).

Sources:
K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. Scalable Test Problems for Evolutionary Multiobjective Optimization. In Evolutionary Multiobjective Optimization, pages 105-145. Springer, 2005.
H. Jain and K. Deb. An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point Based Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive Approach. IEEE Transactions on Evolutionary Computation, 18(4):602-622, 2014.
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import ConstrainedMultiObjSyntheticProblem


class C2DTLZ2(ConstrainedMultiObjSyntheticProblem):
    """C2-DTLZ2 (12-D inputs, 2 objectives, 1 constraint; feasible <= 0), negated to maximize."""

    available_dimensions = (3, 100)
    num_objectives = 2
    num_constraints = 1
    botorch_cls = _mo.C2DTLZ2
    botorch_kwargs = {"dim": 12, "num_objectives": 2}
