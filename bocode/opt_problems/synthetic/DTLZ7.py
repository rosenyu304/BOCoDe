"""DTLZ7 multi-objective synthetic test function (22-D, 3 objectives; disconnected front).

Sources:
K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. Scalable Test Problems for
Evolutionary Multiobjective Optimization. In Evolutionary Multiobjective Optimization,
Springer, 2005, pp. 105-145. https://doi.org/10.1007/1-84628-137-7_6
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py

The Pareto front consists of 2^(M-1) disconnected regions (4 for M = 3), so a method
must keep several separate pockets of the front alive at once. Paper-standard size
k = 20 distance variables (n = M - 1 + 20 = 22 for M = 3). Reference point [-15]*M is
BoTorch's (``DTLZ7._ref_val = 15``).
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import MultiObjSyntheticProblem


class DTLZ7(MultiObjSyntheticProblem):
    """DTLZ7 (22-D inputs, 3 objectives; 4 disconnected regions), negated to maximize."""

    available_dimensions = (4, 100)
    num_objectives = 3
    botorch_cls = _mo.DTLZ7
    botorch_kwargs = {"dim": 22, "num_objectives": 3}
