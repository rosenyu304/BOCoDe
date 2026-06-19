"""DTLZ2 multi-objective synthetic test function (7-D, 3 objectives).

Sources:
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import MultiObjSyntheticProblem


class DTLZ2(MultiObjSyntheticProblem):
    """DTLZ2 (7-D inputs, 3 objectives; concave front), negated to maximize."""

    available_dimensions = 7
    num_objectives = 3
    botorch_cls = _mo.DTLZ2
    botorch_kwargs = {"dim": 7, "num_objectives": 3}
