"""ZDT2 multi-objective synthetic test function (6-D, 2 objectives).

Sources:
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import MultiObjSyntheticProblem


class ZDT2(MultiObjSyntheticProblem):
    """ZDT2 (6-D inputs, 2 objectives; concave front), negated to maximize."""

    available_dimensions = 6
    num_objectives = 2
    botorch_cls = _mo.ZDT2
    botorch_kwargs = {"dim": 6, "num_objectives": 2}
