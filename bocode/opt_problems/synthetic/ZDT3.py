"""ZDT3 multi-objective synthetic test function (6-D, 2 objectives).

Sources:
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import MultiObjSyntheticProblem


class ZDT3(MultiObjSyntheticProblem):
    """ZDT3 (6-D inputs, 2 objectives; disconnected front), negated to maximize."""

    available_dimensions = (2, 100)
    num_objectives = 2
    botorch_cls = _mo.ZDT3
    botorch_kwargs = {"dim": 6, "num_objectives": 2}
