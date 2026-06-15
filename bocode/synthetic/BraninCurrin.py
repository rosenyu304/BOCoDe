"""Branin-Currin multi-objective synthetic test function (2-D, 2 objectives).

Sources:
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import MultiObjSyntheticProblem


class BraninCurrin(MultiObjSyntheticProblem):
    """Branin-Currin (2-D inputs, 2 objectives), negated to maximize."""

    available_dimensions = 2
    num_objectives = 2
    botorch_cls = _mo.BraninCurrin
    botorch_kwargs = {}
