"""Branin synthetic test function (2-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import SingleObjSyntheticProblem


class Branin(SingleObjSyntheticProblem):
    """Branin function (2-D, three global optima)."""

    available_dimensions = 2
    botorch_cls = _syn.Branin
    botorch_kwargs = {}
