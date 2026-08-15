"""Holder Table synthetic test function (2-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import SingleObjSyntheticProblem


class HolderTable(SingleObjSyntheticProblem):
    """Holder Table function (2-D, four symmetric global minima)."""

    available_dimensions = 2
    botorch_cls = _syn.HolderTable
    botorch_kwargs: dict = {}
