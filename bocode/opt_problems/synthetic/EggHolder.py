"""Egg Holder synthetic test function (2-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import SingleObjSyntheticProblem


class EggHolder(SingleObjSyntheticProblem):
    """Egg Holder function (2-D, highly multimodal with many deep local minima)."""

    available_dimensions = 2
    botorch_cls = _syn.EggHolder
    botorch_kwargs: dict = {}
