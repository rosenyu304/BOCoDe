"""Michalewicz synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import SingleObjSyntheticProblem


class Michalewicz(SingleObjSyntheticProblem):
    """Michalewicz function fixed to 10 dimensions (multimodal, steep valleys)."""

    available_dimensions = (2, 500)
    botorch_cls = _syn.Michalewicz
    botorch_kwargs = {"dim": 10}
