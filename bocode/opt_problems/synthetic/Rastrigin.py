"""Rastrigin synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import SingleObjSyntheticProblem


class Rastrigin(SingleObjSyntheticProblem):
    """Rastrigin function fixed to 10 dimensions (regular highly multimodal lattice)."""

    available_dimensions = (2, 100)
    botorch_cls = _syn.Rastrigin
    botorch_kwargs = {"dim": 10}
