"""Branin synthetic test function (2-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class Branin(SingleObjSyntheticProblem):
    """Branin function (2-D, three global optima)."""

    available_dimensions = 2
    botorch_cls = _syn.Branin
    botorch_kwargs = {}


class BraninShifted(Branin, ShiftedSingleObjSyntheticProblem):
    """Branin with its first global optimum moved to the (2/3, 1/3) point of the box.

    Same landscape and optimal value as ``Branin``. The shift is anchored on the first
    of Branin's three global optima; the other two translate by the same offset and
    fall outside the box, so the shifted problem has a single global optimum inside it.
    See :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the exact offset.
    """
