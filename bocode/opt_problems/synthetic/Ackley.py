"""Ackley synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class Ackley(SingleObjSyntheticProblem):
    """Ackley function fixed to 10 dimensions (multimodal, single global optimum).

    NOTE: its optimum sits exactly at the center of the box; see ``AckleyShifted``.
    """

    available_dimensions = (2, 100)
    botorch_cls = _syn.Ackley
    botorch_kwargs = {"dim": 10}


class AckleyShifted(Ackley, ShiftedSingleObjSyntheticProblem):
    """Ackley with the optimum shifted off the box center (2/3 even dims, 1/3 odd dims).

    Same landscape and optimal value as ``Ackley``, translated so a center-biased
    initial design no longer starts at the answer. See
    :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the exact offset.
    """
