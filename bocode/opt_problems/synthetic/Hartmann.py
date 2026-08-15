"""Hartmann synthetic test function (6-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class Hartmann(SingleObjSyntheticProblem):
    """Hartmann function (6-D, 6 local minima, one global)."""

    available_dimensions = 6
    botorch_cls = _syn.Hartmann
    botorch_kwargs = {"dim": 6}


class HartmannShifted(Hartmann, ShiftedSingleObjSyntheticProblem):
    """Hartmann with the optimum moved to the 2/3 (even dims) / 1/3 (odd dims) point.

    Same landscape and optimal value as ``Hartmann``. See
    :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the exact offset.
    """
