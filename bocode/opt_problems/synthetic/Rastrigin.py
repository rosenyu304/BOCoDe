"""Rastrigin synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class Rastrigin(SingleObjSyntheticProblem):
    """Rastrigin function fixed to 10 dimensions (regular highly multimodal lattice).

    NOTE: its optimum sits exactly at the center of the box; see ``RastriginShifted``.
    """

    available_dimensions = (2, 100)
    botorch_cls = _syn.Rastrigin
    botorch_kwargs = {"dim": 10}


class RastriginShifted(Rastrigin, ShiftedSingleObjSyntheticProblem):
    """Rastrigin with the optimum shifted off the box center (2/3 even dims, 1/3 odd dims).

    Same landscape and optimal value as ``Rastrigin``, translated so a center-biased
    initial design no longer starts at the answer. See
    :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the exact offset.
    """
