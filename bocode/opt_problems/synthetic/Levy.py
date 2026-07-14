"""Levy synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class Levy(SingleObjSyntheticProblem):
    """Levy function fixed to 10 dimensions (multimodal).

    NOTE: the box center beats 20k random samples here; see ``LevyShifted``.
    """

    available_dimensions = (2, 100)
    botorch_cls = _syn.Levy
    botorch_kwargs = {"dim": 10}


class LevyShifted(Levy, ShiftedSingleObjSyntheticProblem):
    """Levy with the optimum shifted (2/3 point on even dims, 1/3 point on odd dims).

    Same landscape and optimal value as ``Levy``, translated so a center-biased initial
    design no longer starts near the answer. See
    :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the exact offset.
    """
