"""Dixon-Price synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class DixonPrice(SingleObjSyntheticProblem):
    """Dixon-Price function fixed to 10 dimensions (curved valley, ill-conditioned).

    NOTE: the box center beats 20k random samples here; see ``DixonPriceShifted``.
    """

    available_dimensions = (2, 100)
    botorch_cls = _syn.DixonPrice
    botorch_kwargs = {"dim": 10}


class DixonPriceShifted(DixonPrice, ShiftedSingleObjSyntheticProblem):
    """Dixon-Price with the optimum shifted (2/3 point on even dims, 1/3 on odd dims).

    Same landscape and optimal value as ``DixonPrice``, translated so a center-biased
    initial design no longer starts near the answer. See
    :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the exact offset.
    """
