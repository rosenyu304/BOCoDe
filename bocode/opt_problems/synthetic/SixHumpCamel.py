"""Six-Hump Camel synthetic test function (2-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class SixHumpCamel(SingleObjSyntheticProblem):
    """Six-Hump Camel function (2-D, six local minima, two global)."""

    available_dimensions = 2
    botorch_cls = _syn.SixHumpCamel
    botorch_kwargs: dict = {}


class SixHumpCamelShifted(SixHumpCamel, ShiftedSingleObjSyntheticProblem):
    """Six-Hump Camel with its first global optimum at the (2/3, 1/3) point of the box.

    Same landscape and optimal value as ``SixHumpCamel``; both global optima stay inside
    the box. See :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the offset.
    """
