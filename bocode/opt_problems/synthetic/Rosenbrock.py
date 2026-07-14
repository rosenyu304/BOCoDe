"""Rosenbrock synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ShiftedSingleObjSyntheticProblem, SingleObjSyntheticProblem


class Rosenbrock(SingleObjSyntheticProblem):
    """Rosenbrock function fixed to 10 dimensions (narrow curved valley)."""

    available_dimensions = (2, 100)
    botorch_cls = _syn.Rosenbrock
    botorch_kwargs = {"dim": 10}


class RosenbrockShifted(Rosenbrock, ShiftedSingleObjSyntheticProblem):
    """Rosenbrock with the optimum moved to the 2/3 (even dims) / 1/3 (odd dims) point.

    Same landscape and optimal value as ``Rosenbrock``. See
    :class:`~._wrapper.ShiftedSingleObjSyntheticProblem` for the exact offset.
    """
