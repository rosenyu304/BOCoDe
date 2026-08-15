"""Constrained smooth Hartmann synthetic test function (6-D, 1 constraint), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ConstrainedSingleObjSyntheticProblem


class ConstrainedHartmannSmooth(ConstrainedSingleObjSyntheticProblem):
    """Hartmann (6-D) with a single smooth norm constraint ``||x||^2 <= 1`` (feasible <= 0)."""

    available_dimensions = 6
    num_constraints = 1
    botorch_cls = _syn.ConstrainedHartmannSmooth
    botorch_kwargs = {"dim": 6}
