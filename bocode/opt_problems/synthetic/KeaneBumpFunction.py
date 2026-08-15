"""Keane bump synthetic test function (10-D, 2 constraints), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ConstrainedSingleObjSyntheticProblem


class KeaneBumpFunction(ConstrainedSingleObjSyntheticProblem):
    """Keane bump function (10-D, multimodal) with 2 inequality constraints.

    Constraints (feasible <= 0): product of coordinates >= 0.75 and their sum
    <= 7.5 * dim / 2.
    """

    available_dimensions = 10
    num_constraints = 2
    botorch_cls = _syn.KeaneBumpFunction
    botorch_kwargs = {"dim": 10}
