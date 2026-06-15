"""Rosenbrock synthetic test function (10-D), negated to maximize.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import SingleObjSyntheticProblem


class Rosenbrock(SingleObjSyntheticProblem):
    """Rosenbrock function fixed to 10 dimensions (narrow curved valley)."""

    available_dimensions = 10
    botorch_cls = _syn.Rosenbrock
    botorch_kwargs = {"dim": 10}
