"""Constrained Gramacy synthetic test function (2-D, 2 constraints), negated to maximize.

Sources:
R. B. Gramacy, G. A. Gray, S. Le Digabel, H. K. H. Lee, P. Ranjan, G. Wells, and S. M. Wild. Modeling an Augmented Lagrangian for Blackbox Constrained Optimization. Technometrics, 58(1):1-11, 2016. http://arxiv.org/abs/1403.4890
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

import botorch.test_functions.synthetic as _syn

from ._wrapper import ConstrainedSingleObjSyntheticProblem


class ConstrainedGramacy(ConstrainedSingleObjSyntheticProblem):
    """Gramacy (2-D) with two nonlinear constraints (feasible <= 0), negated to maximize."""

    available_dimensions = 2
    num_constraints = 2
    botorch_cls = _syn.ConstrainedGramacy
    botorch_kwargs: dict = {}
