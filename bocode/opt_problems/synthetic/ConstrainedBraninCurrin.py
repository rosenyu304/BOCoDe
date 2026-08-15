"""Constrained Branin-Currin multi-objective synthetic test function (2-D, 2 objectives, 1 constraint).

Sources:
S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/2006.05078
BoTorch multi-objective test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py
"""

import botorch.test_functions.multi_objective as _mo

from ._wrapper import ConstrainedMultiObjSyntheticProblem


class ConstrainedBraninCurrin(ConstrainedMultiObjSyntheticProblem):
    """Branin-Currin (2-D, 2 objectives) with a single disk constraint (feasible <= 0), negated to maximize."""

    available_dimensions = 2
    num_objectives = 2
    num_constraints = 1
    botorch_cls = _mo.ConstrainedBraninCurrin
    botorch_kwargs: dict = {}
