"""Disc Brake multi-objective engineering design problem.

Sources:
R. Tanabe and H. Ishibuchi. An easy-to-use real-world multi-objective optimization problem suite. Applied Soft Computing 89:106078, 2020.
BoTorch implementation: M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

import botorch.test_functions.multi_objective as _mo

from ._botorch_wrapper import MultiObjBotorchProblem


class DiscBrake(MultiObjBotorchProblem):
    """Disc brake design: minimise mass and stopping time (2 obj, 4 constraints, 4D)."""

    available_dimensions = 4
    num_objectives = 2
    num_constraints = 4

    def __init__(self) -> None:
        super().__init__(botorch_problem=_mo.DiscBrake)
