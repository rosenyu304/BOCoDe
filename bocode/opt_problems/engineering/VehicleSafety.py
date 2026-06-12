"""Vehicle crash-worthiness multi-objective engineering design problem.

Sources:
X. Liao, Q. Li, X. Yang, W. Zhang, and W. Li. Multiobjective optimization for crash safety design of vehicles using stepwise regression model. Structural and Multidisciplinary Optimization 35(6):561-569, 2008.
R. Tanabe and H. Ishibuchi. An easy-to-use real-world multi-objective optimization problem suite. Applied Soft Computing 89:106078, 2020.
BoTorch implementation: M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

import botorch.test_functions.multi_objective as _mo

from ._botorch_wrapper import MultiObjBotorchProblem


class VehicleSafety(MultiObjBotorchProblem):
    """Vehicle crash safety: 3 objectives, unconstrained, 5D."""

    available_dimensions = 5
    num_objectives = 3
    num_constraints = 0

    def __init__(self) -> None:
        super().__init__(botorch_problem=_mo.VehicleSafety)
