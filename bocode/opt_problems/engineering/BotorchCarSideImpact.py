"""Car side-impact problem, 4-objective BoTorch formulation.

Sources:
H. Jain and K. Deb. An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II. IEEE Transactions on Evolutionary Computation 18(4):602-622, 2014.
R. Tanabe and H. Ishibuchi. An easy-to-use real-world multi-objective optimization problem suite. Applied Soft Computing 89:106078, 2020.
BoTorch implementation: M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

import botorch.test_functions.multi_objective as _mo

from ._botorch_wrapper import MultiObjBotorchProblem


class BotorchCarSideImpact(MultiObjBotorchProblem):
    """Car side impact (BoTorch 4-objective variant), unconstrained, 7D.

    Distinct from :class:`bocode.opt_problems.engineering.CarSideImpact.CarSideImpact`,
    which is the 2-objective / 10-constraint formulation.
    """

    available_dimensions = 7
    num_objectives = 4
    num_constraints = 0

    def __init__(self) -> None:
        super().__init__(botorch_problem=_mo.CarSideImpact)
