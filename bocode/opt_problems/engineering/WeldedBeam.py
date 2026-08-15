"""Welded Beam multi-objective engineering design problem.

Sources:
K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2):182-197, 2002.
BoTorch implementation: M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

import botorch.test_functions.multi_objective as _mo

from ._botorch_wrapper import MultiObjBotorchProblem


class WeldedBeam(MultiObjBotorchProblem):
    """Welded beam design: minimise cost and end deflection (2 obj, 4 constraints, 4D)."""

    available_dimensions = 4
    num_objectives = 2
    num_constraints = 4

    def __init__(self) -> None:
        super().__init__(botorch_problem=_mo.WeldedBeam)
