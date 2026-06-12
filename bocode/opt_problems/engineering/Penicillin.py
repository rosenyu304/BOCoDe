"""Penicillin production simulator (multi-objective).

Sources:
Q. Liang and L. Lai. Scalable Bayesian optimization accelerates process optimization of penicillin production. NeurIPS 2021 AI for Science Workshop, 2021.
BoTorch implementation: M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

import botorch.test_functions.multi_objective as _mo

from ._botorch_wrapper import MultiObjBotorchProblem


class Penicillin(MultiObjBotorchProblem):
    """Penicillin production: maximise yield, minimise time and CO2 (3 obj, 7D)."""

    available_dimensions = 7
    num_objectives = 3
    num_constraints = 0

    def __init__(self) -> None:
        super().__init__(botorch_problem=_mo.Penicillin)
