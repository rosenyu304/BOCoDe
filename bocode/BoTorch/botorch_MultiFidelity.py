import math

import botorch

from .BaseBotorch import BotorchProblem


class AugmentedBranin(BotorchProblem):
    available_dimensions = botorch.test_functions.multi_fidelity.AugmentedBranin().dim
    num_objectives = (
        botorch.test_functions.multi_fidelity.AugmentedBranin().num_objectives
    )
    num_constraints = (
        botorch.test_functions.multi_fidelity.AugmentedBranin().num_constraints
        if hasattr(
            botorch.test_functions.multi_fidelity.AugmentedBranin(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_fidelity.AugmentedBranin,
            optimum=[0.397887, 0.397887, 0.397887],
            x_opt=[
                [math.pi, 1.3867356039019576, 0.1],
                [math.pi, 1.781519779945532, 0.5],
            ],
        )


class AugmentedHartmann(BotorchProblem):
    available_dimensions = botorch.test_functions.multi_fidelity.AugmentedHartmann().dim
    num_objectives = (
        botorch.test_functions.multi_fidelity.AugmentedHartmann().num_objectives
    )
    num_constraints = (
        botorch.test_functions.multi_fidelity.AugmentedHartmann().num_constraints
        if hasattr(
            botorch.test_functions.multi_fidelity.AugmentedHartmann(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_fidelity.AugmentedHartmann,
            optimum=[-3.32237],
            x_opt=[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0]],
        )


class AugmentedRosenbrock(BotorchProblem):
    available_dimensions = (
        botorch.test_functions.multi_fidelity.AugmentedRosenbrock().dim
    )
    num_objectives = (
        botorch.test_functions.multi_fidelity.AugmentedRosenbrock().num_objectives
    )
    num_constraints = (
        botorch.test_functions.multi_fidelity.AugmentedRosenbrock().num_constraints
        if hasattr(
            botorch.test_functions.multi_fidelity.AugmentedRosenbrock(),
            "num_constraints",
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_fidelity.AugmentedRosenbrock,
            optimum=[0.0],
            x_opt=[[1.0, 1.0, 1.0]],
        )
