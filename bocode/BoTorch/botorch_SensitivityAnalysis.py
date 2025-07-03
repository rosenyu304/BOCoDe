import botorch
import botorch.test_functions.sensitivity_analysis

from .BaseBotorch import BotorchProblem


class Ishigami(BotorchProblem):
    available_dimensions = botorch.test_functions.sensitivity_analysis.Ishigami().dim
    num_objectives = (
        botorch.test_functions.sensitivity_analysis.Ishigami().num_objectives
    )
    num_constraints = (
        botorch.test_functions.sensitivity_analysis.Ishigami().num_constraints
        if hasattr(
            botorch.test_functions.sensitivity_analysis.Ishigami(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.sensitivity_analysis.Ishigami
        )


class Gsobol(BotorchProblem):
    available_dimensions = [6, 8, 15]
    num_objectives = botorch.test_functions.sensitivity_analysis.Gsobol(
        dim=6
    ).num_objectives
    num_constraints = (
        botorch.test_functions.sensitivity_analysis.Gsobol(dim=6).num_constraints
        if hasattr(
            botorch.test_functions.sensitivity_analysis.Gsobol(dim=6), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.sensitivity_analysis.Gsobol, dim=6
        )


class Morris(BotorchProblem):
    available_dimensions = botorch.test_functions.sensitivity_analysis.Morris().dim
    num_objectives = botorch.test_functions.sensitivity_analysis.Morris().num_objectives
    num_constraints = (
        botorch.test_functions.sensitivity_analysis.Morris().num_constraints
        if hasattr(
            botorch.test_functions.sensitivity_analysis.Morris(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.sensitivity_analysis.Morris
        )
