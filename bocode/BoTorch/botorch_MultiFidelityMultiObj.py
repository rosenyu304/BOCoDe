import botorch
import botorch.test_functions.multi_objective_multi_fidelity

from .BaseBotorch import MultiObjBotorchProblem


class MOMFBraninCurrin(MultiObjBotorchProblem):
    available_dimensions = 3
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective_multi_fidelity.MOMFBraninCurrin().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective_multi_fidelity.MOMFBraninCurrin(),
            "num_constraints",
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective_multi_fidelity.MOMFBraninCurrin
        )


class MOMFPark1(MultiObjBotorchProblem):
    available_dimensions = 5
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective_multi_fidelity.MOMFPark().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective_multi_fidelity.MOMFPark(),
            "num_constraints",
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective_multi_fidelity.MOMFPark,
        )
