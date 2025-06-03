import botorch.test_functions.multi_objective
import botorch.test_functions.multi_objective_multi_fidelity
import botorch.test_functions.sensitivity_analysis
from .BaseBotorch import BotorchProblem
import botorch
import math

class Ishigami(BotorchProblem):

    available_dimensions = botorch.test_functions.sensitivity_analysis.Ishigami().dim
    num_objectives = botorch.test_functions.sensitivity_analysis.Ishigami().num_objectives

    def __init__(self):
        super().__init__(botorch_problem=botorch.test_functions.sensitivity_analysis.Ishigami
                         )
        
class Gsobol(BotorchProblem):

    available_dimensions = [6, 8, 15]
    num_objectives = botorch.test_functions.sensitivity_analysis.Gsobol(dim=6).num_objectives

    def __init__(self):
        super().__init__(botorch_problem=botorch.test_functions.sensitivity_analysis.Gsobol, dim=6
                         )
        
class Morris(BotorchProblem):

    available_dimensions = botorch.test_functions.sensitivity_analysis.Morris().dim
    num_objectives = botorch.test_functions.sensitivity_analysis.Morris().num_objectives

    def __init__(self):
        super().__init__(botorch_problem=botorch.test_functions.sensitivity_analysis.Morris
                         )