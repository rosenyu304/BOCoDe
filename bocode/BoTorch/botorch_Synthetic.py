import botorch
import botorch.test_functions.synthetic

from .BaseBotorch import BotorchProblem


class Ackley(BotorchProblem):
    """
    Eriksson D, Poloczek M (2021) Scalable constrained bayesian optimization.
    In: International Conference on Artificial Intelligence and Statistics, PMLR, pp 730–738
    """

    available_dimensions = botorch.test_functions.synthetic.Ackley().dim
    num_objectives = botorch.test_functions.synthetic.Ackley().num_objectives
    num_constraints = (
        botorch.test_functions.synthetic.Ackley().num_constraints
        if hasattr(botorch.test_functions.synthetic.Ackley(), "num_constraints")
        else 0
    )

    def __init__(self):
        super().__init__(botorch_problem=botorch.test_functions.synthetic.Ackley)
