from ...base import BenchmarkProblem


class GearTrain(BenchmarkProblem):
    """
    Sandgren, E. (1990). Nonlinear Integer and Discrete Programming in Mechanical Design Optimization."
    ASME. J. Mech. Des. June 1990; 112(2): 223–229.

    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly algorithm. Computers & Structures 89(23-24):2325–2336.

    Minimize the gear-ratio error of a compound gear train; the four variables are
    gear-teeth counts in [12, 60]. ``is_discrete=True`` (default) gives the original
    mixed-integer formulation (teeth are integers); ``is_discrete=False`` gives the
    continuous relaxation used by some firefly-algorithm studies.
    """

    available_dimensions = 4
    num_objectives = 1
    num_constraints = 0

    # 4D objective, 0 constraints, X = n-by-4

    tags = {"single_objective", "unconstrained", "mixed", "4D"}

    def __init__(self, is_discrete: bool = True):
        self.is_discrete = is_discrete
        # Teeth counts live in [12, 60]; the integer/continuous distinction is
        # carried by variable_types, not by the bounds (so scale()/sample() work).
        self.variable_types = ["integer"] * 4 if is_discrete else None

        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            bounds=[(12, 60)] * 4,
        )

    def _evaluate_implementation(self, X):
        fx = -((1 / 6.931 - (X[:, 0] * X[:, 1]) / (X[:, 2] * X[:, 3])) ** 2).reshape(
            -1, 1
        )

        return None, fx
