from ..base import BenchmarkProblem, DataType


class GearTrain(BenchmarkProblem):
    """
    Sandgren, E. (1990). Nonlinear Integer and Discrete Programming in Mechanical Design Optimization."
    ASME. J. Mech. Des. June 1990; 112(2): 223–229.
    """

    available_dimensions = 4
    input_type = DataType.MIXED
    num_objectives = 1
    num_constraints = 0

    # 4D objective, 0 constraints, X = n-by-4

    tags = {"single_objective", "unconstrained", "mixed", "4D"}

    def __init__(self, is_discrete=True):
        self.is_discrete = is_discrete

        if is_discrete:
            # bounds is an array of set from 12 to 60
            bounds = [set(range(12, 61))] * 4
        else:
            bounds = [(0, 1)] * 4

        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            bounds=bounds,
        )

    def _evaluate_implementation(self, X):
        # X = super().scale(X, to_verify)

        fx = -((1 / 6.931 - (X[:, 0] * X[:, 1]) / (X[:, 2] * X[:, 3])) ** 2).reshape(
            -1, 1
        )

        return None, fx
