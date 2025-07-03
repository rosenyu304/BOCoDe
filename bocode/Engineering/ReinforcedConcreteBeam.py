import torch

from ..base import BenchmarkProblem, DataType


class ReinforcedConcreteBeam(BenchmarkProblem):
    """
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
    algorithm. Computers & Structures 89(23-24):2325–2336
    """

    # 3D objective, 2 constraints, X = n-by-3

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 2

    tags = {"single_objective", "constrained", "continuous", "3D"}

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=2,
            bounds=[(0.2, 15), (28, 40), (5, 10)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        gx = torch.zeros((n, self.num_constraints))

        As = X[:, 0]
        h = X[:, 1]
        b = X[:, 2]

        test_function = -(29.4 * As + 0.6 * b * h)
        fx = test_function.reshape(n, self.num_objectives)

        gx[:, 0] = h / b - 4
        gx[:, 1] = 180 + 7.35 * As * As / b - As * h

        return gx, fx
