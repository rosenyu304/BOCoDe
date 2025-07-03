import torch

from ..base import BenchmarkProblem, DataType


class CompressionSpring(BenchmarkProblem):
    """
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly algorithm.
    Computers & Structures 89(23-24):2325–2336
    """

    # 3D objective, 4 constraints, X = n-by-3

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 4

    tags = {"single_objective", "constrained", "3D"}

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=4,
            bounds=[(0.05, 1), (0.25, 1.3), (2, 15)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        gx = torch.zeros((n, self.num_constraints))

        d = X[:, 0]
        D = X[:, 1]
        N = X[:, 2]

        test_function = -((N + 2) * D * d**2)
        fx = test_function.reshape(n, self.num_objectives)

        gx[:, 0] = 1 - (D * D * D * N / (71785 * d * d * d * d))
        gx[:, 1] = (
            (4 * D * D - D * d) / (12566 * (D * d * d * d - d * d * d * d))
            + 1 / (5108 * d * d)
            - 1
        )
        gx[:, 2] = 1 - 140.45 * d / (D * D * N)
        gx[:, 3] = (D + d) / 1.5 - 1

        return gx, fx
