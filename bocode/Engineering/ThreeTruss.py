import torch

from ..base import BenchmarkProblem, DataType


class ThreeTruss(BenchmarkProblem):
    """
    Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for global engineering optimization.
    Engineering computations 29(5):464–483
    """

    # 2D objective, 3 constraints, X = n-by-2

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 3

    tags = {"single_objective", "constrained", "2D"}

    def __init__(self):
        super().__init__(
            dim=2, num_objectives=1, num_constraints=3, bounds=[(0, 1), (0, 1)]
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        for i in range(n):
            for j in range(2):
                if X[i, j] <= 1e-5:
                    X[i, j] = 1e-5

        gx = torch.zeros((n, self.num_constraints))

        x1 = X[:, 0]
        x2 = X[:, 1]

        L = 100
        P = 2
        sigma = 2

        test_function = -(2 * 2**0.5 * x1 + x2) * L
        fx = test_function.reshape(n, self.num_objectives)

        gx[:, 0] = (2**0.5 * x1 + x2) / (2**0.5 * x1 * x1 + 2 * x1 * x2) * P - sigma
        gx[:, 1] = (x2) / (2**0.5 * x1 * x1 + 2 * x1 * x2) * P - sigma
        gx[:, 2] = (1) / (x1 + 2**0.5 * x2) * P - sigma

        return gx, fx
