import torch

from ..base import BenchmarkProblem, DataType


class KeaneBump(BenchmarkProblem):
    """
    Keane A (1994) Experiences with optimizers instructural design. In: Proceedings of the conference
    on adaptive computing in engineering design and control, pp 14–27
    """

    # N-D objective, 2 constraints, X = n-by-dim

    available_dimensions = (5, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 2

    tags = {"single_objective", "constrained", "continuous", "ND"}

    def __init__(self, dim=18):
        super().__init__(
            dim, num_objectives=1, num_constraints=2, bounds=[(0, 10)] * dim
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)
        dim = X.shape[1]

        fx = torch.zeros(n, 1).to(torch.float64)
        gx1 = torch.zeros(n, 1).to(torch.float64)
        gx2 = torch.zeros(n, 1).to(torch.float64)

        for i in range(n):
            x = X[i, :]

            cos4 = 0
            cos2 = 1
            sq_denom = 0

            pi_sum = 1
            sigma_sum = 0

            for j in range(dim):
                cos4 += torch.cos(x[j]) ** 4
                cos2 *= torch.cos(x[j]) ** 2
                sq_denom += (j + 1) * (x[j]) ** 2

                pi_sum *= x[j]
                sigma_sum += x[j]

            test_function = torch.abs((cos4 - 2 * cos2) / torch.sqrt(sq_denom))
            fx[i] = test_function

            gx1[i] = 0.75 - pi_sum
            gx2[i] = sigma_sum - 7.5 * dim

        gx = torch.cat((gx1, gx2), 1)
        return gx, fx
