import torch

from ..base import BenchmarkProblem, DataType


class BaseOpfunuCEC(BenchmarkProblem):
    """
    Base class for CEC benchmarks.
    """

    problem = None
    available_dimensions = None
    input_type = DataType.CONTINUOUS
    num_objectives = 1

    def __init__(self, dim=None, num_objectives=1, num_constraints=0):
        if dim is None:
            dim = self.__class__.problem().ndim

        bounds = self.__class__.problem(ndim=dim).bounds
        bounds = [tuple(bound) for bound in bounds]

        self.problem = self.__class__.problem(ndim=dim)
        super().__init__(
            dim=dim,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            bounds=bounds,
            optimum=[-self.problem.f_global],
            x_opt=[self.problem.x_global],
        )

    def _evaluate_implementation(self, x: torch.Tensor):
        """
        Evaluate the CEC benchmark function.
        """
        fx = torch.zeros(x.shape[0], self.num_objectives)

        for i, el in enumerate(x):
            f = self.problem.evaluate(el.tolist())
            fx[i, :] = torch.Tensor([f])

        return None, -fx
