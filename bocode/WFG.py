"""
Functions published by the Walking Fish Group (WFG)
Huband, S.; Hingston, P.; Barone, L.; While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation, vol.10, no.5, pp. 477-506.
"""

from typing import Tuple

import optproblems.wfg
import torch

from .base import BenchmarkProblem, DataType
from .exceptions import FunctionDefinitionAssertionError


class BaseWFG(BenchmarkProblem):
    input_type = DataType.CONTINUOUS
    available_dimensions = (2, None)
    num_objectives = (2, None)
    num_constraints = 0

    WFGProblem = None

    def __init__(self, dim: int, num_objectives: int = 2, k: int = None):
        """
        Optional Parameter k: The number of position related parameters (must be less than dim and a multiple of num_objectives-1). Default is 4 if num_objectives==2, otherwise 2*(num_objectives-1).
        Note: Some inputs for dim and num_objectives may not be valid for the specific WFG function.
        """
        if num_objectives > dim:
            raise FunctionDefinitionAssertionError(
                "Number of objectives must be less than or equal to dimension"
            )
        if k is None:
            if dim <= (4 if num_objectives == 2 else 2 * (num_objectives - 1)):
                k = num_objectives - 1
            else:
                k = 4 if num_objectives == 2 else 2 * (num_objectives - 1)

        self._specialCheck(dim, num_objectives, k)

        self.problem = self.__class__.WFGProblem(num_objectives, dim, k)

        super().__init__(
            dim=dim,
            num_objectives=num_objectives,
            num_constraints=0,
            bounds=[(0.0, 2.0 * (i + 1)) for i in range(dim)],
        )

    def _specialCheck(self, dim, num_objectives, k):
        """
        Override this method in subclasses to implement any special checks for the specific WFG function.
        """
        pass

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fx = torch.zeros(x.shape[0], self.num_objectives)

        for i, el in enumerate(x):
            f = self.problem(el)
            fx[i, :] = torch.tensor(f)

        return None, fx


class WFG1(BaseWFG):
    WFGProblem = optproblems.wfg.WFG1


class WFG2(BaseWFG):
    WFGProblem = optproblems.wfg.WFG2

    def _specialCheck(self, dim, num_objectives, k):
        if (dim - k) % 2 != 0:
            raise FunctionDefinitionAssertionError()

    def __init__(self, dim: int, num_objectives: int = 2, k: int = None):
        super().__init__(dim, num_objectives, k)


class WFG3(BaseWFG):
    WFGProblem = optproblems.wfg.WFG3

    def _specialCheck(self, dim, num_objectives, k):
        if (dim - k) % 2 != 0:
            raise FunctionDefinitionAssertionError()

    def __init__(self, dim: int, num_objectives: int = 2, k: int = None):
        super().__init__(dim, num_objectives, k)


class WFG4(BaseWFG):
    WFGProblem = optproblems.wfg.WFG4


class WFG5(BaseWFG):
    WFGProblem = optproblems.wfg.WFG5


class WFG6(BaseWFG):
    WFGProblem = optproblems.wfg.WFG6


class WFG7(BaseWFG):
    WFGProblem = optproblems.wfg.WFG7


class WFG8(BaseWFG):
    WFGProblem = optproblems.wfg.WFG8


class WFG9(BaseWFG):
    WFGProblem = optproblems.wfg.WFG9
