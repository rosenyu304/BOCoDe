"""
Functions published by the Deb, Thiele, Laumanns, Zitzler (DTLZ)

(1) Deb, K.; Thiele, L.; Laumanns, M.; Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization, Technical Report, Computer Engineering and Networks Laboratory (TIK), Swiss Federal Institute of Technology (ETH). https://dx.doi.org/10.3929/ethz-a-004284199
(2) Deb, K.; Thiele, L.; Laumanns, M.; Zitzler, E. (2002). Scalable multi-objective optimization test problems, Proceedings of the IEEE Congress on Evolutionary Computation, pp. 825-830
"""

from typing import Tuple

import optproblems.dtlz
import torch

from .base import BenchmarkProblem, DataType
from .exceptions import FunctionDefinitionAssertionError


class BaseDTLZ(BenchmarkProblem):
    available_dimensions = (2, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (2, None)
    num_constraints = 0

    DTLZProblem = None

    def __init__(self, dim: int, num_objectives: int = 2):
        """
        Optional Parameter k: The number of position related parameters (must be less than dim and a multiple of num_objectives-1). Default is 4 if num_objectives==2, otherwise 2*(num_objectives-1).
        Note: Some inputs for dim and num_objectives may not be valid for the specific DTLZ function.
        """
        if num_objectives >= dim:
            raise FunctionDefinitionAssertionError(
                "Number of objectives must be less than dim"
            )

        self.problem = self.DTLZProblem(num_objectives, dim)

        super().__init__(
            dim=dim,
            num_objectives=num_objectives,
            num_constraints=0,
            bounds=[(0, 1)] * dim,
        )

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fx = torch.zeros(x.shape[0], self.num_objectives)

        for i, el in enumerate(x):
            f = self.problem(el)
            fx[i, :] = torch.tensor(f)

        return None, fx


class DTLZ1(BaseDTLZ):
    DTLZProblem = optproblems.dtlz.DTLZ1


class DTLZ2(BaseDTLZ):
    DTLZProblem = optproblems.dtlz.DTLZ2


class DTLZ3(BaseDTLZ):
    DTLZProblem = optproblems.dtlz.DTLZ3


class DTLZ4(BaseDTLZ):
    DTLZProblem = optproblems.dtlz.DTLZ4


class DTLZ5(BaseDTLZ):
    DTLZProblem = optproblems.dtlz.DTLZ5


class DTLZ6(BaseDTLZ):
    DTLZProblem = optproblems.dtlz.DTLZ6


class DTLZ7(BaseDTLZ):
    DTLZProblem = optproblems.dtlz.DTLZ7
