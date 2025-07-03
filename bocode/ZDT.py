"""
Functions published by the Zitzler, Deb, and Thiele (ZDT)
Zitzler, E., Deb, K., and Thiele, L. (2000). Comparison of Multiobjective Evolutionary Algorithms: Empirical Results. Evolutionary Computation 8(2).
"""

from typing import Tuple

import optproblems.zdt
import torch

from .base import BenchmarkProblem, DataType


class BaseZDT(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    ZDTProblem = None

    def __init__(self, dim: int):
        self.problem = self.__class__.ZDTProblem(dim)

        super().__init__(
            dim=dim,
            num_objectives=2,
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


class ZDT1(BaseZDT):
    ZDTProblem = optproblems.zdt.ZDT1


class ZDT2(BaseZDT):
    ZDTProblem = optproblems.zdt.ZDT2


class ZDT3(BaseZDT):
    ZDTProblem = optproblems.zdt.ZDT3


class ZDT4(BaseZDT):
    ZDTProblem = optproblems.zdt.ZDT4


class ZDT5(BenchmarkProblem):
    available_dimensions = 80
    input_type = DataType.DISCRETE
    num_objectives = 2
    num_constraints = 0
    ZDTProblem = optproblems.zdt.ZDT5

    def __init__(self):
        self.problem = self.__class__.ZDTProblem()

        super().__init__(
            dim=80,
            num_objectives=2,
            num_constraints=0,
            bounds=[(0, 1)] * 80,
        )

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fx = torch.zeros(x.shape[0], self.num_objectives)

        for i, el in enumerate(x):
            # Split el into sublist of 30 bits. Then rest each 5 bits.

            sublists = [el[:30].tolist()] + el[30:].reshape(10, 5).tolist()

            f = self.problem(sublists)

            fx[i, :] = torch.tensor(f)

        return None, fx


class ZDT6(BaseZDT):
    ZDTProblem = optproblems.zdt.ZDT6
