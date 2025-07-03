"""
Sources:
(1) Šehić Kenan, Gramfort Alexandre, Salmon Joseph and Nardi Luigi, "LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso", Proceedings of the 1st International Conference on Automated Machine Learning, 2022.
"""

import torch

from ..base import BenchmarkProblem, DataType


class LassoSyntMedium(BenchmarkProblem):
    """
    ...
    """

    available_dimensions = 100
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "LassoSyntMedium",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 100-Dim",
            "IMPORTS: LassoBench",
        ]

        super().__init__(
            dim=100,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1, 1)] * 100,
            tags=tags,
        )

    def _evaluate_implementation(self, X):
        import LassoBench

        fx = torch.zeros(X.shape[0], 1)
        synt_bench = LassoBench.SyntheticBenchmark(pick_bench="synt_medium")
        for i in range(X.shape[0]):
            fx[i, 0] = -synt_bench.evaluate(X[i, :].to(torch.double).numpy())

        return None, fx
