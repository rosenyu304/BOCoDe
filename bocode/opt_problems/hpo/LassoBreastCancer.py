"""Weighted-Lasso HPO on the Wisconsin breast-cancer dataset (9 features; the original LassoBench libsvm version uses 10).

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import WeightedLassoHPO


class LassoBreastCancer(WeightedLassoHPO):
    """Weighted-Lasso HPO on the Wisconsin breast-cancer dataset (9 features; the original LassoBench libsvm version uses 10)."""

    available_dimensions = 9
    openml_id = 15
