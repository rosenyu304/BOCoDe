"""Weighted-Lasso HPO on the leukemia gene-expression dataset (7129 features).

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import WeightedLassoHPO


class LassoLeukemia(WeightedLassoHPO):
    """Weighted-Lasso HPO on the leukemia gene-expression dataset (7129 features)."""

    available_dimensions = 7129
    openml_id = 1104
