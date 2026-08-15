"""Weighted-Lasso HPO on the Pima diabetes dataset (8 features).

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import WeightedLassoHPO


class LassoDiabetes(WeightedLassoHPO):
    """Weighted-Lasso HPO on the Pima diabetes dataset (8 features)."""

    available_dimensions = 8
    openml_id = 37
