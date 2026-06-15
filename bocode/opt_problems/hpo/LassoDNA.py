"""Weighted-Lasso HPO on the DNA splice-junction dataset (180 features).

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import WeightedLassoHPO


class LassoDNA(WeightedLassoHPO):
    """Weighted-Lasso HPO on the DNA splice-junction dataset (180 features)."""

    available_dimensions = 180
    openml_id = 40670
