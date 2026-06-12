"""Weighted-Lasso hyperparameter optimization on the LassoBench diabetes dataset.

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import LassoBenchRealProblem


class LassoDiabetes(LassoBenchRealProblem):
    """Weighted-Lasso hyperparameter optimization on the LassoBench diabetes dataset. (8-dimensional weighted-Lasso tuning)."""

    available_dimensions = 8
    pick_data = "diabetes"
