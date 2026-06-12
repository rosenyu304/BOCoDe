"""Weighted-Lasso hyperparameter optimization on the LassoBench DNA dataset.

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import LassoBenchRealProblem


class LassoDNA(LassoBenchRealProblem):
    """Weighted-Lasso hyperparameter optimization on the LassoBench DNA dataset. (180-dimensional weighted-Lasso tuning)."""

    available_dimensions = 180
    pick_data = "DNA"
