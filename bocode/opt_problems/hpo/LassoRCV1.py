"""Weighted-Lasso hyperparameter optimization on the LassoBench RCV1 dataset.

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import LassoBenchRealProblem


class LassoRCV1(LassoBenchRealProblem):
    """Weighted-Lasso hyperparameter optimization on the LassoBench RCV1 dataset. (47236-dimensional weighted-Lasso tuning)."""

    available_dimensions = 47236
    pick_data = "rcv1"
