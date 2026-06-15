"""Weighted-Lasso HPO on the RCV1 text dataset (47236 sparse features).

This is the highest-dimensional Lasso problem. RCV1 is fetched via
``sklearn.datasets.fetch_rcv1`` (large download on first use) and the training
rows are subsampled to keep the repeated sparse-Lasso fits tractable.

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from ._lasso_base import WeightedLassoHPO, _load_rcv1


class LassoRCV1(WeightedLassoHPO):
    """Weighted-Lasso HPO on RCV1 (47236 features)."""

    available_dimensions = 47236
    openml_id = -1  # not used; see load_data

    def load_data(self):
        return _load_rcv1()
