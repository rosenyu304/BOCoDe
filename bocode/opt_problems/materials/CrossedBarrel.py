"""Crossed-barrel mechanical design optimization.

Sources:
A. E. Gongora, B. Xu, W. Perry, et al. A Bayesian experimental autonomous researcher for mechanical design. Science Advances 6(15):eaaz1708, 2020.
Dataset via the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking
"""

from ._dataset_problem import MaterialsDatasetProblem


class CrossedBarrel(MaterialsDatasetProblem):
    """Maximise toughness of a crossed-barrel structure (4 geometric inputs)."""

    available_dimensions = 4
    csv_name = "Crossed barrel_dataset.csv"
    feature_columns = ["n", "theta", "r", "t"]
    objective_column = "toughness"
    minimize = False
