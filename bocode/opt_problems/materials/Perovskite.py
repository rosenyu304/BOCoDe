"""Halide-perovskite compositional-stability optimization.

Sources:
S. Sun, A. Tiihonen, F. Oviedo, et al. A data fusion approach to optimize compositional stability of halide perovskites. Matter 4(4):1305-1322, 2021.
Dataset via the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking
"""

from ._dataset_problem import MaterialsDatasetProblem


class Perovskite(MaterialsDatasetProblem):
    """Minimise the instability index of a halide perovskite (3 composition inputs)."""

    available_dimensions = 3
    csv_name = "Perovskite_dataset.csv"
    feature_columns = ["CsPbI", "FAPbI", "MAPbI"]
    objective_column = "Instability index"
    minimize = True
