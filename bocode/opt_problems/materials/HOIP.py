"""HOIP — hybrid organic-inorganic perovskite stability (GP+ dataset, all-categorical).

A materials-design dataset of 480 ABX3 hybrid organic-inorganic perovskite
compositions, each defined by three categorical sites (A: 10 organic cations,
B: 3 metals, X: 16 anion/halide choices) with a DFT-computed inter-molecular
binding energy. The search space is the discrete catalog of compositions; the
objective is to minimize the binding energy (the most stable perovskite). All three
variables are categorical, making this a discrete candidate-pool problem — a good
mixed/categorical materials benchmark.

Sources:
A. Yousefpour, Z. Zanjani Foumani, M. Shishehbor, C. Mora, R. Bostanabad. GP+: a Python library for kernel-based learning via Gaussian processes. Advances in Engineering Software, 2024. https://github.com/Bostanabad-Research-Group/GP-Plus
C. Kim, T. D. Huan, S. Krishnan, R. Ramprasad. A hybrid organic-inorganic perovskite dataset. Scientific Data 4:170057, 2017.
"""

from __future__ import annotations

from ._dataset_problem import MaterialsDatasetProblem


class HOIP(MaterialsDatasetProblem):
    """Minimize HOIP perovskite binding energy (3 categorical composition sites)."""

    available_dimensions = 3
    csv_name = "HOIP_dataset.csv"
    feature_columns = ["site_A", "site_B", "site_X"]
    objective_column = "binding_energy"
    minimize = True

    def __init__(self) -> None:
        super().__init__()
        # three categorical composition sites (A: 10, B: 3, X: 16 levels)
        self.variable_types = [
            list(range(int(self._X[:, i].max()) + 1)) for i in range(self._X.shape[1])
        ]
