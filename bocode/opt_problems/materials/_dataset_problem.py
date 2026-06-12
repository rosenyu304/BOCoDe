"""Discrete lookup-table optimization over experimental materials-science datasets.

Sources:
Q. Liang, A. E. Gongora, Z. Ren, et al. Benchmarking the performance of Bayesian optimization across multiple experimental materials science domains. npj Computational Materials 7:188, 2021. https://doi.org/10.1038/s41524-021-00656-9
Datasets and framing from the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking

Each problem wraps a real experimental dataset as a finite candidate pool: the
search space is the set of measured input rows, and ``evaluate`` returns the
measured objective for the nearest candidate. This matches the PV-Lab framing
where Bayesian optimization selects among already-characterized experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ...base import BenchmarkProblem, DataType

_DATA_DIR = Path(__file__).resolve().parent / "data"


class MaterialsDatasetProblem(BenchmarkProblem):
    """Base class for discrete dataset-driven materials problems.

    Subclasses set ``csv_name``, ``feature_columns``, ``objective_column``, and
    ``minimize``. The objective is always exposed for minimization (a maximized
    quantity is negated internally), consistent with BoCoDe's convention.
    """

    input_type = DataType.DISCRETE
    num_objectives = 1
    num_constraints = 0

    csv_name: str = ""
    feature_columns: Optional[list] = None
    objective_column: str = ""
    minimize: bool = True

    def __init__(self) -> None:
        df = pd.read_csv(_DATA_DIR / self.csv_name, encoding="utf-8-sig")
        features = self.feature_columns or [
            c for c in df.columns if c != self.objective_column
        ]
        self._X = torch.tensor(df[features].to_numpy(), dtype=torch.float64)
        y = df[self.objective_column].to_numpy(dtype=float)
        self._y = torch.tensor(y if self.minimize else -y, dtype=torch.float64)
        self._feature_names = features

        bounds = [
            (float(self._X[:, i].min()), float(self._X[:, i].max()))
            for i in range(self._X.shape[1])
        ]
        super().__init__(
            dim=len(features),
            num_objectives=1,
            num_constraints=0,
            bounds=bounds,
        )

    @property
    def candidates(self) -> torch.Tensor:
        """The full set of measured input rows (the discrete search space)."""
        return self._X

    @property
    def values(self) -> torch.Tensor:
        """Objective values aligned with :attr:`candidates` (minimization)."""
        return self._y

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Return the measured objective of the nearest candidate row for each X."""
        Xq = X.to(self._X.dtype)
        # squared Euclidean distance to every candidate, then take the closest
        dists = torch.cdist(Xq, self._X)
        idx = torch.argmin(dists, dim=1)
        fx = self._y[idx].reshape(-1, 1).to(torch.float64)
        return None, fx
