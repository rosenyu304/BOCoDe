"""FixedHPO-B candidate-pool base — discrete HPO-B configuration pools.

Each problem is a finite pool of real hyperparameter configurations evaluated in the
HPO-B meta-dataset (Pineda-Arango et al., 2021), exposed as a discrete black-box
optimization benchmark via the FixedHPO-B framing (Gabriel). The search space is
``[0, 1]^d`` (HPO-B's normalized hyperparameters); the objective is validation
accuracy, which is *maximized*. ``evaluate`` returns the accuracy of the nearest
configuration in the pool (a discrete lookup).

All tasks live in a single compact ``hpob_data.npz`` in ``hpo/data/`` (each key
``"<space>_<dataset>"`` maps to a ``(n, d+1)`` float array: the ``d`` features then
the accuracy). A fixed 2000-configuration subsample is used where the source pool is
larger.

Sources:
S. Pineda-Arango, H. S. Jomaa, M. Wistuba, J. Grabocka. HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML. Advances in Neural Information Processing Systems Datasets and Benchmarks, 2021. https://github.com/releaunifreiburg/HPO-B
S. Gabriel. Fixed HPO-B: HPO-B as a large discrete HPO benchmark. https://github.com/SamuelGabriel/FixedHPO-B
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...base import BenchmarkProblem

_DATA_DIR = Path(__file__).resolve().parent / "data"
_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = np.load(_DATA_DIR / "hpob_data.npz")
    return _TABLES


class HPOBProblem(BenchmarkProblem):
    """Discrete pool of HPO-B configurations; evaluate = nearest config's accuracy."""

    num_objectives = 1
    num_constraints = 0
    task_key: str = ""

    def __init__(self) -> None:
        arr = _tables()[self.task_key]
        self._X = torch.tensor(arr[:, :-1], dtype=torch.float64)
        self._y = torch.tensor(arr[:, -1], dtype=torch.float64)
        super().__init__(
            dim=self._X.shape[1],
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * self._X.shape[1],  # HPO-B search space is [0, 1]^d
        )

    @property
    def candidates(self) -> torch.Tensor:
        """The discrete pool of configurations (the search space)."""
        return self._X

    @property
    def values(self) -> torch.Tensor:
        """Validation accuracy of each candidate (maximization)."""
        return self._y

    def _evaluate_implementation(self, X: torch.Tensor) -> tuple:
        Xq = X.to(self._X.dtype)
        idx = torch.argmin(torch.cdist(Xq, self._X), dim=1)
        return None, self._y[idx].reshape(-1, 1).to(torch.float64)
