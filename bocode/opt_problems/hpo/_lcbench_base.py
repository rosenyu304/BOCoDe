"""LCBench candidate-pool base — discrete neural-network HPO configuration pools.

Each problem is a pool of 2000 real configurations from LCBench (Zimmer et al.,
2021): a funnel-shaped MLP tuned on an OpenML dataset over seven hyperparameters
(three integer, four continuous). The objective is the final validation accuracy
(%), maximized. ``evaluate`` returns the accuracy of the nearest configuration in
the pool, with distances computed in per-dimension min-max-normalized space so the
very different hyperparameter scales (e.g. ``batch_size`` 16-512 vs ``max_dropout``
0-1) are comparable.

All tasks live in a single compact ``lcbench_data.npz`` in ``hpo/data/`` (each key is
the dataset name; the array is ``(2000, 8)`` = seven hyperparameters then accuracy).

Sources:
L. Zimmer, M. Lindauer, F. Hutter. Auto-PyTorch: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence 43(9):3079-3090, 2021. https://github.com/automl/LCBench
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...base import BenchmarkProblem

_DATA_DIR = Path(__file__).resolve().parent / "data"

# The seven tunable LCBench hyperparameters (column order); three are integers.
_HP = [
    "batch_size",
    "learning_rate",
    "max_dropout",
    "max_units",
    "momentum",
    "num_layers",
    "weight_decay",
]
_INTEGER_IDX = [0, 3, 5]  # batch_size, max_units, num_layers
_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = np.load(_DATA_DIR / "lcbench_data.npz")
    return _TABLES


class LCBenchProblem(BenchmarkProblem):
    """Discrete pool of LCBench MLP configurations; evaluate = nearest config accuracy."""

    available_dimensions = 7
    num_objectives = 1
    num_constraints = 0
    task_key: str = ""

    def __init__(self) -> None:
        arr = _tables()[self.task_key]
        self._X = torch.tensor(arr[:, :-1], dtype=torch.float64)
        self._y = torch.tensor(arr[:, -1], dtype=torch.float64)
        lo = self._X.amin(dim=0)
        hi = self._X.amax(dim=0)
        self._range = (hi - lo).clamp_min(1e-9)
        self.variable_types = [
            "integer" if i in _INTEGER_IDX else "continuous" for i in range(len(_HP))
        ]
        super().__init__(
            dim=len(_HP),
            num_objectives=1,
            num_constraints=0,
            bounds=[(float(lo[i]), float(hi[i])) for i in range(len(_HP))],
        )

    @property
    def candidates(self) -> torch.Tensor:
        """The discrete pool of configurations (the search space)."""
        return self._X

    @property
    def values(self) -> torch.Tensor:
        """Final validation accuracy of each candidate (maximization)."""
        return self._y

    def _evaluate_implementation(self, X: torch.Tensor) -> tuple:
        Xq = X.to(self._X.dtype)
        dists = torch.cdist(Xq / self._range, self._X / self._range)
        idx = torch.argmin(dists, dim=1)
        return None, self._y[idx].reshape(-1, 1).to(torch.float64)
