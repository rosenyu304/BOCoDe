"""NAS-Bench-201 — tabular neural-architecture search (6 categorical cell edges).

A standard categorical NAS benchmark. The search space is a 4-node cell with 6
directed edges; each edge picks one of 5 operations, giving 5^6 = 15,625
architectures, each with a precomputed final test accuracy on CIFAR-10 /
CIFAR-100 / ImageNet16-120. The objective is to maximize test accuracy — a pure
tabular lookup (no GPU training).

Six categorical decision variables (the edge operations), each in
``{0:none, 1:skip_connect, 2:nor_conv_1x1, 3:nor_conv_3x3, 4:avg_pool_3x3}``.

BoCoDe ships a compact precomputed accuracy table (``nasbench201_accuracy.npz``,
a few MB: the 15,625 architectures' final test accuracies for the three datasets)
fetched on demand, rather than the multi-GB NAS-Bench-201 archive. The table is
built once by ``tools/build_nasbench201_table.py``.

Sources:
X. Dong, Y. Yang. NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search. International Conference on Learning Representations, 2020. https://github.com/D-X-Y/NAS-Bench-201
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..._fetch import fetch_data_file
from ...base import BenchmarkProblem

# Operation set and edge order — must match tools/build_nasbench201_table.py.
_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
_N_EDGES = 6
_DATASET_KEYS = {
    "cifar10": "acc_cifar10",
    "cifar100": "acc_cifar100",
    "imagenet16": "acc_imagenet16",
}
# base-5 place values for encoding the 6 edge ops into a row index 0..15624
_PLACE = np.array([5**i for i in range(_N_EDGES)], dtype=np.int64)


class NASBench201(BenchmarkProblem):
    """Maximize NAS-Bench-201 test accuracy (6 categorical cell edges, 5 ops each)."""

    available_dimensions = _N_EDGES
    num_objectives = 1
    num_constraints = 0

    def __init__(self, dataset: str = "cifar10") -> None:
        if dataset not in _DATASET_KEYS:
            raise ValueError(f"dataset must be one of {list(_DATASET_KEYS)}")
        self._dataset = dataset
        self._table: np.ndarray | None = None
        self.variable_types = [list(range(len(_OPS)))] * _N_EDGES
        super().__init__(
            dim=_N_EDGES,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, len(_OPS) - 1)] * _N_EDGES,
        )

    def _accuracies(self) -> np.ndarray:
        if self._table is None:
            # The table is small (~0.3 MB), so an in-repo copy (if committed) is
            # used directly; otherwise it is fetched from the bocode data host.
            local = (
                Path(__file__).resolve().parent / "data" / "nasbench201_accuracy.npz"
            )
            path = fetch_data_file(
                "nasbench201_accuracy.npz", local_fallback=str(local)
            )
            self._table = np.load(path)[_DATASET_KEYS[self._dataset]]
        return self._table

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = np.rint(X.detach().cpu().numpy()).astype(int).clip(0, len(_OPS) - 1)
        idx = (x * _PLACE).sum(axis=1)
        acc = self._accuracies()[idx]
        fx = torch.tensor(acc, dtype=torch.float64).reshape(-1, 1)  # maximize accuracy
        return None, fx
