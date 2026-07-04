"""FixedHPO-B candidate-pool base — discrete mixed-variable HPO-B configuration pools.

Each problem is a finite pool of real hyperparameter configurations evaluated in the
HPO-B meta-dataset (Pineda-Arango et al., 2021), exposed as a discrete black-box
optimization benchmark via the FixedHPO-B framing (Gabriel). The search space is
``[0, 1]^d`` (HPO-B's normalized hyperparameters); the objective is validation
accuracy, which is *maximized*. ``evaluate`` returns the accuracy of the nearest
configuration in the pool (a discrete lookup).

Mixed variables: HPO-B one-hot-encodes categorical hyperparameters (``*.ohe.*``
columns) and adds a binary ``*.na`` missing-value indicator per optional
hyperparameter, while integer hyperparameters (tree depth, polynomial degree,
``mtry``, ...) land on a fixed grid of normalized values. ``variable_types`` is
derived per task from its pool: a column with at most ``_MAX_DISCRETE_CARD``
distinct values is declared discrete with exactly that grid as its allowed values
(one-hot/indicator columns become ``[0.0, 1.0]``); all other columns are
continuous. Column names per search space are in ``SPACE_VARIABLES`` (from HPO-B's
``meta-dataset-descriptors.json``). High-cardinality integer hyperparameters
(``num.trees``, ``nrounds``, ``min.node.size``) intentionally stay "continuous":
their normalized grid is finer than any practical rounding.

Log fix: following FixedHPO-B, the log-scale dimensions that HPO-B forgot to
transform (``LOG_FIXED_DIMS``) are log-transformed and re-min-max-normalized at
table-build time (see ``tools/build_hpob_tables.py``), so the pools stored here
already include the fix.

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

# A pool column with at most this many distinct values is treated as a discrete
# variable (its allowed values = the observed grid). Pools have >= 100 configs, so
# genuinely continuous columns carry ~pool-size distinct values and never trip this.
_MAX_DISCRETE_CARD = 64

# Column names per HPO-B-v3 search space, in on-disk order, from the official
# hpob-data/meta-dataset-descriptors.json (``variables_order``). ``*.na`` columns
# are binary missing-value indicators; ``*.ohe.*`` columns are one-hot categories.
SPACE_VARIABLES = {
    "4796": ["minsplit", "minbucket", "cp"],
    "5527": [
        "cost",
        "gamma",
        "gamma.na",
        "degree",
        "degree.na",
        "kernel.ohe._INVALID",
        "kernel.ohe._linear",
        "kernel.ohe._polynomial",
    ],
    "5636": ["minsplit", "minsplit.na", "minbucket", "cp", "maxdepth", "maxdepth.na"],
    "5859": ["minsplit", "minsplit.na", "minbucket", "cp", "maxdepth", "maxdepth.na"],
    "5860": ["alpha", "lambda"],
    "5889": [
        "num.trees",
        "num.trees.na",
        "mtry",
        "sample.fraction",
        "replace.ohe._FALSE",
        "replace.ohe._INVALID",
    ],
    "5891": [
        "cost",
        "gamma",
        "gamma.na",
        "degree",
        "degree.na",
        "kernel.ohe._INVALID",
        "kernel.ohe._linear",
        "kernel.ohe._polynomial",
    ],
    "5906": [
        "eta",
        "max_depth",
        "max_depth.na",
        "min_child_weight",
        "min_child_weight.na",
        "subsample",
        "colsample_bytree",
        "colsample_bytree.na",
        "colsample_bylevel",
        "colsample_bylevel.na",
        "lambda",
        "alpha",
        "nrounds",
        "nrounds.na",
        "booster.ohe._INVALID",
        "booster.ohe._gblinear",
    ],
    "5965": [
        "num.trees",
        "num.trees.na",
        "mtry",
        "sample.fraction",
        "min.node.size",
        "min.node.size.na",
        "replace.ohe._FALSE",
        "replace.ohe._INVALID",
        "respect.unordered.factors.ohe._INVALID",
        "respect.unordered.factors.ohe._TRUE",
    ],
    "5970": ["alpha", "lambda"],
    "5971": [
        "eta",
        "max_depth",
        "max_depth.na",
        "min_child_weight",
        "min_child_weight.na",
        "subsample",
        "colsample_bytree",
        "colsample_bytree.na",
        "colsample_bylevel",
        "colsample_bylevel.na",
        "lambda",
        "alpha",
        "nrounds",
        "nrounds.na",
        "booster.ohe._INVALID",
        "booster.ohe._gblinear",
    ],
    "6766": ["alpha", "lambda"],
    "6767": [
        "eta",
        "subsample",
        "lambda",
        "alpha",
        "nthread",
        "nthread.na",
        "nrounds",
        "nrounds.na",
        "max_depth",
        "max_depth.na",
        "min_child_weight",
        "min_child_weight.na",
        "colsample_bytree",
        "colsample_bytree.na",
        "colsample_bylevel",
        "colsample_bylevel.na",
        "booster.ohe._INVALID",
        "booster.ohe._gblinear",
    ],
    "6794": [
        "num.trees",
        "num.trees.na",
        "mtry",
        "sample.fraction",
        "min.node.size",
        "min.node.size.na",
        "replace.ohe._FALSE",
        "replace.ohe._INVALID",
        "respect.unordered.factors.ohe._INVALID",
        "respect.unordered.factors.ohe._TRUE",
    ],
    "7607": [
        "num.trees",
        "num.trees.na",
        "mtry",
        "min.node.size",
        "sample.fraction",
        "respect.unordered.factors.ohe._INVALID",
        "respect.unordered.factors.ohe._TRUE",
        "replace.ohe._FALSE",
        "replace.ohe._INVALID",
    ],
    "7609": [
        "num.trees",
        "num.trees.na",
        "mtry",
        "min.node.size",
        "sample.fraction",
        "respect.unordered.factors.ohe._INVALID",
        "respect.unordered.factors.ohe._TRUE",
        "replace.ohe._FALSE",
        "replace.ohe._INVALID",
    ],
}

# FixedHPO-B's log fix (``fixedhpob.hpob.log_indices_per_method``): dimensions that
# HPO-B left un-logged despite log-distributed values. Applied at build time.
LOG_FIXED_DIMS = {
    "5891": [0, 1],  # cost, gamma
    "5970": [1],  # lambda
    "6766": [1],  # lambda
    "6767": [0, 2, 3],  # eta, lambda, alpha
    "6794": [2],  # mtry
    "7607": [2],  # mtry
    "7609": [2],  # mtry
}


def _tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = np.load(_DATA_DIR / "hpob_data.npz")
    return _TABLES


def _derive_variable_types(X: np.ndarray) -> list:
    """Per-column variable types from the pool: small grids are discrete."""
    types = []
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if len(u) <= _MAX_DISCRETE_CARD:
            types.append([float(v) for v in u])
        else:
            types.append("continuous")
    return types


class HPOBProblem(BenchmarkProblem):
    """Discrete pool of HPO-B configurations; evaluate = nearest config's accuracy."""

    num_objectives = 1
    num_constraints = 0
    task_key: str = ""

    def __init__(self) -> None:
        arr = _tables()[self.task_key]
        self._X = torch.tensor(arr[:, :-1], dtype=torch.float64)
        self._y = torch.tensor(arr[:, -1], dtype=torch.float64)
        self.variable_types = _derive_variable_types(arr[:, :-1])
        super().__init__(
            dim=self._X.shape[1],
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * self._X.shape[1],  # HPO-B search space is [0, 1]^d
        )

    @property
    def space_id(self) -> str:
        """The HPO-B search-space id (model family shared across datasets)."""
        return self.task_key.split("_")[0]

    @property
    def variable_names(self) -> list:
        """Column names for this search space (HPO-B descriptor order)."""
        return list(SPACE_VARIABLES[self.space_id])

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
