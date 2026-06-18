"""Weighted-Lasso hyperparameter-optimization problems (clean-room reimplementation).

A self-contained reimplementation of the LassoBench real-dataset problems that runs
in the core BoCoDe environment using only scikit-learn — no external ``LassoBench``
package and no ``celer``/``sparse-ho``. Each problem tunes one L1 penalty weight per
feature of a weighted Lasso fit on a real dataset; the decision vector lives in
``[-1, 1]^d`` (log-scaled per-feature penalties) and the objective is the validation
mean-squared error (returned negated, since BoCoDe maximizes).

Implementation notes / differences from upstream LassoBench:
  * Weighted Lasso is realized with scikit-learn's ``Lasso`` by column-rescaling the
    design matrix (penalty ``w_i`` on feature ``i`` == standard Lasso on ``X_i / w_i``).
  * The datasets are the LIBSVM versions (fetched via OpenML), whose feature counts
    match the original LassoBench dimensions (diabetes 8, breast-cancer 10, DNA 180,
    leukemia 7129, rcv1 47236). Data is cached by scikit-learn under
    ``~/scikit_learn_data`` (network on first use).
  * Because the solver, train/validation split, and penalty parametrization differ
    from the original, the *values* are not numerically identical to upstream
    LassoBench; the problem structure (real weighted-Lasso HPO over the same data
    and dimension) is the same. This is BoCoDe's reimplementation.

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

# Per-feature penalty range: input -1 -> 10**-PENALTY_DECADES, +1 -> 10**+PENALTY_DECADES.
PENALTY_DECADES = 3.0


def _load_openml(data_id: int):
    """Fetch a (standardized X, centered y) regression dataset from OpenML, cached."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    ds = fetch_openml(data_id=data_id, as_frame=False)
    X = np.asarray(ds.data, dtype=float)
    y = np.asarray(ds.target)
    # Encode non-numeric targets (classification datasets) as 0..K-1 regression targets.
    if y.dtype.kind in "US" or y.dtype == object:
        _, y = np.unique(y, return_inverse=True)
    y = y.astype(float)
    # Mean-impute any missing feature values, then standardize.
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    X = StandardScaler().fit_transform(X)
    y = y - y.mean()
    return X, y


def _load_rcv1(max_samples: int = 4000):
    """Fetch RCV1 (47236 sparse features); subsample rows to keep Lasso tractable."""
    from sklearn.datasets import fetch_rcv1

    bunch = fetch_rcv1(subset="train")
    X = bunch.data  # sparse, (n, 47236)
    # binary target: first label column, as a 0/1 regression target
    y = np.asarray(bunch.target[:, 0].todense()).ravel().astype(float)
    n = min(max_samples, X.shape[0])
    rng = np.random.RandomState(0)
    idx = rng.choice(X.shape[0], n, replace=False)
    X = X[idx]
    y = y[idx]
    y = y - y.mean()
    return X, y


class WeightedLassoHPO(BenchmarkProblem):
    """Tune one L1 penalty weight per feature of a weighted Lasso (validation MSE).

    Subclasses set ``openml_id`` (the dataset) and ``available_dimensions`` (its
    feature count). The dataset is loaded lazily on first evaluation.
    """

    num_objectives = 1
    num_constraints = 0

    #: OpenML dataset id whose feature count equals ``available_dimensions``.
    openml_id: int = 0

    def __init__(self) -> None:
        dim = self.available_dimensions
        super().__init__(
            dim=dim,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1, 1)] * dim,
        )
        self._data = None  # (X_train, X_val, y_train, y_val), built lazily

    def load_data(self):
        """Return ``(X, y)`` for the problem; override for non-OpenML datasets."""
        return _load_openml(self.openml_id)

    def _ensure_data(self):
        if self._data is None:
            from sklearn.model_selection import train_test_split

            X, y = self.load_data()
            if X.shape[1] != self.dim:
                raise ValueError(
                    f"{type(self).__name__}: dataset has {X.shape[1]} features but "
                    f"available_dimensions={self.dim}."
                )
            self._data = train_test_split(X, y, test_size=0.25, random_state=0)
        return self._data

    def _rescale(self, X, weights):
        """Divide each feature column by its weight (dense or sparse-safe)."""
        import scipy.sparse as sp

        if sp.issparse(X):
            return X.multiply(1.0 / weights).tocsr()
        return X / weights

    def _val_mse(self, alpha_vec: np.ndarray) -> float:
        from sklearn.linear_model import Lasso

        X_tr, X_val, y_tr, y_val = self._ensure_data()
        weights = 10.0 ** (np.clip(alpha_vec, -1.0, 1.0) * PENALTY_DECADES)
        model = Lasso(alpha=1.0, max_iter=5000).fit(self._rescale(X_tr, weights), y_tr)
        pred = self._rescale(X_val, weights) @ model.coef_ + model.intercept_
        pred = np.asarray(pred).ravel()
        return float(np.mean((y_val - pred) ** 2))

    def _evaluate_implementation(self, X: torch.Tensor):
        fx = torch.zeros(X.shape[0], 1, dtype=torch.float64)
        for i in range(X.shape[0]):
            # negate so the validation MSE is maximized (BoCoDe convention)
            fx[i, 0] = -self._val_mse(X[i, :].to(torch.double).numpy())
        return None, fx
