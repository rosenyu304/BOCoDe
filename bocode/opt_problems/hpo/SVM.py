import os

import numpy as np
import pandas as pd
import torch

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVR
except ImportError as _exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The SVM problem requires the optional 'hpo' dependency. "
        "Install it with: pip install 'bocode[hpo]'"
    ) from _exc

from ..._fetch import fetch_data_file
from ...base import BenchmarkProblem

"""
Source:
Xu, Z., Wang, H., Phillips, J. M., & Zhe, S. (2025). Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization. In The Thirteenth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=kX8h23UG6v
"""


class SVM(BenchmarkProblem):
    available_dimensions = 388
    num_objectives = 1
    num_constraints = 0

    def __init__(self, seed: int = 0):
        """Weighted-SVR hyperparameter tuning on a 10,000-row slice of the data.

        Args:
            seed: seeds the 10,000-row train/test split. The split DEFINES the
                objective (a different split is a different function), so it must be
                reproducible: an unseeded split made every instantiation optimize a
                different objective and made runs incomparable across seeds. The
                experiment harness (``algorithms._bo_utils.make_problem``) passes the
                RUN's seed here, so a run is reproducible end-to-end; constructing
                ``SVM(seed=k)`` twice always yields the identical split.
        """
        self.dims = 388
        self.seed = seed
        self.lb = np.zeros(
            388,
        )
        self.ub = np.ones(
            388,
        )
        self.X, self.y = self._load_data()

        # Seeded, instance-local RNG: does not touch (or depend on) global numpy state.
        rng = np.random.default_rng(seed)
        idxs = rng.choice(
            np.arange(len(self.X)), min(10000, len(self.X)), replace=False
        )
        half = len(idxs) // 2
        self._X_train = self.X[idxs[:half]]
        self._X_test = self.X[idxs[half:]]
        self._y_train = self.y[idxs[:half]]
        self._y_test = self.y[idxs[half:]]
        super().__init__(
            dim=self.dims,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 1)] * self.dims,
        )

    def _load_data(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        local_path = os.path.join(dir_path, "data", "slice_localization_data.csv")
        data_path = fetch_data_file(
            "slice_localization_data.csv", local_fallback=local_path
        )
        data = pd.read_csv(data_path).to_numpy()
        X = data[:, :385]
        y = data[:, -1]
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        return X, y

    def __call__(self, x: np.array, verbose):
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1
        x = x**2

        C = 0.01 * (500 ** x[387])
        gamma = 0.1 * (30 ** x[386])
        epsilon = 0.01 * (100 ** x[385])
        length_scales = np.exp(4 * x[:385] - 2)

        svr = SVR(
            gamma=gamma,
            epsilon=epsilon,
            C=C,
            cache_size=1500,
            tol=0.001,
            verbose=verbose,
        )
        svr.fit(self._X_train / length_scales, self._y_train)
        pred = svr.predict(self._X_test / length_scales)
        error = np.sqrt(np.mean(np.square(pred - self._y_test)))

        return -error

    def _evaluate_implementation(
        self, x: torch.Tensor, scaling=False, verbose=False
    ) -> torch.Tensor:
        # return None, self.__call__(x.numpy())
        x_np = x.detach().cpu().numpy()
        results = []
        for xi in x_np:
            result = self.__call__(xi, verbose)
            results.append(result)

        return None, torch.tensor(results, dtype=torch.float32).unsqueeze(-1)
