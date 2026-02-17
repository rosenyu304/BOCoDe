import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from ..base import BenchmarkProblem, DataType

"""
Source:
Xu, Z., Wang, H., Phillips, J. M., & Zhe, S. (2025). Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization. In The Thirteenth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=kX8h23UG6v
"""


class SVM(BenchmarkProblem):
    available_dimensions = 388
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        self.dims = 388
        self.lb = np.zeros(
            388,
        )
        self.ub = np.ones(
            388,
        )
        self.X, self.y = self._load_data()

        idxs = np.random.choice(
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
        data_path = os.path.join(dir_path, "data", "slice_localization_data.csv")
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
