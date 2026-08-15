"""Bayesmark HPO base — continuous-relaxation scikit-learn hyperparameter tuning.

Each problem tunes one scikit-learn model on one dataset, maximizing the mean
5-fold cross-validated **accuracy** (Bayesmark's optimizer objective is the
CV *loss* ``-accuracy``; BoCoDe maximizes, so we return ``+accuracy``). This is
the continuous-HPO subset used by PFNs4BO (Müller et al., 2023): the nine
scikit-learn search spaces of Bayesmark, each on the four classification
datasets, optimized for accuracy — 9 models x 4 datasets = 36 tasks.

Continuous relaxation is self-contained: the decision vector lives in the unit
cube ``[0, 1]^dim`` and each coordinate is decoded to its real hyperparameter
*inside* the objective, faithfully to Bayesmark's search-space warping:

* ``linear`` real -> ``lo + u * (hi - lo)``;
* ``log`` real    -> ``lo * (hi / lo) ** u``   (log-uniform);
* ``logit`` real  -> ``expit(logit(lo) + u * (logit(hi) - logit(lo)))``;
* ``int``         -> the decoded real value rounded to the nearest integer.

The search spaces (ranges, log/logit scaling, int/real types) and the fixed
model settings are reproduced verbatim from Bayesmark's
``bayesmark/sklearn_funcs.py`` (commit 8c420e9), with two changes forced by
modern scikit-learn: ``LogisticRegression``'s removed ``multi_class="ovr"`` is
reproduced by wrapping in ``OneVsRestClassifier`` (Bayesmark used the old
liblinear OvR behavior), and the deprecated ``normalize`` regression flag is not
reached because this subset is classification-only. Deterministic given ``seed``
(seeds the 80/20 shuffle-split that defines the objective, and every estimator's
``random_state``).

Sources:
Uber. Bayesmark: benchmark framework for Bayesian optimization. https://github.com/uber/bayesmark (bayesmark/sklearn_funcs.py, commit 8c420e935718f0d6867153b781e58943ecaf2338)
S. Müller, M. Feurer, N. Hollmann, F. Hutter. PFNs4BO: In-Context Learning for Bayesian Optimization. ICML 2023, arXiv:2305.17535. https://github.com/automl/PFNs4BO
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.special import expit, logit
    from sklearn.datasets import (
        load_breast_cancer,
        load_digits,
        load_iris,
        load_wine,
    )
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
except ImportError as _exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The Bayesmark problems require the optional 'hpo' dependency. "
        "Install it with: pip install 'bocode[hpo]'"
    ) from _exc

from ...base import BenchmarkProblem

CV_SPLITS = 5  # Bayesmark uses 5-fold CV (bayesmark/sklearn_funcs.py:CV_SPLITS)

# The four Bayesmark classification datasets (loaded from scikit-learn, exactly
# as Bayesmark's data loader does). Regression datasets are excluded: PFNs4BO
# optimized these tasks for accuracy, a classification metric.
_DATASETS = {
    "breast": load_breast_cancer,
    "digits": load_digits,
    "iris": load_iris,
    "wine": load_wine,
}

# Per-model Bayesmark search spaces, verbatim from bayesmark/sklearn_funcs.py.
# Each entry maps hyperparameter name -> (type, space, lo, hi); dict insertion
# order is the coordinate order of the decision vector.
_CONFIGS: dict[str, dict[str, tuple]] = {
    "kNN": {
        "n_neighbors": ("int", "linear", 1, 25),
        "p": ("int", "linear", 1, 4),
    },
    "SVM": {
        "C": ("real", "log", 1.0, 1e3),
        "gamma": ("real", "log", 1e-4, 1e-3),
        "tol": ("real", "log", 1e-5, 1e-1),
    },
    "DT": {
        "max_depth": ("int", "linear", 1, 15),
        "min_samples_split": ("real", "logit", 0.01, 0.99),
        "min_samples_leaf": ("real", "logit", 0.01, 0.49),
        "min_weight_fraction_leaf": ("real", "logit", 0.01, 0.49),
        "max_features": ("real", "logit", 0.01, 0.99),
        "min_impurity_decrease": ("real", "linear", 0.0, 0.5),
    },
    "RF": {
        "max_depth": ("int", "linear", 1, 15),
        "max_features": ("real", "logit", 0.01, 0.99),
        "min_samples_split": ("real", "logit", 0.01, 0.99),
        "min_samples_leaf": ("real", "logit", 0.01, 0.49),
        "min_weight_fraction_leaf": ("real", "logit", 0.01, 0.49),
        "min_impurity_decrease": ("real", "linear", 0.0, 0.5),
    },
    "MLP-adam": {
        "hidden_layer_sizes": ("int", "linear", 50, 200),
        "alpha": ("real", "log", 1e-5, 1e1),
        "batch_size": ("int", "linear", 10, 250),
        "learning_rate_init": ("real", "log", 1e-5, 1e-1),
        "tol": ("real", "log", 1e-5, 1e-1),
        "validation_fraction": ("real", "logit", 0.1, 0.9),
        "beta_1": ("real", "logit", 0.5, 0.99),
        "beta_2": ("real", "logit", 0.9, 1.0 - 1e-6),
        "epsilon": ("real", "log", 1e-9, 1e-6),
    },
    "MLP-sgd": {
        "hidden_layer_sizes": ("int", "linear", 50, 200),
        "alpha": ("real", "log", 1e-5, 1e1),
        "batch_size": ("int", "linear", 10, 250),
        "learning_rate_init": ("real", "log", 1e-5, 1e-1),
        "power_t": ("real", "logit", 0.1, 0.9),
        "tol": ("real", "log", 1e-5, 1e-1),
        "momentum": ("real", "logit", 0.001, 0.999),
        "validation_fraction": ("real", "logit", 0.1, 0.9),
    },
    "ada": {
        "n_estimators": ("int", "linear", 10, 100),
        "learning_rate": ("real", "log", 1e-4, 1e1),
    },
    "lasso": {
        "C": ("real", "log", 1e-2, 1e2),
        "intercept_scaling": ("real", "log", 1e-2, 1e2),
    },
    "linear": {
        "C": ("real", "log", 1e-2, 1e2),
        "intercept_scaling": ("real", "log", 1e-2, 1e2),
    },
}

# Fixed (non-tuned) model settings, verbatim from Bayesmark's MODELS_CLF.
_FIXED: dict[str, dict] = {
    "kNN": {},
    "SVM": {"kernel": "rbf", "probability": True},
    "DT": {"max_leaf_nodes": None},
    "RF": {"n_estimators": 10, "max_leaf_nodes": None},
    "MLP-adam": {"solver": "adam", "early_stopping": True},
    "MLP-sgd": {
        "solver": "sgd",
        "early_stopping": True,
        "learning_rate": "invscaling",
        "nesterovs_momentum": True,
    },
    "ada": {},
    "lasso": {"penalty": "l1", "fit_intercept": True, "solver": "liblinear"},
    "linear": {"penalty": "l2", "fit_intercept": True, "solver": "liblinear"},
}

_DATA_CACHE: dict[str, tuple] = {}


def _load(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset not in _DATA_CACHE:
        X, y = _DATASETS[dataset](return_X_y=True)
        _DATA_CACHE[dataset] = (np.asarray(X, dtype=float), np.asarray(y))
    return _DATA_CACHE[dataset]


def _decode(u: float, spec: tuple) -> float | int:
    """Decode a unit-cube coordinate to its real hyperparameter (Bayesmark warp)."""
    typ, space, lo, hi = spec
    if space == "log":
        val = lo * (hi / lo) ** u
    elif space == "logit":
        val = float(expit(logit(lo) + u * (logit(hi) - logit(lo))))
    else:  # linear
        val = lo + u * (hi - lo)
    if typ == "int":
        return int(round(val))
    return float(val)


def _build_estimator(model: str, params: dict, seed: int):
    """Build the Bayesmark classifier for ``model`` with decoded ``params``."""
    fixed = dict(_FIXED[model])
    if model == "kNN":
        est = KNeighborsClassifier(**fixed, **params)
    elif model == "SVM":
        est = SVC(**fixed, **params)
    elif model == "DT":
        est = DecisionTreeClassifier(**fixed, **params)
    elif model == "RF":
        est = RandomForestClassifier(**fixed, **params)
    elif model in ("MLP-adam", "MLP-sgd"):
        est = MLPClassifier(**fixed, **params)
    elif model == "ada":
        est = AdaBoostClassifier(**fixed, **params)
    elif model in ("lasso", "linear"):
        # Bayesmark used LogisticRegression(solver="liblinear", multi_class="ovr").
        # multi_class was removed in scikit-learn >=1.7 and liblinear no longer
        # auto-applies OvR to multiclass, so wrap to reproduce the OvR behavior.
        base = LogisticRegression(**fixed, **params)
        est = OneVsRestClassifier(base)
    else:  # pragma: no cover - unreachable given the fixed model set
        raise KeyError(model)

    # Determinism: seed any estimator (or its wrapped estimator) that supports it.
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    if "estimator__random_state" in est.get_params():
        est.set_params(estimator__random_state=seed)
    return est


class BayesmarkProblem(BenchmarkProblem):
    """Continuous-relaxation Bayesmark task: maximize mean 5-fold CV accuracy."""

    num_objectives = 1
    num_constraints = 0
    model_key: str = ""
    dataset_key: str = ""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._config = _CONFIGS[self.model_key]
        self._specs = list(self._config.values())
        dim = len(self._specs)
        X, y = _load(self.dataset_key)
        # 80/20 shuffle-split defines the objective (Bayesmark holds out 20% and
        # runs CV on the 80% training portion); the split must be reproducible.
        self._X_train, _, self._y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True
        )
        super().__init__(
            dim=dim,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * dim,
        )

    def _score_one(self, u: np.ndarray) -> float:
        params = {
            name: _decode(float(u[i]), spec)
            for i, (name, spec) in enumerate(self._config.items())
        }
        est = _build_estimator(self.model_key, params, self.seed)
        try:
            scores = cross_val_score(
                est,
                self._X_train,
                self._y_train,
                scoring="accuracy",
                cv=CV_SPLITS,
            )
            return float(np.mean(scores))
        except Exception:
            # A degenerate hyperparameter combination that scikit-learn refuses to
            # fit is the worst possible point for a maximizer.
            return 0.0

    def _evaluate_implementation(self, X: torch.Tensor) -> tuple:
        X_np = np.clip(X.detach().cpu().numpy(), 0.0, 1.0)
        vals = [self._score_one(row) for row in X_np]
        return None, torch.tensor(vals, dtype=torch.float64).reshape(-1, 1)
