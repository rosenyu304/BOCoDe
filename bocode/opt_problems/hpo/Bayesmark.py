"""Bayesmark HPO tasks — one continuous-relaxation scikit-learn tuning problem per
(model, dataset) pair. Auto-generated; see :mod:`._bayesmark_base` for the shared
objective (mean 5-fold CV accuracy, maximized) and search-space decoding.

The 36 tasks are the PFNs4BO (Müller et al., 2023) Bayesmark subset: the nine
scikit-learn search spaces of Bayesmark, each on the four classification datasets
(breast, digits, iris, wine), optimized for accuracy.

Sources:
Uber. Bayesmark: benchmark framework for Bayesian optimization. https://github.com/uber/bayesmark
S. Müller, M. Feurer, N. Hollmann, F. Hutter. PFNs4BO: In-Context Learning for Bayesian Optimization. ICML 2023, arXiv:2305.17535. https://github.com/automl/PFNs4BO
"""

from __future__ import annotations

from ._bayesmark_base import BayesmarkProblem


class Bayesmark_kNN_breast(BayesmarkProblem):
    """Bayesmark kNN tuning on the breast dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "kNN"
    dataset_key = "breast"


class Bayesmark_kNN_digits(BayesmarkProblem):
    """Bayesmark kNN tuning on the digits dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "kNN"
    dataset_key = "digits"


class Bayesmark_kNN_iris(BayesmarkProblem):
    """Bayesmark kNN tuning on the iris dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "kNN"
    dataset_key = "iris"


class Bayesmark_kNN_wine(BayesmarkProblem):
    """Bayesmark kNN tuning on the wine dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "kNN"
    dataset_key = "wine"


class Bayesmark_SVM_breast(BayesmarkProblem):
    """Bayesmark SVM tuning on the breast dataset (3 hyperparameters), CV accuracy."""

    available_dimensions = 3
    model_key = "SVM"
    dataset_key = "breast"


class Bayesmark_SVM_digits(BayesmarkProblem):
    """Bayesmark SVM tuning on the digits dataset (3 hyperparameters), CV accuracy."""

    available_dimensions = 3
    model_key = "SVM"
    dataset_key = "digits"


class Bayesmark_SVM_iris(BayesmarkProblem):
    """Bayesmark SVM tuning on the iris dataset (3 hyperparameters), CV accuracy."""

    available_dimensions = 3
    model_key = "SVM"
    dataset_key = "iris"


class Bayesmark_SVM_wine(BayesmarkProblem):
    """Bayesmark SVM tuning on the wine dataset (3 hyperparameters), CV accuracy."""

    available_dimensions = 3
    model_key = "SVM"
    dataset_key = "wine"


class Bayesmark_DT_breast(BayesmarkProblem):
    """Bayesmark DT tuning on the breast dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "DT"
    dataset_key = "breast"


class Bayesmark_DT_digits(BayesmarkProblem):
    """Bayesmark DT tuning on the digits dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "DT"
    dataset_key = "digits"


class Bayesmark_DT_iris(BayesmarkProblem):
    """Bayesmark DT tuning on the iris dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "DT"
    dataset_key = "iris"


class Bayesmark_DT_wine(BayesmarkProblem):
    """Bayesmark DT tuning on the wine dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "DT"
    dataset_key = "wine"


class Bayesmark_RF_breast(BayesmarkProblem):
    """Bayesmark RF tuning on the breast dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "RF"
    dataset_key = "breast"


class Bayesmark_RF_digits(BayesmarkProblem):
    """Bayesmark RF tuning on the digits dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "RF"
    dataset_key = "digits"


class Bayesmark_RF_iris(BayesmarkProblem):
    """Bayesmark RF tuning on the iris dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "RF"
    dataset_key = "iris"


class Bayesmark_RF_wine(BayesmarkProblem):
    """Bayesmark RF tuning on the wine dataset (6 hyperparameters), CV accuracy."""

    available_dimensions = 6
    model_key = "RF"
    dataset_key = "wine"


class Bayesmark_MLPadam_breast(BayesmarkProblem):
    """Bayesmark MLP-adam tuning on the breast dataset (9 hyperparameters), CV accuracy."""

    available_dimensions = 9
    model_key = "MLP-adam"
    dataset_key = "breast"


class Bayesmark_MLPadam_digits(BayesmarkProblem):
    """Bayesmark MLP-adam tuning on the digits dataset (9 hyperparameters), CV accuracy."""

    available_dimensions = 9
    model_key = "MLP-adam"
    dataset_key = "digits"


class Bayesmark_MLPadam_iris(BayesmarkProblem):
    """Bayesmark MLP-adam tuning on the iris dataset (9 hyperparameters), CV accuracy."""

    available_dimensions = 9
    model_key = "MLP-adam"
    dataset_key = "iris"


class Bayesmark_MLPadam_wine(BayesmarkProblem):
    """Bayesmark MLP-adam tuning on the wine dataset (9 hyperparameters), CV accuracy."""

    available_dimensions = 9
    model_key = "MLP-adam"
    dataset_key = "wine"


class Bayesmark_MLPsgd_breast(BayesmarkProblem):
    """Bayesmark MLP-sgd tuning on the breast dataset (8 hyperparameters), CV accuracy."""

    available_dimensions = 8
    model_key = "MLP-sgd"
    dataset_key = "breast"


class Bayesmark_MLPsgd_digits(BayesmarkProblem):
    """Bayesmark MLP-sgd tuning on the digits dataset (8 hyperparameters), CV accuracy."""

    available_dimensions = 8
    model_key = "MLP-sgd"
    dataset_key = "digits"


class Bayesmark_MLPsgd_iris(BayesmarkProblem):
    """Bayesmark MLP-sgd tuning on the iris dataset (8 hyperparameters), CV accuracy."""

    available_dimensions = 8
    model_key = "MLP-sgd"
    dataset_key = "iris"


class Bayesmark_MLPsgd_wine(BayesmarkProblem):
    """Bayesmark MLP-sgd tuning on the wine dataset (8 hyperparameters), CV accuracy."""

    available_dimensions = 8
    model_key = "MLP-sgd"
    dataset_key = "wine"


class Bayesmark_ada_breast(BayesmarkProblem):
    """Bayesmark ada tuning on the breast dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "ada"
    dataset_key = "breast"


class Bayesmark_ada_digits(BayesmarkProblem):
    """Bayesmark ada tuning on the digits dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "ada"
    dataset_key = "digits"


class Bayesmark_ada_iris(BayesmarkProblem):
    """Bayesmark ada tuning on the iris dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "ada"
    dataset_key = "iris"


class Bayesmark_ada_wine(BayesmarkProblem):
    """Bayesmark ada tuning on the wine dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "ada"
    dataset_key = "wine"


class Bayesmark_lasso_breast(BayesmarkProblem):
    """Bayesmark lasso tuning on the breast dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "lasso"
    dataset_key = "breast"


class Bayesmark_lasso_digits(BayesmarkProblem):
    """Bayesmark lasso tuning on the digits dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "lasso"
    dataset_key = "digits"


class Bayesmark_lasso_iris(BayesmarkProblem):
    """Bayesmark lasso tuning on the iris dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "lasso"
    dataset_key = "iris"


class Bayesmark_lasso_wine(BayesmarkProblem):
    """Bayesmark lasso tuning on the wine dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "lasso"
    dataset_key = "wine"


class Bayesmark_linear_breast(BayesmarkProblem):
    """Bayesmark linear tuning on the breast dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "linear"
    dataset_key = "breast"


class Bayesmark_linear_digits(BayesmarkProblem):
    """Bayesmark linear tuning on the digits dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "linear"
    dataset_key = "digits"


class Bayesmark_linear_iris(BayesmarkProblem):
    """Bayesmark linear tuning on the iris dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "linear"
    dataset_key = "iris"


class Bayesmark_linear_wine(BayesmarkProblem):
    """Bayesmark linear tuning on the wine dataset (2 hyperparameters), CV accuracy."""

    available_dimensions = 2
    model_key = "linear"
    dataset_key = "wine"
