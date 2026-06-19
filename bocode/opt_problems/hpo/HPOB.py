"""FixedHPO-B benchmarks — discrete HPO-B configuration pools (maximize accuracy).

Four representative HPO-B search spaces across a range of dimensions, each a discrete
pool of real evaluated hyperparameter configurations (see :mod:`._hpob_base`). The
HPO-B search-space id and OpenML dataset id are given per problem; the model family
follows the standard HPO-B mapping.

Sources:
S. Pineda-Arango, H. S. Jomaa, M. Wistuba, J. Grabocka. HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML. Advances in Neural Information Processing Systems Datasets and Benchmarks, 2021. https://github.com/releaunifreiburg/HPO-B
S. Gabriel. Fixed HPO-B. https://github.com/SamuelGabriel/FixedHPO-B
"""

from __future__ import annotations

from ._hpob_base import HPOBProblem


class HPOBSvm(HPOBProblem):
    """SVM hyperparameter tuning (HPO-B search space 5527, dataset 10101), 8 dims."""

    available_dimensions = 8
    csv_name = "hpob_svm.csv"


class HPOBRpart(HPOBProblem):
    """Decision-tree (rpart) tuning (HPO-B search space 5636, dataset 31), 6 dims."""

    available_dimensions = 6
    csv_name = "hpob_rpart.csv"


class HPOBRanger(HPOBProblem):
    """Random-forest (ranger) tuning (HPO-B search space 5965, dataset 9946), 10 dims."""

    available_dimensions = 10
    csv_name = "hpob_ranger.csv"


class HPOBXgboost(HPOBProblem):
    """XGBoost tuning (HPO-B search space 5971, dataset 6566), 16 dims."""

    available_dimensions = 16
    csv_name = "hpob_xgboost.csv"
