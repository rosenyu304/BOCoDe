"""LCBench benchmarks — discrete neural-network HPO pools on four OpenML datasets.

Four representative LCBench tasks (see :mod:`._lcbench_base`): tuning a funnel MLP's
seven hyperparameters on the named OpenML dataset, maximizing final validation
accuracy. Mixed-variable (batch_size, max_units, num_layers are integers).

Sources:
L. Zimmer, M. Lindauer, F. Hutter. Auto-PyTorch: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence 43(9):3079-3090, 2021. https://github.com/automl/LCBench
"""

from __future__ import annotations

from ._lcbench_base import LCBenchProblem


class LCBenchCreditG(LCBenchProblem):
    """LCBench MLP tuning on the credit-g dataset (7 hyperparameters)."""

    csv_name = "lcbench_credit_g.csv"


class LCBenchHiggs(LCBenchProblem):
    """LCBench MLP tuning on the higgs dataset (7 hyperparameters)."""

    csv_name = "lcbench_higgs.csv"


class LCBenchFashionMNIST(LCBenchProblem):
    """LCBench MLP tuning on the Fashion-MNIST dataset (7 hyperparameters)."""

    csv_name = "lcbench_Fashion_MNIST.csv"


class LCBenchAlbert(LCBenchProblem):
    """LCBench MLP tuning on the albert dataset (7 hyperparameters)."""

    csv_name = "lcbench_albert.csv"
