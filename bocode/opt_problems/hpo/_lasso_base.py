"""Shared base for the LassoBench real-dataset hyperparameter problems.

Sources:
Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. Proceedings of the 1st International Conference on Automated Machine Learning (AutoML), 2022.

Each problem tunes one regularization weight per feature of a weighted Lasso on a
real dataset; the decision vector lives in ``[-1, 1]^d`` (log-scaled per-feature
penalties) and the objective is the LassoBench validation loss (returned negated,
since BoCoDe maximizes). Evaluation is delegated to the upstream ``LassoBench``
package, imported lazily so the rest of the library works without it.
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem, DataType


def _import_lassobench():
    try:
        import LassoBench  # noqa: N813
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "LassoBench problems require the optional 'lasso' dependency, which is "
            "installed from git (not on PyPI):\n"
            "    pip install 'bocode[lasso]'"
        ) from exc
    return LassoBench


class LassoBenchRealProblem(BenchmarkProblem):
    """Weighted-Lasso hyperparameter optimization on a real LassoBench dataset.

    Subclasses set ``pick_data`` (the LassoBench dataset key) and
    ``available_dimensions`` (the feature count).
    """

    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    #: LassoBench dataset identifier, e.g. "DNA", "breast_cancer".
    pick_data: str = ""

    def __init__(self) -> None:
        dim = self.available_dimensions
        super().__init__(
            dim=dim,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1, 1)] * dim,
        )
        self._bench = None  # lazily constructed on first evaluation

    def _evaluate_implementation(self, X: torch.Tensor):
        if self._bench is None:
            lassobench = _import_lassobench()
            self._bench = lassobench.RealBenchmark(pick_data=self.pick_data)

        fx = torch.zeros(X.shape[0], 1)
        for i in range(X.shape[0]):
            # LassoBench returns a loss to minimize; negate for BoCoDe's convention.
            fx[i, 0] = -self._bench.evaluate(X[i, :].to(torch.double).numpy())
        return None, fx
