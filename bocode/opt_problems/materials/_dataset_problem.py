"""Discrete lookup-table optimization over experimental materials-science datasets.

Sources:
Q. Liang, A. E. Gongora, Z. Ren, et al. Benchmarking the performance of Bayesian optimization across multiple experimental materials science domains. npj Computational Materials 7:188, 2021. https://doi.org/10.1038/s41524-021-00656-9
Datasets and framing from the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking

Each problem wraps a real experimental dataset as a finite candidate pool: the
search space is the set of measured input rows, and ``evaluate`` returns the
measured objective for the nearest candidate. This matches the PV-Lab framing
where Bayesian optimization selects among already-characterized experiments.

Two conventions apply to every dataset here:

* **Replicates are averaged.** Where the same input row was measured more than
  once, the candidate pool holds one entry per unique input vector carrying the
  *mean* of its replicate measurements.
* **Nearest-neighbour lookup is scale-aware.** The nearest candidate is found in
  min-max normalized feature space, so the lookup does not depend on the physical
  units the features happen to be recorded in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ...base import BenchmarkProblem

_DATA_DIR = Path(__file__).resolve().parent / "data"


class MaterialsDatasetProblem(BenchmarkProblem):
    """Base class for discrete dataset-driven materials problems.

    Subclasses set ``csv_name``, ``feature_columns``, ``objective_column``, and
    ``minimize``. The returned objective is always exposed for *maximization*
    (BoCoDe's convention): a quantity that should be minimized (e.g. a loss) is
    negated internally, a quantity to be maximized is returned as-is.
    """

    num_objectives = 1
    num_constraints = 0

    csv_name: str = ""
    feature_columns: list | None = None
    objective_column: str = ""
    minimize: bool = True

    def __init__(self) -> None:
        df = pd.read_csv(_DATA_DIR / self.csv_name, encoding="utf-8-sig")
        features = self.feature_columns or [
            c for c in df.columns if c != self.objective_column
        ]
        # Average replicate measurements. Several PV-Lab datasets repeat the same
        # input row with different measured outcomes (AgNP: 3295 rows -> 164 unique
        # inputs; CrossedBarrel: 1800 -> 600, each measured 3x). Grouping by the
        # feature vector and taking the mean makes the objective a well-defined
        # function of the inputs; without it the value returned for a candidate
        # depends on which duplicate row happens to be found first, which is an
        # arbitrary (and noisy) artefact of the CSV row order.
        df = df.groupby(features, as_index=False, sort=False)[
            self.objective_column
        ].mean()

        self._X = torch.tensor(df[features].to_numpy(), dtype=torch.float64)
        y = df[self.objective_column].to_numpy(dtype=float)
        # BoCoDe maximizes the returned objective, so a quantity that should be
        # minimized (e.g. a loss) is negated and a quantity to be maximized is
        # returned as-is.
        self._y = torch.tensor(-y if self.minimize else y, dtype=torch.float64)
        self._feature_names = features

        # Per-feature min/max used to normalize the nearest-neighbour metric (see
        # ``_evaluate_implementation``); a constant feature gets a clamped divisor.
        self._lo = self._X.min(dim=0).values
        self._span = (self._X.max(dim=0).values - self._lo).clamp(min=1e-12)
        self._Xn = (self._X - self._lo) / self._span

        bounds = [
            (float(self._X[:, i].min()), float(self._X[:, i].max()))
            for i in range(self._X.shape[1])
        ]
        super().__init__(
            dim=len(features),
            num_objectives=1,
            num_constraints=0,
            bounds=bounds,
        )

    @property
    def candidates(self) -> torch.Tensor:
        """The full set of measured input rows (the discrete search space)."""
        return self._X

    @property
    def values(self) -> torch.Tensor:
        """Objective values aligned with :attr:`candidates` (maximization)."""
        return self._y

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return the measured objective of the nearest candidate row for each X.

        Distances are taken in *min-max normalized* feature space, not in raw
        physical units. The features have wildly different units and ranges (AgNP
        spans 38.3, 30.0, 30.0, 19.0 and 783.0 across its five flow rates; the
        crossed-barrel spans 6.0, 200.0, 1.0 and 0.7), so a raw Euclidean distance
        is dominated by whichever feature happens to carry the largest numeric range
        and is dimensionally meaningless (it adds degC^2 to nm^2). Normalizing each
        feature by its span over the candidate set makes every feature contribute
        comparably and makes the lookup invariant to a change of units. The value
        returned is still the raw measured objective of the selected row.
        """
        Xq = (X.to(self._X.dtype) - self._lo) / self._span
        dists = torch.cdist(Xq, self._Xn)
        idx = torch.argmin(dists, dim=1)
        fx = self._y[idx].reshape(-1, 1).to(torch.float64)
        return None, fx
