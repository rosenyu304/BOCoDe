"""Continuous-surrogate HPO-B base (Rosen 2026-07-07).

The discrete-pool :class:`HPOBProblem` evaluates by nearest-config lookup over the recorded
pool. For continuous-relaxation / mixed-integer BO this *compresses the landscape*: a proposed
config is snapped to the nearest recorded config, which is almost always high-accuracy, so the
sampled objective collapses to a narrow high band and BO convergence looks FLAT (verified: a GP
improved ~0.000 over 200 iters on saturated tasks).

This variant instead fits an XGBoost regression surrogate ``f:[0,1]^d -> accuracy`` on the SAME
pool ``(X, y)`` and evaluates *that* (with integer/discrete dims snapped to their allowed levels
via :meth:`enforce_variable_types`). Because the surrogate is trained on the whole pool it also
predicts the true LOW-accuracy regions, giving a smooth **mixed-integer** landscape with real,
non-flat convergence — matching HPO-B's own continuous-surrogate mode and the PFNs4BO protocol.
Deterministic (fixed xgboost seed); same search space / variable types as the discrete pool.
"""

from __future__ import annotations

import numpy as np
import torch

from ._hpob_base import HPOBProblem

# Same recipe used to (re)build the HPO-B continuous surrogates; modest depth generalizes the
# discrete pool smoothly. Deterministic.
_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "seed": 0,
}
_NUM_ROUND = 200


class HPOBSurrogateProblem(HPOBProblem):
    """HPO-B task evaluated by an XGBoost surrogate over the pool (not nearest-config lookup).

    Identical search space, bounds, and (mixed) variable types as :class:`HPOBProblem`; only the
    objective differs. Maximize validation accuracy. Non-flat / discriminative for BO.
    """

    def __init__(self) -> None:
        super().__init__()
        self._bst = None  # lazily trained on first evaluate (uses self._X, self._y from the pool)

    def _surrogate(self):
        if self._bst is None:
            import xgboost as xgb

            dtrain = xgb.DMatrix(
                self._X.detach().cpu().numpy(), label=self._y.detach().cpu().numpy()
            )
            self._bst = xgb.train(_XGB_PARAMS, dtrain, num_boost_round=_NUM_ROUND)
        return self._bst

    def _evaluate_implementation(self, X: torch.Tensor) -> tuple:
        import xgboost as xgb

        Xs = self.enforce_variable_types(
            X.to(self._X.dtype)
        )  # snap integer/discrete dims
        pred = self._surrogate().predict(xgb.DMatrix(Xs.detach().cpu().numpy()))
        y = torch.tensor(
            np.asarray(pred, dtype=np.float64), dtype=torch.float64
        ).reshape(-1, 1)
        return None, y
