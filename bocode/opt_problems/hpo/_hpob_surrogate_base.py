"""Continuous-surrogate HPO-B base (Rosen 2026-07-07).

🔴 DO NOT USE FOR BENCHMARKING. WITHDRAWN 2026-07-14. NOT IN THE JOB QUEUE. 🔴

    This benchmark is INVALID and cannot be repaired by freezing the surrogate. The XGBoost
    surrogate is continuous over [0,1]^d and EXTRAPOLATES ABOVE EVERY POINT IT WAS TRAINED ON, so
    optimizers report objective values for hyperparameter configurations THAT DO NOT EXIST.

    MEASURED over all 161 HPOBSurr runs, against the problem's own `values` pool as ground truth:
        134/161 runs (83%) finish ABOVE the true maximum of the config pool.
        worst: HPOBSurr_5970_3492 turbo  best=0.8512 vs pool_max=0.7464  (+0.1048)
        by algo: random_search 50, single_task_gp 50, turbo 33, baxus 1
    RANDOM SEARCH does it in 50 runs -- the over-prediction regions are a LARGE FRACTION of the
    space, not a corner an optimizer sneaks into. So this task measures WHO BEST EXPLOITS THE
    SURROGATE'S EXTRAPOLATION ERROR, not who tunes hyperparameters. It also makes the HPO-B /
    PFNs4BO normalized regret go NEGATIVE, which is impossible for an honest benchmark.

    CLAMPING predictions to the pool's [min, max] is NOT a fix: the argmax then becomes a plateau
    of artifacts, and the benchmark still is not measuring hyperparameter tuning.

    USE THE DISCRETE `HPOB_*` POOLS INSTEAD (`_hpob_base.py`): exact table lookup, so exceeding the
    pool max is impossible BY CONSTRUCTION. Verified: 0/4800 random probes and 0/198 existing
    results exceed pool_max. This kills BOTH HPO-B bugs at once -- no runtime training (no machine
    dependence) and no continuous extrapolation (no fake optima).

    The premise below -- that the discrete pools are too FLAT to be discriminative -- does not hold
    at suite level: 76 of 92 discrete pools have span > 0.05. A flat-but-HONEST benchmark is
    publishable; one where every method beats the true optimum is not.

    (The machine-dependence bug -- the surrogate was trained at RUNTIME, so a different xgboost
    build gave a different objective; e.g. xgboost 3.3.0 vs 3.2.0 differ by up to 4e-3 on the
    initial design -- is separately fixed below by shipping a pre-trained booster. That fix is
    correct and verified bit-identical across versions, but it only freezes a function whose
    OPTIMA ARE ARTIFACTS. It is retained so the class still loads reproducibly, not as a licence
    to benchmark with it.)

--- original rationale, retained for the record; superseded by the above ---

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

from pathlib import Path

import numpy as np
import torch

from ._hpob_base import HPOBProblem

# Pre-trained boosters, shipped alongside hpob_data.npz. Built by build_hpob_surrogates.py.
SURROGATE_BUNDLE = Path(__file__).parent / "data" / "hpob_surrogates.npz"

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
        self._bst = None  # loaded from the SHIPPED bundle on first evaluate

    def _surrogate(self):
        """Load the pre-trained, SHIPPED booster. Never train at runtime.

        Training the surrogate at runtime made the objective MACHINE-DEPENDENT: a booster fitted by
        a different xgboost build predicts slightly differently, so `f` itself differed between
        workers (measured 2026-07-14: xgboost 3.3.0 here vs 3.2.0 on the workers -> up to 4e-3
        disagreement on the INITIAL DESIGN, before any BO). That silently invalidated 181 results.
        The booster is now trained once by build_hpob_surrogates.py and shipped, exactly as
        hpob_data.npz is, so every machine evaluates a byte-identical objective.
        """
        if self._bst is None:
            import xgboost as xgb

            name = type(self).__name__
            if not SURROGATE_BUNDLE.exists():
                raise FileNotFoundError(
                    f"{SURROGATE_BUNDLE} is missing. HPO-B surrogates must be pre-trained and "
                    f"shipped, not fitted at runtime (the runtime fit is machine-dependent). "
                    f"Regenerate with: python -m bocode.opt_problems.hpo.build_hpob_surrogates"
                )
            with np.load(SURROGATE_BUNDLE) as bundle:
                if name not in bundle:
                    raise KeyError(
                        f"no pre-trained surrogate for {name} in {SURROGATE_BUNDLE}; "
                        f"regenerate with build_hpob_surrogates.py"
                    )
                blob = bytearray(bundle[name].tobytes())
            self._bst = xgb.Booster()
            self._bst.load_model(blob)
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
