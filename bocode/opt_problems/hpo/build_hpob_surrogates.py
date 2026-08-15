"""Pre-train the HPO-B XGBoost surrogates ONCE and serialize them for shipping.

WHY THIS EXISTS (2026-07-14). `_hpob_surrogate_base.py` used to call `xgb.train(...)` lazily at
runtime, on every machine. That is deterministic *within* one environment but NOT across
environments: a booster trained by a different xgboost build makes slightly different predictions,
so THE OBJECTIVE FUNCTION ITSELF DIFFERED BETWEEN MACHINES.

Measured: one machine ran xgboost 3.3.0, others 3.2.0. Results produced on one build and
re-evaluated on the other disagreed by up to 4e-3 ON THE INITIAL DESIGN -- i.e. before any BO
had happened. 181 results were invalidated by this.

Pinning the xgboost version would be a weaker patch: it leaves the bug class alive for the next
upgrade. Instead we train once, serialize the booster, and ship it, exactly as the repo already
ships hpob_data.npz. Every machine then loads a byte-identical model and evaluates the SAME
objective.

Regenerate (only if the pool data or the XGB recipe changes):

    python -m bocode.opt_problems.hpo.build_hpob_surrogates
"""

from __future__ import annotations

import numpy as np

from ._hpob_surrogate_base import _NUM_ROUND, _XGB_PARAMS, SURROGATE_BUNDLE


def main() -> None:
    import xgboost as xgb

    import bocode

    tasks = sorted(t for t in bocode.list_problems() if t.startswith("HPOBSurr_"))
    print(f"training {len(tasks)} HPO-B surrogates with xgboost {xgb.__version__}")

    blobs: dict[str, np.ndarray] = {}
    for i, name in enumerate(tasks, 1):
        p = bocode.get_problem(name)()
        dtrain = xgb.DMatrix(
            p._X.detach().cpu().numpy(), label=p._y.detach().cpu().numpy()
        )
        bst = xgb.train(_XGB_PARAMS, dtrain, num_boost_round=_NUM_ROUND)
        # JSON is xgboost's portable, version-stable model format -- the point of the exercise is
        # that a 3.2.0 build must load this and predict identically to the 3.3.0 build that wrote it.
        blobs[name] = np.frombuffer(
            bytes(bst.save_raw(raw_format="json")), dtype=np.uint8
        )
        print(f"  [{i:3d}/{len(tasks)}] {name:28s} {blobs[name].nbytes / 1024:8.1f} KB")

    np.savez_compressed(SURROGATE_BUNDLE, **blobs)
    total = sum(v.nbytes for v in blobs.values()) / 1e6
    print(f"\nwrote {SURROGATE_BUNDLE}  ({len(blobs)} models, {total:.1f} MB raw)")


if __name__ == "__main__":
    main()
