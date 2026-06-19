"""Build the compact FixedHPO-B tables bundled in bocode/opt_problems/hpo/data/.

Extracts a fixed subsample of HPO-B configurations for a few representative search
spaces into small CSVs (the HPO-B search space is already [0,1]^d). One-time:

    # download the HPO-B meta-dataset (see https://github.com/SamuelGabriel/FixedHPO-B)
    curl -L "https://github.com/sebastianpinedaar/hpo-data/raw/refs/heads/main/hpob-data.zip?download=" -o hpob-data.zip
    unzip hpob-data.zip
    python tools/build_hpob_tables.py --data /path/to/hpob-data/meta-test-dataset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# (output name, HPO-B search-space id, OpenML dataset id, dim) — all empty log fix.
TASKS = [
    ("hpob_svm", "5527", "10101", 8),
    ("hpob_rpart", "5636", "31", 6),
    ("hpob_ranger", "5965", "9946", 10),
    ("hpob_xgboost", "5971", "6566", 16),
]
MAX_POOL = 2000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to meta-test-dataset.json")
    args = ap.parse_args()

    data = json.load(open(args.data))
    out = Path(__file__).resolve().parent.parent / "bocode/opt_problems/hpo/data"
    out.mkdir(parents=True, exist_ok=True)
    for name, method, dataset, dim in TASKS:
        X = np.asarray(data[method][dataset]["X"], dtype=float)
        y = np.asarray(data[method][dataset]["y"], dtype=float).ravel()
        assert X.shape[1] == dim, (name, X.shape, dim)
        if len(X) > MAX_POOL:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(X), MAX_POOL, replace=False)
            X, y = X[idx], y[idx]
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(dim)])
        df["accuracy"] = y
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(df)} configs, {dim}D)")


if __name__ == "__main__":
    main()
