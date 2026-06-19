"""Build the compact LCBench tables bundled in bocode/opt_problems/hpo/data/.

Extracts the seven tunable hyperparameters + final validation accuracy for each of
2000 configurations, for a few representative LCBench datasets, into small CSVs.
One-time:

    # download the lightweight LCBench data (figshare project 74151, data_2k_lw.zip)
    curl -L https://ndownloader.figshare.com/files/21188598 -o data_2k_lw.zip
    unzip data_2k_lw.zip
    python tools/build_lcbench_tables.py --data /path/to/data_2k_lw.json

Note: LCBench stores num_layers=1 as JSON ``true`` (since 1 == True), so values are
coerced with float() (float(True) == 1.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HP = [
    "batch_size",
    "learning_rate",
    "max_dropout",
    "max_units",
    "momentum",
    "num_layers",
    "weight_decay",
]
DATASETS = ["credit-g", "higgs", "Fashion-MNIST", "albert"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to data_2k_lw.json")
    args = ap.parse_args()

    data = json.load(open(args.data))
    out = Path(__file__).resolve().parent.parent / "bocode/opt_problems/hpo/data"
    out.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        cfgs = data[ds]
        rows = [[float(cfgs[c]["config"][h]) for h in HP] for c in cfgs]
        accs = [float(cfgs[c]["results"]["final_val_accuracy"]) for c in cfgs]
        df = pd.DataFrame(rows, columns=HP)
        df["accuracy"] = accs
        safe = ds.replace("-", "_").replace(".", "_")
        df.to_csv(out / f"lcbench_{safe}.csv", index=False)
        print(f"wrote lcbench_{safe}.csv ({len(df)} configs)")


if __name__ == "__main__":
    main()
