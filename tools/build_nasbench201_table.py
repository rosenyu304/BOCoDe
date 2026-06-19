"""Build the compact NAS-Bench-201 accuracy table shipped with BoCoDe.

NAS-Bench-201's official archive is multi-GB; BoCoDe only needs each architecture's
final test accuracy. This script queries the NATS-Bench / NAS-Bench-201 API for all
5^6 = 15,625 architectures and writes a few-MB ``nasbench201_accuracy.npz`` holding
three float arrays (CIFAR-10 / CIFAR-100 / ImageNet16-120 test accuracy), indexed by
the same base-5 edge-operation encoding that ``bocode.opt_problems.nas.NASBench201``
uses. Upload the resulting npz to the bocode-data HF dataset.

One-time setup (heavy download done once, not shipped):
    pip install nats_bench
    # download the topology-space data (NATS-tss-v1_0-3ffb9-simple) per the
    # NATS-Bench README, then:
    python tools/build_nasbench201_table.py --data /path/to/NATS-tss-v1_0-3ffb9-simple

Run output: bocode/opt_problems/nas/data/nasbench201_accuracy.npz
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

# Must match bocode/opt_problems/nas/NASBench201.py exactly.
_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
# edge order: (1<-0), (2<-0), (2<-1), (3<-0), (3<-1), (3<-2)
_PLACE = [5**i for i in range(6)]
_DATASETS = {
    "acc_cifar10": "cifar10",
    "acc_cifar100": "cifar100",
    "acc_imagenet16": "ImageNet16-120",
}


def _arch_str(ops: tuple[int, ...]) -> str:
    o = [_OPS[i] for i in ops]
    return f"|{o[0]}~0|+|{o[1]}~0|{o[2]}~1|+|{o[3]}~0|{o[4]}~1|{o[5]}~2|"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to the NATS-tss data dir/file")
    ap.add_argument("--hp", default="200", help="training budget (epochs) to read")
    args = ap.parse_args()

    from nats_bench import create

    api = create(args.data, "tss", fast_mode=True, verbose=False)
    out = {k: np.zeros(5**6, dtype=np.float64) for k in _DATASETS}

    for ops in itertools.product(range(len(_OPS)), repeat=6):
        idx = sum(op * p for op, p in zip(ops, _PLACE, strict=True))
        api_idx = api.query_index_by_arch(_arch_str(ops))
        for key, dset in _DATASETS.items():
            info = api.get_more_info(api_idx, dset, hp=args.hp, is_random=False)
            out[key][idx] = info["test-accuracy"]

    dest = Path(__file__).resolve().parent.parent / (
        "bocode/opt_problems/nas/data/nasbench201_accuracy.npz"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
