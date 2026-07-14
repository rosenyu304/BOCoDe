"""Phase 2 — compute hypervolume traces OFFLINE, after the runs.

WHY THIS EXISTS
---------------
Computing the hypervolume (HV) *inside* the optimization loop was doing two things
wrong at once:

1. **It was slow enough to break the campaign.** Exact HV is #P-hard in the number
   of objectives; BoTorch's box-decomposition cost explodes with `m`. Measured, on
   one 1,020-point run:

       m=3 (RE33)                 30 ms   per HV call
       m=4 (BotorchCarSideImpact) 74 ms   per HV call
       m=6 (RE61)                698 ms   per HV call

   The loop called it once per iteration, so RE61 cost ~340 s *per seed* — ~700x a
   single-objective problem. With jobs ordered T1..T4, T2 starved T3 and T4: zero
   T3/T4 jobs were reached on any of 5 machines.

2. **It was computing the WRONG number anyway.** Each run inferred its own reference
   point from its own samples, so different algorithms were scored against different
   reference points and their HVs were not comparable. (RE61 reported HV = 1.4e29.)

Both problems vanish if HV is computed after the fact: the runs store the raw
`X`/`y`, so the entire trace can be reconstructed against ONE fixed reference point
per problem.

THE SPEEDUPS
------------
* **pymoo instead of BoTorch** for the HV value. pymoo ships a compiled-C exact HV
  (WFG); it is numerically identical to BoTorch (agreement ~1e-16) and far faster:

       m=3   163x      m=4   224x      m=6   9.5x

* **Skip dominated points.** The HV trace only changes when a new point actually
  joins the Pareto front. Random search draws mostly dominated points, so most
  iterations need no recomputation at all.

* **Process-level parallelism.** Every (problem, algorithm, seed) run is independent
  — embarrassingly parallel. Capped at 12 workers per the compute rules.

* **GPU: not usable, and not worth it.** `torch.cuda.is_available()` is **False** in
  the `bocode` env (torch 2.12+cu130 needs CUDA 13.0; the driver is 12.6), so the GPU
  is unavailable to this env at all. Even with a working driver, exact HV is a
  branch-heavy recursive box decomposition, not a dense tensor kernel — it does not
  map well onto CUDA. The compiled-C WFG path above is the right answer.

REFERENCE POINT
---------------
Comparability requires ONE reference point per problem, shared by every algorithm.
Priority:
  1. ``problem.ref_point`` — a fixed, published value (BoTorch `_ref_point`,
     Tanabe & Ishibuchi's reproblems, pymoo). Preferred.
  2. Otherwise: derive it ONCE from the union of every observed objective vector
     across ALL runs of that problem (all algorithms, all seeds) and record it. This
     is still shared, hence still comparable — but it is data-dependent, so it is
     flagged in the output as ``ref_point_source="derived"``.

Usage::

    python compute_hv.py                 # all multi-objective problems in Results/
    python compute_hv.py --problems RE61 RE33
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
for _cand in (_here / "BOCoDe", _here.parent / "BOCoDe", _here):
    if (_cand / "algorithms").is_dir():
        BOCODE = _cand
        break
else:  # pragma: no cover
    raise SystemExit("cannot locate the BOCoDe repo")
sys.path.insert(0, str(BOCODE))

import bocode  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402

RESULTS = BOCODE / "Results"


def multi_objective_problems() -> list[str]:
    out = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        try:
            md = bocode.get_metadata(d.name)
        except KeyError:
            continue
        if (md.get("num_objectives") or 0) >= 2:
            out.append(d.name)
    return out


def runs_of(problem: str) -> list[Path]:
    return sorted(RESULTS.joinpath(problem).glob("*/seed*.npz"))


def _load_y(f: Path) -> np.ndarray | None:
    d = np.load(f, allow_pickle=True)
    if "y" not in d:
        return None
    y = np.asarray(d["y"], dtype=float)
    return y if y.ndim == 2 else None


def reference_point(problem: str, files: list[Path]) -> tuple[np.ndarray, str]:
    """One reference point per PROBLEM, shared by every algorithm (BoCoDe maximizes)."""
    try:
        rp = getattr(bocode.get_problem(problem)(), "ref_point", None)
    except Exception:  # noqa: BLE001
        rp = None
    if rp is not None:
        return np.asarray(rp, dtype=float).ravel(), "problem.ref_point (fixed/published)"

    # Fallback: derive ONCE from the union of every run of this problem.
    ys = [y for y in (_load_y(f) for f in files) if y is not None]
    if not ys:
        raise ValueError(f"{problem}: no usable y")
    allY = np.concatenate(ys, axis=0)
    lo, hi = allY.min(0), allY.max(0)
    return lo - 0.1 * (hi - lo), "derived (union of all runs; NOT published)"


def hv_trace(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cumulative HV after each evaluation. Recomputes only when the front changes.

    BoCoDe MAXIMIZES; pymoo MINIMIZES -> negate both the points and the ref point.
    """
    ind = HV(ref_point=-ref)
    n = len(y)
    trace = np.zeros(n, dtype=float)
    front: list[np.ndarray] = []   # current non-dominated set (maximization)
    last = 0.0
    for i in range(n):
        p = y[i]
        # dominated by the incumbent front? then HV cannot change -> skip entirely
        if any(np.all(q >= p) and np.any(q > p) for q in front):
            trace[i] = last
            continue
        front = [q for q in front if not (np.all(p >= q) and np.any(p > q))]
        front.append(p)
        last = float(ind(-np.asarray(front)))
        trace[i] = last
    return trace


def process(args) -> dict:
    problem, ref, ref_src = args
    out = []
    for f in runs_of(problem):
        y = _load_y(f)
        if y is None:
            out.append({"file": str(f), "status": "skipped (no 2-D y)"})
            continue
        t0 = time.perf_counter()
        tr = hv_trace(y, ref)
        d = dict(np.load(f, allow_pickle=True))
        d["hv_trace"] = tr
        d["hv_final"] = tr[-1]
        d["ref_point"] = ref
        d["ref_point_source"] = ref_src
        np.savez(f, **d)
        out.append({"file": f.name, "algo": f.parent.name, "hv_final": float(tr[-1]),
                    "secs": round(time.perf_counter() - t0, 2)})
    return {"problem": problem, "ref_point": ref.tolist(),
            "ref_point_source": ref_src, "runs": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)  # compute cap
    ap.add_argument("--out", default=str(_here / "hv_summary.json"))
    a = ap.parse_args()

    probs = a.problems or multi_objective_problems()
    tasks = []
    for p in probs:
        fs = runs_of(p)
        if not fs:
            continue
        ref, src = reference_point(p, fs)
        tasks.append((p, ref, src))

    print(f"HV phase: {len(tasks)} multi-objective problems, {a.workers} workers")
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(process, tasks):
            derived = "DERIVED" if r["ref_point_source"].startswith("derived") else "fixed"
            print(f"  {r['problem']:24s} runs={len(r['runs']):3d}  ref={derived}")
            results.append(r)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"done in {time.perf_counter() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
