"""Validate every stored result by RE-EVALUATING it under the current code.

WHY THIS EXISTS (and why it subsumes the commit_sha check)
----------------------------------------------------------
Every ``.npz`` is stamped with the commit that produced it, and that stamp already caught 525
results made against corrupted benchmark definitions. But the SHA is only a *proxy*:

    SHA asks:            "was this made by the right code?"
    re-evaluation asks:  "does this result still describe the problem we have TODAY?"

The second is the question that actually matters, and it is strictly stronger:

  * A result produced at the "right" SHA on a problem that someone LATER fixed is still invalid,
    and the SHA scheme cannot see that.
  * ~295 results predate provenance stamping entirely and have NO SHA. They are not verifiable by
    SHA at all -- but they ARE verifiable this way, so they can be RECOVERED instead of binned.

The check is ground truth: every Result stores both ``X`` and ``y``. Re-evaluate the stored ``X``
with today's code; if it reproduces the stored ``y``, the result is valid BY CONSTRUCTION, whatever
commit produced it. If it does not, the problem changed underneath it and the result is invalid --
no matter how good its provenance looks.

    python validate_results.py                      # report only
    python validate_results.py --quarantine         # move mismatches to Results_INVALID/
    python validate_results.py --sample 200         # spot-check (rows per file)

Run this before ANY analysis or plot.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent
for _c in (_here.parent, _here.parent.parent / "BOCoDe"):
    if (_c / "algorithms").is_dir():
        sys.path.insert(0, str(_c))
        break
import bocode  # noqa: E402
from algorithms._bo_utils import DTYPE, ProblemObjective, penalized  # noqa: E402

ROOT = Path("/home/rosenyu/Documents/Rosen/Bocode_dev")
RESULTS = ROOT / "Results"
QUARANTINE = ROOT / "Results_INVALID" / "INVALID_stale_reeval"

# float32 cast inside BenchmarkProblem.evaluate() means we cannot demand bit-equality.
# Tolerance must sit BETWEEN float32 noise and the smallest real bug we must catch.
#   base.evaluate() casts X to float32 -> relative noise ~1.2e-7 (measured: 1.17e-7 on
#     Walker2DProblem, 1.48e-7 on CEC2020_p1). Anything at or below that is NOT a signal.
#   The HPO-B machine-dependence bug was 2e-3..4e-3 relative.
# 1e-3 (the original) was LOOSER than the bug -> it passed HPOBSurr as "ok" and hid the
# invalidation of 181 results. 1e-9 was BELOW float32 epsilon -> it falsely accused 137 good
# files, including results re-evaluating to the same printed value. 1e-6 is ~10x above the
# noise floor and ~1000x below the bug: it catches the bug class and cannot fire on float noise.
RTOL, ATOL = 1e-6, 1e-8


_DETERMINISM: dict[str, bool] = {}


def is_deterministic(name: str) -> bool:
    """Does this problem return the same y for the same X?

    Re-evaluation can only falsify a result if the objective is deterministic. Some problems
    (Rover, the MuJoCo *PolicySearchProblem variants) roll out an unseeded simulation and return a
    different value every call. For those, a mismatch says nothing about whether the result is
    stale -- it just says the objective is noisy. Reporting them as INVALID would be a false
    accusation, so they are SKIPPED and must be verified some other way (SHA / re-run).
    """
    if name in _DETERMINISM:
        return _DETERMINISM[name]
    try:
        p = bocode.get_problem(name)()
        o = ProblemObjective(p)
        X = torch.rand(3, o.dim, dtype=DTYPE)
        a, _ = o.evaluate_raw(X)
        b, _ = ProblemObjective(bocode.get_problem(name)()).evaluate_raw(X)
        c, _ = o.evaluate_raw(X)
        det = bool(
            np.allclose(a.ravel().detach(), b.ravel().detach(), equal_nan=True)
            and np.allclose(a.ravel().detach(), c.ravel().detach(), equal_nan=True)
        )
    except Exception:  # noqa: BLE001
        det = True  # cannot probe -> fall through to the normal check
    _DETERMINISM[name] = det
    return det


def check(f: Path, n_sample: int) -> tuple[str, str]:
    """Return (verdict, detail). verdict in {ok, MISMATCH, skip, error}."""
    try:
        d = np.load(f, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return "error", f"unreadable: {type(exc).__name__}"

    if "X" not in d or "y" not in d:
        return "skip", "no X/y"
    X = np.asarray(d["X"], dtype=float)
    y = np.asarray(d["y"], dtype=float)
    if X.size == 0 or y.size == 0:
        return "skip", "empty X/y"
    # multi-objective y is (n, m); the single-objective re-evaluation below compares a column.
    if y.ndim == 2 and y.shape[1] > 1:
        return "skip", "multi-objective"

    prob_name = f.parts[-3]
    if not is_deterministic(prob_name):
        return "stochastic", "noisy objective -- re-evaluation cannot verify; check SHA instead"
    try:
        p = bocode.get_problem(prob_name)()
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {str(exc)[:40]}"

    n = min(n_sample, X.shape[0])
    Xs = torch.tensor(X[:n], dtype=DTYPE)

    # CRITICAL: the stored X is in the UNIT CUBE. The algorithms never call
    # problem.evaluate(X) directly -- they go through ProblemObjective, which scales the unit
    # cube into the problem's native bounds first (_scale_clamped). Calling problem.evaluate()
    # on the raw unit-cube X compares apples to oranges for every problem whose bounds are not
    # [0,1] (e.g. AntProblem is [-1,1]), and reports a spurious mismatch. Re-evaluate through
    # the SAME path the run used.
    try:
        obj = ProblemObjective(p)
        raw, cons = obj.evaluate_raw(Xs)
        pen = penalized(raw, cons)
    except Exception as exc:  # noqa: BLE001
        return "error", f"evaluate failed: {type(exc).__name__}"

    # Storage convention differs by algorithm: random_search stores the PENALIZED objective
    # (Y = obj(X)), while the TuRBO/SCBO family store the RAW objective with constraints in a
    # separate `c` array. Accept either -- both are faithful re-derivations of the same run.
    want = (y[:n, 0] if y.ndim == 2 else y[:n]).astype(float)
    for cand in (raw, pen):
        got = cand.ravel()[:n].detach().cpu().numpy().astype(float)
        both = np.isfinite(got) & np.isfinite(want)
        if not both.any():
            return "skip", "all non-finite"
        if np.allclose(got[both], want[both], rtol=RTOL, atol=ATOL):
            return "ok", ""
    got = raw.ravel()[:n].detach().cpu().numpy().astype(float)
    both = np.isfinite(got) & np.isfinite(want)
    i = int(np.argmax(np.abs(got[both] - want[both])))
    return "MISMATCH", f"stored y={want[both][i]:.6g} but re-evaluates to {got[both][i]:.6g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=64, help="rows per file to re-evaluate")
    ap.add_argument("--quarantine", action="store_true")
    a = ap.parse_args()

    files = sorted(RESULTS.glob("*/*/seed*.npz"))
    print(f"validating {len(files)} results by RE-EVALUATION under the current code\n")

    verdicts: Counter = Counter()
    bad: list[tuple[Path, str]] = []
    no_sha = 0
    for f in files:
        try:
            has_sha = "commit_sha" in np.load(f, allow_pickle=True)
        except Exception:  # noqa: BLE001
            has_sha = False
        if not has_sha:
            no_sha += 1
        v, detail = check(f, a.sample)
        verdicts[v] += 1
        if v == "MISMATCH":
            bad.append((f, detail))
            print(f"  ❌ {f.parts[-3]:26s} {f.parts[-2]:20s} {f.name}  {detail}")

    print(f"\n  ok         {verdicts['ok']:5d}   (re-evaluation reproduces the stored y)")
    print(f"  MISMATCH   {verdicts['MISMATCH']:5d}   (the problem CHANGED under it -> INVALID)")
    print(f"  stochastic {verdicts['stochastic']:5d}   (noisy objective -- NOT verifiable this way)")
    print(f"  skipped    {verdicts['skip']:5d}   (multi-objective / empty)")
    print(f"  errors     {verdicts['error']:5d}")
    print(f"  (of all files, {no_sha} carry NO commit_sha and are verifiable ONLY this way)")

    if bad:
        probs = Counter(f.parts[-3] for f, _ in bad)
        print("\n  mismatching problems: " + ", ".join(f"{k}({v})" for k, v in probs.most_common()))
    if a.quarantine and bad:
        for f, detail in bad:
            tgt = QUARANTINE / f.parts[-3] / f.parts[-2]
            tgt.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(tgt / f.name))
        (QUARANTINE / "REASON.txt").write_text(
            "Quarantined by experiments/validate_results.py.\n\n"
            "Re-evaluating the stored X under the CURRENT code does not reproduce the stored y,\n"
            "so the problem definition changed after these results were produced. They describe a\n"
            "benchmark that no longer exists. Some carry a valid-looking commit_sha and some carry\n"
            "none at all -- neither tells you this; only re-evaluation does.\n\n"
            + "\n".join(f"{f.relative_to(RESULTS)}: {d}" for f, d in bad)
            + "\n"
        )
        print(f"\n  quarantined {len(bad)} -> {QUARANTINE}")

    sys.exit(1 if bad and not a.quarantine else 0)


if __name__ == "__main__":
    main()
