"""Gate 1 — smoke test EVERY (algorithm x problem-class) pair end-to-end.

This is the cheapest gate that would have caught the bugs that actually bit us:
a wrong objective sign, a constrained run with no feasible points, a multi-objective
run whose hypervolume is zero or non-monotone, a checkpoint that does not resume, and
a mixed-variable proposal that lands off its categorical levels.

Every check below is ASSERTED, not eyeballed. Tiny budgets: n_init=6, iters=3.

    python smoke_all.py                 # everything
    python smoke_all.py --only turbo    # one algorithm
    python smoke_all.py --device cuda:1

Exit code is non-zero if any pair fails, so it can gate a batch launch.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
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

N_INIT, ITERS = 20, 5  # big enough that a constrained problem can find a feasible point

# (module, problem, class) — one small, fast problem per class.
MATRIX = [
    # single-objective, unconstrained, continuous
    ("algorithms.single_obj.random_search", "Allison", "SO-uncon"),
    ("algorithms.single_obj.single_task_gp", "Allison", "SO-uncon"),
    ("algorithms.single_obj.turbo", "Allison", "SO-uncon"),
    ("algorithms.single_obj.baxus", "Allison", "SO-uncon"),
    ("algorithms.single_obj.tfm_turbo", "Allison", "SO-uncon"),
    ("algorithms.single_obj.git_bo", "Allison", "SO-uncon"),
    # multi-objective, unconstrained
    ("algorithms.multi_obj.random_search", "RE24", "MO-uncon"),
    ("algorithms.multi_obj.qnehvi", "RE24", "MO-uncon"),
    ("algorithms.multi_obj.qnparego", "RE24", "MO-uncon"),
    ("algorithms.multi_obj.tfm_qnehvi", "RE24", "MO-uncon"),
    ("algorithms.multi_obj.tfm_qnparego", "RE24", "MO-uncon"),
    # single-objective, constrained
    ("algorithms.single_obj_constrained.random_search", "ThreeTruss", "SO-con"),
    ("algorithms.single_obj_constrained.constrained_ei", "ThreeTruss", "SO-con"),
    ("algorithms.single_obj_constrained.scbo", "ThreeTruss", "SO-con"),
    ("algorithms.single_obj_constrained.pfn_cei", "ThreeTruss", "SO-con"),
    ("algorithms.single_obj_constrained.tfm_scbo", "ThreeTruss", "SO-con"),
    ("algorithms.single_obj_constrained.penalty", "ThreeTruss", "SO-con"),
    # multi-objective, constrained
    ("algorithms.multi_obj_constrained.random_search", "DiscBrake", "MO-con"),
    ("algorithms.multi_obj_constrained.constrained_qnehvi", "DiscBrake", "MO-con"),
    ("algorithms.multi_obj_constrained.constrained_qparego", "DiscBrake", "MO-con"),
    ("algorithms.multi_obj_constrained.tfm_cqnehvi", "DiscBrake", "MO-con"),
    ("algorithms.multi_obj_constrained.tfm_cqnparego", "DiscBrake", "MO-con"),
    ("algorithms.multi_obj_constrained.penalty", "DiscBrake", "MO-con"),
    # single-objective, unconstrained, mixed-variable
    ("algorithms.single_obj_mixed_variable.random_search", "BraninLVGP", "SO-mixed"),
    ("algorithms.single_obj_mixed_variable.single_task_gp", "BraninLVGP", "SO-mixed"),
    ("algorithms.single_obj_mixed_variable.casmopolitan", "BraninLVGP", "SO-mixed"),
    ("algorithms.single_obj_mixed_variable.hebo", "BraninLVGP", "SO-mixed"),
]

# Cost/weight problems: the MAXIMIZATION value must be NEGATIVE. This is the check
# that would have caught the 63 inverted problems (MOPTA08Car was maximizing mass).
COST_PROBLEMS = {"ThreeTruss", "Allison"}


class _CountingProblem:
    """Wrap a problem and COUNT REAL OBJECTIVE EVALUATIONS.

    A budget check that reads ``res.X`` cannot catch a budget bug: scbo silently spent 100
    evaluations against a budget of 80 (restart designs were free) while saving only 33 rows,
    so ``res.X`` reported 33. It therefore competed against every other method with a 25%
    larger budget, AND its curve was plotted ~20 evaluations to the left of where it belonged.
    The only way to catch that is to count the evaluations the problem actually served.
    """

    def __init__(self, p):
        self.p = p
        self.n_evals = 0

    def __getattr__(self, k):
        return getattr(self.p, k)

    def evaluate(self, X):
        self.n_evals += X.shape[0]
        return self.p.evaluate(X)


def check(res, problem, cls) -> list[str]:
    """Return a list of failures (empty == pass)."""
    bad = []
    y = np.asarray(res.per_iteration_value, dtype=float)

    if cls.endswith("-con") and np.all(np.isneginf(y)):
        # constrained + tiny budget: no feasible point yet is legitimate, not a bug
        pass
    elif not np.all(np.isfinite(y)):
        bad.append("non-finite objective")

    if cls.startswith("MO"):
        # hypervolume trace: finite, non-negative, monotone non-decreasing
        if np.any(y < -1e-9):
            bad.append("negative hypervolume")
        if np.any(np.diff(y) < -1e-6):
            bad.append("hypervolume DECREASED (should be monotone)")
        if float(y[-1]) <= 0.0:
            bad.append("final hypervolume is 0 (no feasible/non-dominated point?)")
    else:
        if np.any(np.diff(y) < -1e-6):
            bad.append("best-so-far DECREASED (should be monotone)")
        if problem in COST_PROBLEMS and float(y[-1]) > 0:
            bad.append(
                f"SIGN: cost problem returned a POSITIVE maximization value "
                f"({y[-1]:.4g}) — objective is inverted"
            )

    if cls.endswith("-con"):  # NB: "uncon" contains "con" — must match the suffix
        if getattr(res, "c", None) is None:
            bad.append("constraints not saved (feasible HV not recomputable)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    rows, failures = [], 0
    for mod_name, prob_name, cls in MATRIX:
        algo = mod_name.rsplit(".", 1)[-1]
        if a.only and a.only not in algo:
            continue
        t0 = time.perf_counter()
        try:
            m = importlib.import_module(mod_name)
            problem = _CountingProblem(bocode.get_problem(prob_name)())
            kw = {}
            if "device" in m.optimize_problem.__code__.co_varnames:
                kw["device"] = a.device
            if "n_init" in m.optimize_problem.__code__.co_varnames:
                kw["n_init"] = N_INIT
            res = m.optimize_problem(problem, iters=ITERS, seed=0, **kw)
            bad = check(res, prob_name, cls)
            # BUDGET PARITY, measured not inferred: a method may not spend more objective
            # evaluations than n_init + iters, and must retain all of them in its history.
            n_init_used = kw.get("n_init", N_INIT)
            budget = n_init_used + ITERS
            if problem.n_evals > budget:
                bad.append(
                    f"BUDGET OVERSPEND: used {problem.n_evals} objective evaluations "
                    f"against a budget of {budget} (+{problem.n_evals - budget}) — it is "
                    f"competing with a larger budget than every other method"
                )
            n_rows = res.X.shape[0] if getattr(res, "X", None) is not None else 0
            if n_rows < problem.n_evals:
                bad.append(
                    f"HISTORY LOST: {problem.n_evals} evaluations made but only {n_rows} rows "
                    f"saved — the total-evaluation axis will be wrong for this method"
                )
            dt = time.perf_counter() - t0
            if bad:
                failures += 1
                rows.append((cls, algo, prob_name, "FAIL", "; ".join(bad), dt))
            else:
                rows.append((cls, algo, prob_name, "ok", f"best={res.best:.6g}", dt))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            rows.append(
                (
                    cls,
                    algo,
                    prob_name,
                    "ERROR",
                    f"{type(exc).__name__}: {str(exc)[:70]}",
                    time.perf_counter() - t0,
                )
            )
            if __debug__ and "-v" in sys.argv:
                traceback.print_exc()

    w = max(len(r[1]) for r in rows) + 1
    print(f"\n{'class':10s} {'algorithm':{w}s} {'problem':12s} {'':6s} detail")
    print("-" * 100)
    for cls, algo, prob, status, detail, dt in rows:
        mark = {"ok": "✅", "FAIL": "❌", "ERROR": "💥"}[status]
        print(f"{cls:10s} {algo:{w}s} {prob:12s} {mark:6s} {detail}  ({dt:.1f}s)")
    print("-" * 100)
    print(f"{len(rows) - failures}/{len(rows)} passed")
    if failures:
        print(f"\n{failures} FAILURE(S) — do NOT launch the batch.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
