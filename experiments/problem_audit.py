"""PRE-FLIGHT PROBLEM AUDIT — run this before spending a single cluster-hour.

A benchmark can be *broken* in ways that look like an algorithm failure:

  * a constraint that is **never satisfiable** -> the problem has NO feasible region and is
    literally unsolvable (CantileverBeam had this; CEC2020_p20 has it);
  * a constraint that is **constant** -> dead code, contributes nothing;
  * an objective or constraint that is a **step function with a handful of distinct values**
    -> a GP cannot be fit to it, which surfaces as `ModelFittingError` and gets misdiagnosed
    as a bug in the optimizer (this is exactly what happened with scbo on CEC2020_p20);
  * an objective that is **effectively constant** -> nothing to optimize (QPowerModel,
    InvertedPendulumProblem).

Every one of those was found BY ACCIDENT in this project, after burning compute. This script
finds them on purpose, in minutes, before the batch.

The usual cause of the step functions is an infinity/NaN sentinel, e.g.
`bocode/opt_problems/cec2020_rw/CEC2020_p1_20.py:1531`:

    g[np.isinf(g)] = 1e6
    g[np.isnan(g)] = 1e6

which collapses a whole region of the constraint into one value.

    python problem_audit.py                    # every registered problem
    python problem_audit.py --constrained      # just T3/T4
    python problem_audit.py --n 500

Exit code is non-zero if any problem is UNSOLVABLE, so this can gate a launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
for _c in (_here.parent, _here.parent.parent / "BOCoDe"):
    if (_c / "algorithms").is_dir():
        sys.path.insert(0, str(_c))
        break
import bocode  # noqa: E402

# A "few distinct values" means a GP cannot model it. Chosen generously: a genuinely smooth
# function of a continuous input gives ~n distinct values out of n samples.
STEP_THRESHOLD = 10


def audit(name: str, n: int) -> dict:
    p = bocode.get_problem(name)()
    X = p.sample(n, seed=0)
    values, cons = p.evaluate(X)

    r: dict = {
        "problem": name,
        "dim": p.dim,
        "n_con": p.num_constraints,
        "verdict": "ok",
        "issues": [],
    }

    y = values[:, 0]
    n_uy = int(torch.unique(y).numel())
    r["obj_distinct"] = n_uy
    if not torch.isfinite(y).all():
        r["issues"].append("objective has NaN/inf")
    if n_uy == 1:
        r["issues"].append("objective is CONSTANT — nothing to optimize")
    elif n_uy <= STEP_THRESHOLD:
        r["issues"].append(
            f"objective is a STEP FUNCTION ({n_uy} distinct values) — GP cannot fit it"
        )

    if p.num_constraints:
        feas_each = []
        for j in range(cons.shape[1]):
            cj = cons[:, j]
            u = int(torch.unique(cj).numel())
            sat = float((cj <= 0).float().mean())
            feas_each.append(sat)
            if u == 1:
                r["issues"].append(
                    f"c{j} is CONSTANT ({cj[0].item():.4g}) — dead constraint"
                )
            elif u <= STEP_THRESHOLD:
                # A COARSE constraint is a WARNING, not a disqualification. A problem with one
                # low-cardinality constraint among many smooth ones, and a real feasible region,
                # is HARD -- not broken. (Firing on this alone would have deleted all 15 MODAct
                # problems, i.e. more than half of Table 4.)
                r["issues"].append(
                    f"c{j} is COARSE ({u} distinct values) — GP fit may be poor (warning only)"
                )
            # CRITICAL DISTINCTION:
            #   * a STEP-FUNCTION constraint (few distinct values) that is never <= 0 can NEVER
            #     be satisfied -- that is STRUCTURAL, the feasible region is empty. Unsolvable.
            #   * a SMOOTH constraint that merely happens to be > 0 in this sample may still have
            #     a tiny feasible region that random sampling missed. That is HARD, not broken --
            #     and CEC2020 problems are famously like this. Do NOT exclude those.
            if sat == 0.0:
                if u <= STEP_THRESHOLD:
                    r["issues"].append(
                        f"c{j} takes only {u} distinct values, none <= 0 "
                        f"(min={float(cj.min()):.4g}) — the feasible set is EMPTY: UNSOLVABLE"
                    )
                else:
                    r["issues"].append(
                        f"c{j} not satisfied in {n} samples (min={float(cj.min()):.4g}, "
                        f"{u} distinct) — feasible region may be tiny (HARD, not broken)"
                    )
        feasible = int((cons <= 0).all(dim=1).sum())
        r["feasible"] = f"{feasible}/{n}"
        if feasible == 0:
            r["issues"].append(
                f"no feasible point in {n} LHS samples (may just be a tiny feasible region)"
            )

    # verdict
    # EXCLUDE a problem ONLY on structural evidence that it cannot be optimized:
    #   (a) a constraint is CONSTANT                      -> dead constraint
    #   (b) a constraint is provably never satisfiable    -> feasible set is EMPTY
    #   (c) the OBJECTIVE is near-discrete                -> nothing for a GP to model
    # NOT excluded: a single coarse constraint, or "no feasible point found by random sampling"
    # (that is a tiny feasible region -- finding it is the whole point of constrained BO).
    if any(("EMPTY" in i) or ("CONSTANT — nothing" in i) for i in r["issues"]):
        r["verdict"] = "UNSOLVABLE"
    elif any(i.startswith("objective is a STEP FUNCTION") for i in r["issues"]):
        r["verdict"] = "DEGENERATE"
    elif any("is CONSTANT" in i for i in r["issues"]):
        r["verdict"] = "dead-constraint"  # warn; does not block optimization
    elif r["issues"]:
        r["verdict"] = "hard"  # coarse constraint and/or tiny feasible region — KEEP
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--constrained", action="store_true", help="only T3/T4")
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--out", default=str(_here / "problem_audit.json"))
    a = ap.parse_args()

    names = a.problems or bocode.list_problems(input_type="continuous")
    if a.constrained:
        names = [
            n for n in names if (bocode.get_metadata(n).get("num_constraints") or 0) > 0
        ]

    bad, out = [], []
    for nm in sorted(names):
        try:
            r = audit(nm, a.n)
        except Exception as exc:  # noqa: BLE001
            r = {
                "problem": nm,
                "verdict": "ERROR",
                "issues": [f"{type(exc).__name__}: {exc}"],
            }
        out.append(r)
        if r["verdict"] != "ok":
            bad.append(r)
            mark = {
                "UNSOLVABLE": "🔴",
                "DEGENERATE": "🟠",
                "dead-constraint": "🟡",
                "hard": "  ",
                "suspect": "🟡",
                "ERROR": "💥",
            }.get(r["verdict"], "?")
            print(
                f"{mark} {r['problem']:22s} {r['verdict']:11s} {r.get('feasible', ''):>9s}"
            )
            for i in r["issues"]:
                print(f"      - {i}")

    Path(a.out).write_text(json.dumps(out, indent=1))
    from collections import Counter

    cnt = Counter(r["verdict"] for r in out)
    n_unsolv = cnt["UNSOLVABLE"] + cnt["DEGENERATE"] + cnt["ERROR"]
    print(
        f"\naudited {len(out)}: "
        + ", ".join(f"{v} {k}" for k, v in sorted(cnt.items()))
    )
    print(f"  -> EXCLUDE {n_unsolv} (UNSOLVABLE + DEGENERATE + ERROR)")
    print(
        f"  -> KEEP    {cnt['ok'] + cnt['hard'] + cnt['dead-constraint']} "
        f"(ok + hard + dead-constraint: coarse constraints / tiny feasible regions are NOT broken)"
    )
    print(f"wrote {a.out}")
    sys.exit(1 if n_unsolv else 0)


if __name__ == "__main__":
    main()
