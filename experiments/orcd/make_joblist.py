"""Generate the deterministic job list the watchdog consumes.

One line per (problem, algorithm, seed):

    problem <tab> algo <tab> seed <tab> partition <tab> device <tab> iters <tab> gres

ORDERING (Rosen's rules, applied in this order):
  1. method priority: random_search -> single_task_gp/qnehvi -> TFM -> other GP
  2. LOW DIMENSION FIRST (ascending dim, then ascending #constraints)

ROUTING (measured, not guessed — see Autoresearch_Skill.md §2b):
  * GPU  : one-big-GP and transformer workloads — single_task_gp, turbo, baxus, qnehvi,
           qnparego, the constrained-MO acquisitions, and every TFM method.
           (measured: GPU 2.7x FASTER than CPU for one large GP)
  * CPU  : random_search (no model at all), and anything that fits ONE GP PER CONSTRAINT
           (scbo, penalty). (measured: GPU 2.0x SLOWER for 32 small sequential GP fits —
           they are launch-latency bound and do not belong on a GPU)

`constrained_ei` is DROPPED (Rosen, 2026-07-14): its cost scales with #constraints
(corr 0.988; 33.8 h/run on 88-constraint Truss72D).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _c in (_here.parent.parent / "BOCoDe",):
    if (_c / "algorithms").is_dir():
        sys.path.insert(0, str(_c))
import bocode  # noqa: E402

# Problems that must NOT enter the campaign. Running them wastes compute and pollutes tables.
EXCLUDE = {
    # permutation problems mislabelled continuous; every tour collapses to city 1 (Y == 0)
    "TSP_51Cities", "TSP_100Cities",
    # NOTE: "InvertedPendulumProblem" was excluded here for "reward is exactly 1.0 everywhere".
    # That was a BUG, not a property: all 11 MuJoCo *Problem variants took a single env.step()
    # and never rolled out an episode. Fixed 2026-07-14 -> the problem works and is BACK IN.
    # An exclusion whose cause gets fixed must be removed, or the fix silently does nothing.
    #
    # SCOPE exclusion (Rosen, 2026-07-14) -- NOT a defect:
    "LassoRCV1",  # dim = 47,236 (one Lasso weight per RCV1 TF-IDF feature; next largest is 7,129).
    # No GP method can fit 47k ARD lengthscales, so it would be n/a for nearly every column, and as a
    # single 47k outlier it distorts every metadata plot (the suite is 74% at dim <= 10). It is NOT
    # broken: its old alpha=1.0 bug (1258x alpha_max -> all coefficients zeroed -> the objective
    # ignored x entirely) is FIXED. Dropped on scope. The other Lasso tasks stay.
    # BROKEN feasibility / bounds (pre-existing, documented):
    "TwoBarTruss",               # 0% feasible: bounds bug, documented in its docstring
    # expect unit-cube input while every other problem receives already-scaled X -> RangeException
    "Truss120D", "Truss200D",
    # non-reproducible or needs an uninstalled extra
    "MOPTA08Car",                # native binary, subprocess per eval
    # UNBOUNDED SINGLE EVALUATION (Rosen, 2026-07-14: "ignore these two for now").
    # A single objective evaluation can run for >30 minutes without returning, so a job that
    # lands on such a design hangs until its wall-clock expires. A per-problem time budget
    # cannot rescue it: the budget can only be checked BETWEEN evaluations, and the process is
    # stuck INSIDE one. Same failure mode as SVM.
    "Truss120D",                 # FEM solve does not converge on some geometries
    "SwimmerPolicySearchProblem",  # MuJoCo rollout does not terminate on some policies
    # ALIAS, not a bug (Rosen, 2026-07-14). CEC2020_p31 IS GearTrain: verified numerically
    # identical (max|delta| = 0). The official CEC2020 MATLAB confirms RC31 is genuinely
    # bound-constrained only (g = h = 0 identically) — the paper's Table 3, which lists
    # g=1/h=1, is wrong; the report's own prose only restates the bounds. So there are no
    # constraints to add that would make p31 a distinct problem.
    # We keep GearTrain (which correctly types the gear teeth as INTEGERS) and run it; p31
    # stays in the registry as an alias so the CEC name still resolves, but it is not run
    # separately — doing so would double-count one function in every aggregate table.
    "CEC2020_p31",
}

# method -> (priority, device)   lower priority number = run first
GPU, CPU = "cuda", "cpu"
T1 = [("random_search", 0, CPU), ("single_task_gp", 1, GPU), ("git_bo", 2, GPU),
      ("tfm_turbo", 2, GPU), ("turbo", 3, GPU), ("baxus", 3, GPU)]
T2 = [("mo_random_search", 0, CPU), ("qnehvi", 1, GPU), ("tfm_qnehvi", 2, GPU),
      ("tfm_qnparego", 2, GPU), ("qnparego", 3, GPU)]
T3 = [("con_random_search", 0, CPU), ("pfn_cei", 2, GPU), ("tfm_scbo", 2, GPU),
      ("scbo", 3, CPU), ("penalty", 3, CPU)]           # scbo/penalty: GP-per-constraint -> CPU
T4 = [("mocon_random_search", 0, CPU), ("constrained_qnehvi", 1, GPU),
      ("tfm_cqnehvi", 2, GPU), ("tfm_cqnparego", 2, GPU),
      ("constrained_qparego", 3, GPU), ("mocon_penalty", 3, CPU)]


def tables():
    out = {"T1": [], "T2": [], "T3": [], "T4": []}
    for n in bocode.list_problems(input_type="continuous"):
        if n in EXCLUDE:
            continue
        m = bocode.get_metadata(n)
        no = m.get("num_objectives") or 0
        nc = m.get("num_constraints") or 0
        dim = m.get("dim") or 0
        key = ("T2" if no >= 2 else "T1") if nc == 0 else ("T4" if no >= 2 else "T3")
        out[key].append((n, dim, nc))
    for k in out:
        out[k].sort(key=lambda t: (t[1], t[2]))          # LOW DIM FIRST, then fewest constraints
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--max-problems", type=int, default=None, help="dry run: only the N smallest")
    ap.add_argument("--tables", nargs="+", default=["T1", "T2", "T3", "T4"])
    ap.add_argument("--gpu-partition", default="mit_preemptable")
    ap.add_argument("--cpu-partition", default="mit_normal")
    ap.add_argument("--out", default=str(_here / "joblist.tsv"))
    a = ap.parse_args()

    lo, hi = (a.seeds.split("-") + [a.seeds])[:2]
    seeds = list(range(int(lo), int(hi) + 1))

    tbl = tables()
    methods = {"T1": T1, "T2": T2, "T3": T3, "T4": T4}
    rows = []
    for t in a.tables:
        probs = tbl[t]
        if a.max_problems:
            probs = probs[: a.max_problems]
        for algo, prio, dev in sorted(methods[t], key=lambda x: x[1]):
            for prob, dim, nc in probs:
                for s in seeds:
                    part = a.gpu_partition if dev == GPU else a.cpu_partition
                    gres = "gpu:1" if dev == GPU else ""
                    rows.append((prio, dim, nc, prob, algo, s, part, dev, a.iters, gres))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))           # priority, then dim, then #con
    with open(a.out, "w") as f:
        f.write("# problem\talgo\tseed\tpartition\tdevice\titers\tgres\n")
        for _, _, _, prob, algo, s, part, dev, it, gres in rows:
            f.write(f"{prob}\t{algo}\t{s}\t{part}\t{dev}\t{it}\t{gres}\n")

    n_gpu = sum(1 for r in rows if r[7] == GPU)
    print(f"wrote {len(rows)} jobs -> {a.out}")
    print(f"  GPU jobs: {n_gpu}   CPU jobs: {len(rows) - n_gpu}")
    for t in a.tables:
        print(f"  {t}: {len(tbl[t]) if not a.max_problems else min(a.max_problems, len(tbl[t]))} problems")


if __name__ == "__main__":
    main()
