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

`constrained_ei` is BACK IN (2026-07-14, second pass). It was dropped because its cost scales
with #constraints (corr 0.988; 33.8 h/run on the 88-constraint Truss72D) -- but that was a
decision made when the only compute we were using was contended GPU. It is a CPU-favoured method
(one GP per constraint), and mit_normal's 96 cores were sitting COMPLETELY IDLE. Spending idle
CPU on it is free, and the 12h wall is survivable because the run is checkpointed and requeued.
WATCH Truss72D/Truss120D specifically: those are the runs that will need several requeues.

The `partition` column is now LEGACY and is IGNORED by watchdog v2, which picks the partition at
submit time from the row's DEVICE, filling whichever partition has headroom. Hard-assigning a
partition per row is what queued 56 jobs behind ourselves on pi_faez while three other partitions
sat empty.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
# The repo lives in a different place depending on where this generator is invoked from.
# Handle every layout so ``bocode`` imports and EXCLUDED_PROBLEMS.txt is found regardless:
#   in-repo   : <repo>/experiments/orcd/make_joblist.py            -> ../..            (repo root)
#   campaign  : <...>/Bocode_dev/2026_07_Experiment/orcd/make_joblist.py -> ../../BOCoDe
#   cluster   : /orcd/data/faez/001/rosen/bocode/make_joblist.py   -> ./BOCoDe
# The repo root is whichever candidate contains both ``algorithms/`` and ``bocode/``.
REPO = None
for _c in (_here.parent.parent, _here / "BOCoDe", _here.parent.parent / "BOCoDe"):
    if (_c / "algorithms").is_dir() and (_c / "bocode").is_dir():
        sys.path.insert(0, str(_c))
        REPO = _c
        break
import bocode  # noqa: E402

def _audit_exclusions() -> set[str]:
    """Problems the PRE-FLIGHT AUDIT proved are broken (experiments/problem_audit.py).

    These are BROKEN BENCHMARKS, not infrastructure failures. Letting the watchdog retry them
    and file them in FAILED.tsv would misattribute a problem-side defect to the optimizer --
    which is exactly what happened when CEC2020_p20's empty feasible set surfaced as an
    scbo ModelFittingError.
    """
    if REPO is None:
        return set()
    f = REPO / "experiments" / "EXCLUDED_PROBLEMS.txt"
    if not f.exists():
        return set()
    return {ln.strip() for ln in f.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}


# Problems that must NOT enter the campaign. Running them wastes compute and pollutes tables.
# EVERY entry re-verified empirically on 2026-07-14 (64 random points via ProblemObjective, plus a
# dense scan where the dimension allowed it). An exclusion is a scientific claim; a stale reason is
# a silent lie, and an exclusion whose cause has been FIXED must be REMOVED or the fix does nothing.
EXCLUDE = {
    # TSP: the old reason ("every tour collapses to city 1, Y == 0") is FIXED and FALSE -- the
    # objective now varies (spread 646 / 1151, 56 / 63 distinct over 64 points). It stays excluded for
    # a DIFFERENT, still-valid reason, stated in TSP.py itself: the PERMUTATION CONSTRAINT that
    # defines the TSP is not enforced anywhere, so a continuous/categorical relaxation does not
    # produce valid tours. A varying objective is NOT evidence of validity here.
    "TSP_51Cities", "TSP_100Cities",
    # TwoBarTruss: VERIFIED still broken, but the reason was imprecise. Dense 400x400 scan of the
    # whole unit cube: 0 / 160,000 feasible. Constraint 3 is violated at 100% of points, yet the
    # minimum violation is 1.01e-06 -- the feasible set is missed INFINITESIMALLY. Cause (already in
    # its docstring): bounds are [(0,1),(0,1)] while the constraints require x1 >= 0.1 and x2 >= 1.0,
    # so x2 >= 1.0 is attainable only exactly at the upper bound. The feasible set lies OUTSIDE the
    # box. FIXABLE by taking the real bounds from Rao (1987) -- do NOT guess them; guessing bounds
    # fabricates a benchmark.
    "TwoBarTruss",
    # LassoRCV1: SCOPE, not correctness (Rosen's call). dim = 47,236; no GP fits 47k ARD lengthscales
    # and it distorts every metadata plot. Its old alpha=1.0 bug IS fixed -- do not "fix it back in".
    "LassoRCV1",
    # Truss200D: EXCLUDED ON ROSEN'S EXPLICIT ORDER (2026-07-14), not on reproduced evidence.
    # Reported reason was "FEM solve is UNBOUNDED -- a single eval ran >30 min on a 500-row LHS
    # geometry". THAT DID NOT REPRODUCE HERE: 60 LHS rows evaluated post-fix, none took >5 s.
    # Most likely the hang predates the double-scaling fix (scaling=False) -- badly-scaled inputs
    # drive the FEM solver into pathological geometry. Recording this so the exclusion can be
    # revisited on evidence rather than inherited as folklore. Truss120D is UNAFFECTED and stays IN.
    "Truss200D",
    # PowerElectronics: EXCLUDED (Rosen, 2026-07-16). Needs the optional 'engibench' extra + a native
    # ngspice (v42-v45), neither installed on the ORCD cluster, so the problem cannot even be
    # instantiated -- EVERY method (qnehvi/qnparego/mesmo/...) fails at construction and produces 0
    # results. A dependency gap, not a benchmark defect; re-include if engibench + ngspice get set up.
    "PowerElectronics",
}
# REMOVED FROM EXCLUDE on 2026-07-14, because the bugs behind them are fixed and verified:
#   InvertedPendulumProblem -- "gym reward is exactly 1.0 everywhere" was the SINGLE-TIMESTEP bug.
#       Episodes now roll out: spread 8.0, 8 distinct values over 64 points. BACK IN.
#   Truss120D / Truss200D  -- "expect unit-cube input -> RangeException" was the double-scaling bug,
#       fixed by scaling=False. They now evaluate cleanly: 98.4% / 100% feasible, spread 1.8e4 / 3.3e4.
#   MOPTA08Car             -- the native binary RUNS (spread 47.08, 64 distinct). 0% feasible on a
#       random sample is EXPECTED for MOPTA08 (124-D, 68 constraints) and is what constrained BO is
#       for; it is not evidence of breakage. BACK IN (may want a larger n_init).
EXCLUDE |= _audit_exclusions()

# method -> (priority, device)   lower priority number = run first
#
# PRIORITY (Rosen, 2026-07-14): the methods with ZERO or near-zero results come FIRST. A paper
# needs every method, not one with hundreds of seeds. random_search/single_task_gp already have
# 760/115 results, so they are DEMOTED to the back of the queue; the starved methods lead.
GPU, CPU = "cuda", "cpu"
T1 = [("git_bo", 0, GPU), ("tfm_turbo", 0, GPU), ("turbo", 1, GPU), ("baxus", 1, GPU),
      ("single_task_gp", 8, GPU), ("random_search", 9, CPU)]
T2 = [("qnparego", 0, GPU), ("mesmo", 0, GPU), ("tfm_qnehvi", 1, GPU),
      ("tfm_qnparego", 1, GPU), ("qnehvi", 2, GPU), ("mo_random_search", 9, CPU)]
T3 = [("scbo", 0, CPU), ("pfn_cei", 0, GPU), ("penalty", 0, CPU),
      ("constrained_ei", 0, CPU), ("tfm_scbo", 1, GPU), ("con_random_search", 9, CPU)]
T4 = [("constrained_qnehvi", 0, GPU), ("constrained_qparego", 0, GPU),
      ("tfm_cqnehvi", 1, GPU), ("tfm_cqnparego", 1, GPU),
      ("mocon_penalty", 2, CPU), ("mocon_random_search", 9, CPU)]

# DEVICE ROUTING is MEASURED, twice, independently (see COLLEAGUE_HANDOFF.md):
#   CPU  — scbo, penalty, mocon_penalty, constrained_ei, every random_search. These fit ONE GP PER
#          CONSTRAINT: dozens of tiny sequential fits that are launch-latency bound. The GPU never
#          gets fed and is 2.0x SLOWER. They also give us something to put on mit_normal's 96
#          otherwise-idle cores.
#   GPU  — one-big-GP and transformer workloads (2.7-4.6x faster there).


# ---------------------------------------------------------------------------------------------
# WITHDRAWN / PAUSED / EXCLUDED (Rosen, 2026-07-14). ~3,120 rows of the previous joblist were work
# we had ALREADY DECIDED NOT TO DO, and the watchdog was burning its 3 retries on each of them.
DROP_PREFIX = (
    "HPOBSurr",   # WITHDRAWN: the surrogate extrapolates above the true pool optimum in 83% of runs
    "SVM",        # dropped by Rosen
    "TSP_",       # permutation constraint is not enforced -> a relaxation does not give valid tours
)
DROP_SUITE = {
    "Mixed-variable synthetic",   # PAUSED: synthetics run at the END, after the real-world tables
}
DROP_NAMES = {
    "Truss120D", "Truss200D",          # FEM does not converge on some geometries
    "SwimmerPolicySearchProblem",      # MuJoCo rollout does not terminate on some policies
    "LassoRCV1",                       # dim 47,236 -- no GP fits it; scope call
    "TwoBarTruss",                     # 0% feasible (bounds bug)
    "MOPTA08Car",                      # native binary, subprocess per eval
}

# DUPLICATE FUNCTIONS (Rosen's order). The same function under two names wastes compute AND
# double-counts one problem in every aggregate table. Keep the CEC / benchmark-paper name.
#
# NB the car-side-impact group: BotorchCarSideImpact / CarSideImpact are ALIASES and go. RE41 and
# CRE31 are KEPT BOTH -- they are the paper names and they live in DIFFERENT tables (RE41 is the
# unconstrained multi-objective form, CRE31 the constrained one). Collapsing them to one would
# silently delete an entire constrained-multi-objective row from the results.
DROP_DUPLICATE = {
    "BotorchCarSideImpact", "CarSideImpact",   # == RE41 / CRE31
    "VehicleSafety",        # == RE34
    "DiscBrake",            # == CRE23
    "ThreeTruss",           # == CEC2020_p20
    "GoldsteinMixed",       # == GoldsteinLVGP
    "SteppedCantileverBeam",# == CantileverBeam
    # CEC2020_p31 is an ALIAS of GearTrain, and it is p31 that goes -- NOT GearTrain (Rosen,
    # 2026-07-14). The official CEC2020 MATLAB RC31 is bit-identical to our p31 and is genuinely
    # bound-constrained only (the paper's Table 3 claiming g=1,h=1 is wrong). We keep GearTrain
    # because it types the gear teeth as INTEGERS; p31 types them continuous, which would let an
    # optimizer propose 34.7 teeth. (p17/p21/p31 RESULTS remain valid -- their math never changed.)
    "CEC2020_p31",
    # The water problem: WaterProblem's f3 was 1e-6x too small and it has been removed upstream;
    # WaterResources had 3 wrong constraints (one scaled 0.1x, one SIGN-FLIPPED, one mistyped) and,
    # once fixed, is identical to CRE51. Keep the CRE benchmark-paper name as the representative.
    # Both are excluded here explicitly so the joblist is correct even though the CLUSTER REPO
    # (6cdca29) does not yet carry those upstream fixes -- see the note in the handoff.
    "WaterResources",       # == CRE51
    "WaterProblem",         # removed upstream (f3 scale bug); was a duplicate of WaterResources
}
# KEPT ON PURPOSE (same formula, DIFFERENT boxes -> genuinely different search domains):
#   CEC2020_p17 / CompressionSpring ; EulerBeamMixed / EulerBernoulliBeamBending

# MANY-OBJECTIVE (m >= 5): exact hypervolume is combinatorially intractable (a timing script HUNG on
# it). Every HV-based acquisition is skipped on these; the SCALARIZING ones still work because their
# cost is independent of the objective count. Those cells are reported "intractable at m>=5", not
# left blank. Applies to RE91(9), RE61(6), CRE51/CTSEI1-4/Mazda/WaterProblem(5).
HV_METHODS = {"qnehvi", "tfm_qnehvi", "constrained_qnehvi", "tfm_cqnehvi", "mesmo"}
MANY_OBJ_MIN = 5

# TFM (TabPFN) METHODS -> H100/H200 ONLY, ONE PER CARD (Rosen, emphatic, 2026-07-14).
# pfn_cei feeds TabPFN ONE TARGET COLUMN PER CONSTRAINT in a single forward pass, m = 1 + n_con. On
# CEC2020_p35 (148 constraints) that is a 149-column x 1000-candidate activation tensor: it does not
# fit in a 24 GB card and it OOMed on a 4090 (tried to allocate 7.09 GiB of a 23.55 GiB card that was
# already sharing 18 GiB with 8 packed campaign processes). It fits in an 80 GB H100/H200.
# So: pi_faez only (node2900 8xH100, node2435 4xH100, node4002 4xH200), and NEVER packed --
# packing is what turned a tight fit into an OOM. Packing stays on for the small GP methods only.
TFM_METHODS = {"git_bo", "pfn_cei", "tfm_turbo", "tfm_scbo",
               "tfm_qnehvi", "tfm_qnparego", "tfm_cqnehvi", "tfm_cqnparego"}


def _dropped(name: str, meta: dict) -> bool:
    if name in DROP_NAMES or name in DROP_DUPLICATE or name in EXCLUDE:
        return True
    if any(name.startswith(p) for p in DROP_PREFIX):
        return True
    return str(meta.get("suite")) in DROP_SUITE


def tables():
    out = {"T1": [], "T2": [], "T3": [], "T4": []}
    # ALL REAL-WORLD problems (Rosen, 2026-07-14): single + multi objective, un- + constrained,
    # CONTINUOUS *AND* MIXED. The synthetics are paused and are run at the end -- they are not in
    # list_problems() anyway (they live in bocode/synthetic/ and resolve only via get_problem).
    #
    # Mixed-variable problems used to be silently dropped here by `input_type="continuous"`, which
    # cut the whole mixed column out of the campaign. The continuous algorithms DO run on them:
    # base.enforce_variable_types() rounds ints and snaps categoricals, so a continuous proposal is
    # projected onto the mixed domain. That is the standard continuous-relaxation baseline.
    for n in bocode.list_problems():
        m = bocode.get_metadata(n)
        if _dropped(n, m):
            continue
        no = m.get("num_objectives") or 0
        nc = m.get("num_constraints") or 0
        dim = m.get("dim") or 0
        key = ("T2" if no >= 2 else "T1") if nc == 0 else ("T4" if no >= 2 else "T3")
        out[key].append((n, dim, nc, no))
    for k in out:
        out[k].sort(key=lambda t: (t[1], t[2]))          # LOW DIM FIRST, then fewest constraints
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    # ITERATION BUDGET = 250 (Rosen, 2026-07-14; .claude/Autoresearch_Skill.md). Supersedes 1000/500.
    # At the concurrency ORCD actually gives us, 1000 iterations was weeks of wall clock and was never
    # going to finish. The 724 completed runs are NOT re-run: their curves are >= 251 long, so they are
    # SUPERSETS of a 250-iteration run and get TRUNCATED at analysis time (per_iteration_value[:n_init+250]).
    # The budget must be EQUAL across methods on a problem or the comparison is void.
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--max-problems", type=int, default=None, help="dry run: only the N smallest")
    ap.add_argument("--tables", nargs="+", default=["T1", "T2", "T3", "T4"])
    # Every GPU row used to be hard-assigned to ONE partition, so we queued behind ourselves on
    # pi_faez (7 running / 55 pending) while mit_preemptable and mit_normal_gpu sat EMPTY. That is
    # a routing bug, not a compute shortage. Round-robin across all three; SLURM's
    # QOSMaxGRESPerUser (mit_preemptable caps gres/gpu=4, mit_normal_gpu=2) throttles each one for
    # us -- we do not have to fight it, and preemption is fine because checkpointing resumes.
    ap.add_argument("--gpu-partitions", default="pi_faez,mit_preemptable,mit_normal_gpu")
    ap.add_argument("--cpu-partition", default="mit_normal")
    ap.add_argument("--out", default=str(_here / "joblist.tsv"))
    a = ap.parse_args()

    lo, hi = (a.seeds.split("-") + [a.seeds])[:2]
    seeds = list(range(int(lo), int(hi) + 1))

    tbl = tables()
    methods = {"T1": T1, "T2": T2, "T3": T3, "T4": T4}
    rows = []
    skipped_hv = 0
    for t in a.tables:
        probs = tbl[t]
        if a.max_problems:
            probs = probs[: a.max_problems]
        for algo, prio, dev in sorted(methods[t], key=lambda x: x[1]):
            for prob, dim, nc, nobj in probs:
                # m >= 5: exact hypervolume is intractable. Skip the HV acquisitions; the scalarizing
                # ones (qnparego / constrained_qparego / the tfm_*parego pair) still run.
                if nobj >= MANY_OBJ_MIN and algo in HV_METHODS:
                    skipped_hv += len(seeds)
                    continue
                for s in seeds:
                    # The `partition` column is a legacy HINT and is IGNORED by watchdog v2, which
                    # routes by ROUTING CLASS at submit time (tfm -> pi_faez H100/H200 only, unpacked;
                    # gp -> any GPU partition, packed; cpu -> mit_normal).
                    cls = "tfm" if algo in TFM_METHODS else ("gpu" if dev == GPU else "cpu")
                    gres = "gpu:1" if dev == GPU else ""
                    rows.append((prio, dim, nc, prob, algo, s, cls, dev, a.iters, gres))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))           # priority, then dim, then #con
    with open(a.out, "w") as f:
        f.write("# problem\talgo\tseed\tclass\tdevice\titers\tgres\n")
        for _, _, _, prob, algo, s, cls, dev, it, gres in rows:
            f.write(f"{prob}\t{algo}\t{s}\t{cls}\t{dev}\t{it}\t{gres}\n")
    print(f"  skipped {skipped_hv} hypervolume rows on m>=5 problems (intractable)")

    n_gpu = sum(1 for r in rows if r[7] == GPU)
    print(f"wrote {len(rows)} jobs -> {a.out}")
    print(f"  GPU jobs: {n_gpu}   CPU jobs: {len(rows) - n_gpu}")
    for t in a.tables:
        print(f"  {t}: {len(tbl[t]) if not a.max_problems else min(a.max_problems, len(tbl[t]))} problems")


if __name__ == "__main__":
    main()
