"""Batch experiment runner — every algorithm against every compatible problem.

Runs Bayesian-optimization baselines over a grid of (algorithm x problem x seed),
saving the full per-iteration trace of each run to a ``.npz`` and writing a summary
row (best value, and the gap to the known optimum where one is recorded). Designed
to scale from a laptop smoke test to a full 100+-problem sweep on a cluster.

Algorithm/problem compatibility is decided from each problem's metadata
(``num_objectives``, ``num_constraints``): single/multi-objective and
constrained/unconstrained problems are matched to the algorithms in the
corresponding ``algorithms/`` sub-package.

Examples
--------
Smoke test (a few problems, all compatible algorithms, one seed)::

    python examples/run_experiments.py --problems Branin Sellar PressureVessel --seeds 0

Full sweep over every registered problem and 5 seeds::

    python examples/run_experiments.py --problems all --seeds 0 10 20 30 40 --iters 50

One shard of a SLURM array job (see examples/README.md)::

    python examples/run_experiments.py --problems all --seeds 0 10 20 \
        --task-id $SLURM_ARRAY_TASK_ID --num-tasks $SLURM_ARRAY_TASK_COUNT

The TFM algorithms (GIT-BO, PFN-CEI) need a separate TabPFN environment and are
excluded unless ``--include-tfm`` is passed (see docs/tfm_setup.md).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bocode  # noqa: E402

# Algorithm sub-package per (objectives, constrained) category. TFM modules
# (git_bo, pfn_cei) are listed separately and only run with --include-tfm.
_ALGOS = {
    ("single", False): [
        "random_search",
        "single_task_gp",
        "standard_gp",
        "vanilla_highdim_bo",
        "turbo",
    ],
    ("single", True): ["random_search", "constrained_ei", "scbo"],
    ("multi", False): ["random_search", "qnehvi", "qnparego"],
    ("multi", True): ["constrained_qnehvi", "constrained_qparego"],
}
_PKG = {
    ("single", False): "algorithms.single_obj",
    ("single", True): "algorithms.single_obj_constrained",
    ("multi", False): "algorithms.multi_obj",
    ("multi", True): "algorithms.multi_obj_constrained",
}
_TFM = {  # module name -> sub-package (need the TabPFN env)
    "git_bo": "algorithms.single_obj",
    "pfn_cei": "algorithms.single_obj_constrained",
}


def _category(name: str):
    """Return ('single'|'multi', constrained: bool) from the class attributes.

    Uses class attributes so it works for both registered problems and the
    synthetic test functions (which are not in the metadata registry).
    """
    cls = bocode.get_problem(name)
    objs = "multi" if (getattr(cls, "num_objectives", 1) or 1) >= 2 else "single"
    return objs, (getattr(cls, "num_constraints", 0) or 0) > 0


def _known_optimum(name: str, problem) -> float | None:
    """Known optimum (min sense) from metadata or the instance, else None."""
    opt = None
    if name in bocode.registry.PROBLEM_REGISTRY:
        opt = bocode.get_metadata(name).get("f_opt")
    if opt is None:
        opt = getattr(problem, "optimum", None)
        if isinstance(opt, (list, tuple)) and len(opt) == 1:
            opt = opt[0]
    return opt if isinstance(opt, (int, float)) else None


def _algos_for(name: str, include_tfm: bool):
    """List of (module_path, algo_name) compatible with this problem."""
    cat = _category(name)
    out = [(f"{_PKG[cat]}.{a}", a) for a in _ALGOS[cat]]
    if include_tfm:
        if cat == ("single", False):
            out.append(("algorithms.single_obj.git_bo", "git_bo"))
        if cat == ("single", True):
            out.append(("algorithms.single_obj_constrained.pfn_cei", "pfn_cei"))
    return out


def _run_one(modpath, algo, problem_name, seed, n_init, iters, outdir):
    """Run a single (algorithm, problem, seed); return a summary dict or None."""
    mod = importlib.import_module(modpath)
    problem = bocode.get_problem(problem_name)()
    fn = mod.optimize_problem
    # random_search takes (problem, iters, seed); BO methods add n_init.
    kwargs = {"seed": seed, "iters": iters}
    if "n_init" in inspect.signature(fn).parameters:
        kwargs["n_init"] = n_init
    res = fn(problem, **kwargs)

    path = Path(outdir) / f"{algo}_{problem_name}_seed{seed}.npz"
    res.to_npz(str(path))

    # optimality gap for single-objective problems with a known optimum
    gap = None
    f_opt = _known_optimum(problem_name, problem)
    if f_opt is not None and getattr(problem, "num_objectives", 1) == 1:
        gap = (-res.best) - f_opt  # algorithms maximize -obj; -best is the min found
    return {
        "algo": algo,
        "problem": problem_name,
        "seed": seed,
        "best": res.best,
        "gap": gap,
        "npz": path.name,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--problems", nargs="+", required=True, help="problem names, or 'all'"
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--outdir", default=str(Path(__file__).parent / "results"))
    p.add_argument(
        "--include-tfm",
        action="store_true",
        help="also run GIT-BO/PFN-CEI (needs the TabPFN env)",
    )
    p.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="this shard's index (for cluster array jobs)",
    )
    p.add_argument("--num-tasks", type=int, default=1, help="total number of shards")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    problems = bocode.list_problems() if args.problems == ["all"] else args.problems

    # build the full (algorithm x problem x seed) job list, then take this shard
    jobs = []
    for name in problems:
        for modpath, algo in _algos_for(name, args.include_tfm):
            for seed in args.seeds:
                jobs.append((modpath, algo, name, seed))
    jobs = jobs[args.task_id :: args.num_tasks]
    print(f"shard {args.task_id}/{args.num_tasks}: {len(jobs)} runs -> {args.outdir}")

    summary, failures = [], []
    for modpath, algo, name, seed in jobs:
        tag = f"{algo} / {name} / seed{seed}"
        try:
            row = _run_one(
                modpath, algo, name, seed, args.n_init, args.iters, args.outdir
            )
            gap = f"  gap={row['gap']:.4g}" if row["gap"] is not None else ""
            print(f"  OK  {tag}: best={row['best']:.6g}{gap}")
            summary.append(row)
        except Exception as exc:  # noqa: BLE001 - log and keep going
            print(f"  SKIP {tag}: {type(exc).__name__}: {exc}")
            failures.append((tag, traceback.format_exc()))

    # write a summary CSV for this shard
    csv = Path(args.outdir) / f"summary_task{args.task_id}.csv"
    with csv.open("w") as fh:
        fh.write("algorithm,problem,seed,best,gap,npz\n")
        for r in summary:
            fh.write(
                f"{r['algo']},{r['problem']},{r['seed']},{r['best']},{r['gap']},{r['npz']}\n"
            )
    print(f"\n{len(summary)} ok, {len(failures)} skipped. Summary: {csv}")


if __name__ == "__main__":
    main()
