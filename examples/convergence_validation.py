"""Convergence-validation harness — do the BO baselines actually converge?

Runs every compatible single-objective algorithm against a set of problems whose
optimum is known, over several seeds, and reports — per (problem, algorithm) — the
mean +/- std best objective found, the gap to the known optimum, and the improvement
over the random-search baseline. This certifies which algorithm/problem combinations
converge (rather than just "runs without error"), and surfaces the ones that don't.

The smoke tests in ``tests/test_algorithms.py`` only check that runs are monotone on
tiny budgets; this harness checks *convergence* on a real budget against ground truth.

Writes a markdown report and a CSV. Example::

    python examples/convergence_validation.py --seeds 0 1 2 3 4 --iters 40

Only single-objective problems are validated (a scalar gap/regret needs a scalar
optimum); pass ``--problems`` to override the default known-optimum set.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bocode  # noqa: E402

# Default validation set: problems with a known/verifiable single-objective optimum.
_DEFAULT = ["Branin", "Sellar", "Allison", "MiniAeroWing", "PEARL", "PressureVessel"]
_UNCONSTRAINED = [
    "random_search",
    "single_task_gp",
    "standard_gp",
    "vanilla_highdim_bo",
    "turbo",
]
_CONSTRAINED = ["random_search", "constrained_ei", "scbo"]


def _known_optimum(name, problem):
    """Known optimum (min sense) from metadata or the instance, else None."""
    opt = None
    if name in bocode.registry.PROBLEM_REGISTRY:
        opt = bocode.get_metadata(name).get("f_opt")
    if opt is None:
        opt = getattr(problem, "optimum", None)
        if isinstance(opt, (list, tuple)) and len(opt) == 1:
            opt = opt[0]
    return opt if isinstance(opt, (int, float)) else None


def _algos_for(problem):
    constrained = (getattr(problem, "num_constraints", 0) or 0) > 0
    pkg = (
        "algorithms.single_obj_constrained" if constrained else "algorithms.single_obj"
    )
    names = _CONSTRAINED if constrained else _UNCONSTRAINED
    return pkg, names


def _run(modpath, problem, seed, n_init, iters):
    fn = importlib.import_module(modpath).optimize_problem
    kwargs = {"seed": seed, "iters": iters}
    if "n_init" in inspect.signature(fn).parameters:
        kwargs["n_init"] = n_init
    res = fn(problem, **kwargs)
    # algorithms maximize -obj; -best is the best (min) objective found. Cast to a
    # plain float (res.best may be numpy/torch; +inf when no feasible point was found).
    return float(-res.best)


def validate(problems, seeds, n_init, iters):
    """Return {problem: {"f_opt": float|None, "algos": {algo: [best per seed]}}}."""
    out = {}
    for name in problems:
        problem = bocode.get_problem(name)()
        f_opt = _known_optimum(name, problem)
        pkg, algos = _algos_for(problem)
        rows = {}
        for algo in algos:
            vals = []
            for seed in seeds:
                try:
                    vals.append(
                        _run(
                            f"{pkg}.{algo}",
                            bocode.get_problem(name)(),
                            seed,
                            n_init,
                            iters,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  SKIP {algo}/{name}/seed{seed}: {type(exc).__name__}: {exc}"
                    )
            if vals:
                rows[algo] = vals
        out[name] = {"f_opt": f_opt, "algos": rows}
        print(f"  done {name} (f_opt={f_opt})")
    return out


def _mean(vals):
    """Mean over finite values (inf when none are finite, e.g. no feasible point)."""
    finite = [v for v in vals if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("inf")


def _std(vals):
    finite = [v for v in vals if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    m = sum(finite) / len(finite)
    return (sum((v - m) ** 2 for v in finite) / len(finite)) ** 0.5


def _verdict(algo, gap, rand_mean, mean_best, f_opt):
    if not math.isfinite(mean_best):
        return "no feasible found"
    if algo == "random_search":
        return "baseline"
    if f_opt is not None and gap is not None and abs(gap) <= 0.05 * (abs(f_opt) + 1e-9):
        return "converges"
    if math.isfinite(rand_mean) and mean_best < rand_mean - 1e-9:
        return "beats random"
    return "no better than random"


def render(results, seeds, n_init, iters):
    lines = [
        "# Convergence validation report",
        "",
        f"Budget: {n_init} initial + {iters} iterations, seeds {list(seeds)}. "
        "`best` is the best objective found (minimization sense), mean +/- std across "
        "seeds; `gap` = best - f_opt; `vs random` = random_mean - algo_mean (positive "
        "is better than random).",
        "",
    ]
    csv = ["problem,f_opt,algorithm,best_mean,best_std,gap,vs_random,verdict"]
    for name, d in results.items():
        f_opt = d["f_opt"]
        rand = d["algos"].get("random_search")
        rand_mean = _mean(rand) if rand else float("nan")
        lines += [
            f"## {name}  (f_opt = {f_opt})",
            "",
            "| algorithm | best (mean ± std) | gap | vs random | verdict |",
            "|---|---|---|---|---|",
        ]
        for algo, vals in d["algos"].items():
            mean = _mean(vals)
            std = _std(vals)
            feasible = math.isfinite(mean)
            n_infeas = sum(1 for v in vals if not math.isfinite(v))
            gap = (mean - f_opt) if (f_opt is not None and feasible) else None
            vsr = (
                (rand_mean - mean) if (feasible and math.isfinite(rand_mean)) else None
            )
            verdict = _verdict(algo, gap, rand_mean, mean, f_opt)
            if feasible:
                btxt = f"{mean:.5g} ± {std:.3g}"
                if n_infeas:
                    btxt += f" ({n_infeas}/{len(vals)} infeasible)"
            else:
                btxt = "no feasible point"
            gtxt = f"{gap:.4g}" if gap is not None else "—"
            vtxt = "—" if (algo == "random_search" or vsr is None) else f"{vsr:+.4g}"
            lines.append(f"| {algo} | {btxt} | {gtxt} | {vtxt} | {verdict} |")
            csv.append(f"{name},{f_opt},{algo},{mean},{std},{gap},{vsr},{verdict}")
        lines.append("")
    return "\n".join(lines) + "\n", "\n".join(csv) + "\n"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--problems", nargs="+", default=_DEFAULT)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument(
        "--report", default=str(Path(__file__).parent / "convergence_report.md")
    )
    args = p.parse_args()

    print(
        f"validating {len(args.problems)} problems x algorithms x {len(args.seeds)} seeds"
    )
    results = validate(args.problems, args.seeds, args.n_init, args.iters)
    md, csv = render(results, args.seeds, args.n_init, args.iters)
    Path(args.report).write_text(md)
    Path(args.report).with_suffix(".csv").write_text(csv)
    print(f"\nwrote {args.report} and {Path(args.report).with_suffix('.csv').name}")
    print("\n" + md)


if __name__ == "__main__":
    main()
