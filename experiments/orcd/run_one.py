"""Run ONE (problem, algorithm, seed) tuple. Resumable, provenance-stamped.

The unit of work for the whole campaign. Writes exactly one file:
    <results>/<problem>/<algorithm>/seed<N>.npz

Every result records the commit SHA that produced it, so a run from stale code can be
detected instead of silently polluting the tables. (An ad-hoc rsync once left four machines
running sign-inverted code for hours; nothing caught it. This is the fix.)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

ALGOS = {
    # single-objective, unconstrained
    "random_search": "algorithms.single_obj.random_search",
    "single_task_gp": "algorithms.single_obj.single_task_gp",
    "standard_gp": "algorithms.single_obj.standard_gp",
    "vanilla_highdim_bo": "algorithms.single_obj.vanilla_highdim_bo",
    "turbo": "algorithms.single_obj.turbo",
    "baxus": "algorithms.single_obj.baxus",
    "git_bo": "algorithms.single_obj.git_bo",
    "git_bo_marzouk": "algorithms.single_obj.git_bo",
    "tfm_turbo": "algorithms.single_obj.tfm_turbo",
    "tabicl_turbo": "algorithms.single_obj.tabicl_turbo",
    "tabicl_baxus": "algorithms.single_obj.tabicl_baxus",
    "rf_turbo": "algorithms.single_obj.rf_turbo",
    "gp_ucb": "algorithms.single_obj.gp_ucb",
    "rf_ucb": "algorithms.single_obj.rf_ucb",
    # multi-objective
    "mo_random_search": "algorithms.multi_obj.random_search",
    "qnehvi": "algorithms.multi_obj.qnehvi",
    "qnparego": "algorithms.multi_obj.qnparego",
    "tfm_qnehvi": "algorithms.multi_obj.tfm_qnehvi",
    "tfm_qnparego": "algorithms.multi_obj.tfm_qnparego",
    "dgemo": "algorithms.multi_obj.dgemo",
    # single-objective, constrained
    "con_random_search": "algorithms.single_obj_constrained.random_search",
    "scbo": "algorithms.single_obj_constrained.scbo",
    "pfn_cei": "algorithms.single_obj_constrained.pfn_cei",
    "tfm_scbo": "algorithms.single_obj_constrained.tfm_scbo",
    "tabicl_scbo": "algorithms.single_obj_constrained.tabicl_scbo",
    "rf_scbo": "algorithms.single_obj_constrained.rf_scbo",
    "rf_cei": "algorithms.single_obj_constrained.rf_cei",
    "penalty": "algorithms.single_obj_constrained.penalty",
    # multi-objective, constrained
    "mocon_random_search": "algorithms.multi_obj_constrained.random_search",
    "constrained_qnehvi": "algorithms.multi_obj_constrained.constrained_qnehvi",
    "constrained_qparego": "algorithms.multi_obj_constrained.constrained_qparego",
    "tfm_cqnehvi": "algorithms.multi_obj_constrained.tfm_cqnehvi",
    "tfm_cqnparego": "algorithms.multi_obj_constrained.tfm_cqnparego",
    "mocon_penalty": "algorithms.multi_obj_constrained.penalty",
    # mixed-variable
    "mv_random_search": "algorithms.single_obj_mixed_variable.random_search",
    "casmopolitan": "algorithms.single_obj_mixed_variable.casmopolitan",
    "tabicl_ucb": "algorithms.single_obj_mixed_variable.tabicl_ucb",
    "tabicl_ei": "algorithms.single_obj_mixed_variable.tabicl_ei",
    "hebo": "algorithms.single_obj_mixed_variable.hebo",
    "bodi": "algorithms.single_obj_mixed_variable.bodi",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--iters", type=int, default=1000
    )  # BO iterations, EXCLUDING n_init
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--results", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sha", default="unknown")
    a = ap.parse_args()

    out = Path(a.results) / a.problem / a.algo / f"seed{a.seed}.npz"
    if out.exists():
        print(f"SKIP {out} already exists")
        return

    import bocode
    from algorithms._bo_utils import default_n_init

    mod = importlib.import_module(ALGOS[a.algo])
    problem = bocode.get_problem(a.problem)()

    kw = {}
    sig = mod.optimize_problem.__code__.co_varnames
    if "device" in sig:
        kw["device"] = a.device
    if "checkpoint" in sig and a.checkpoint:
        kw["checkpoint"] = a.checkpoint

    # random_search has no separate initial design: give it the SAME TOTAL budget as the
    # BO methods (n_init + iters), or the baseline is handed fewer evaluations than the
    # methods it is being compared against.
    iters = a.iters
    if "random_search" in a.algo:
        iters = default_n_init(problem.dim) + a.iters

    if a.algo == "git_bo_marzouk" and "rank" in sig:
        kw["rank"] = "marzouk"
    res = mod.optimize_problem(problem, iters=iters, seed=a.seed, **kw)

    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_npz(str(out))
    # stamp provenance into the .npz so a stale-code run is detectable later
    d = dict(np.load(out, allow_pickle=True))
    d["commit_sha"] = a.sha
    d["host"] = os.uname().nodename
    d["device"] = a.device
    np.savez(out, **d)
    print(f"DONE {out} best={res.best:.6g} sha={a.sha[:8]}")


if __name__ == "__main__":
    sys.exit(main())
