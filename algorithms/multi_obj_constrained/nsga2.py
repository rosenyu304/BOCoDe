"""Constrained NSGA-II for constrained multi-objective problems -- pymoo adapter.

This file does not reimplement NSGA-II: it drives pymoo's official
``pymoo.algorithms.moo.nsga2.NSGA2`` on a *constrained* BoCoDe multi-objective
problem, purely to showcase that BoCoDe interfaces with pymoo unchanged. It is the
same NSGA-II as ``algorithms/multi_obj/nsga2.py``; the only difference is that the
wrapped pymoo ``Problem`` exposes the problem's inequality constraints
(``n_ieq_constr > 0``), so pymoo's constraint-domination handling drives the search
toward the feasible region. BoCoDe's constraints are feasible when ``c <= 0`` and so
are pymoo's, so they pass straight through. The reported ``per_iteration_value`` is
the running **feasible** hypervolume (only points satisfying every constraint count),
matching the constrained qNEHVI baseline. See ``algorithms/_pymoo_utils.py``.

CPU only. Run::

    python -m algorithms.multi_obj_constrained.nsga2 --problem WeldedBeam --init 12 --iters 50

Sources:
K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE TEVC 6(2), 2002.
J. Blank and K. Deb. pymoo: Multi-Objective Optimization in Python. IEEE Access, 2020. https://pymoo.org
"""

from __future__ import annotations

from pymoo.algorithms.moo.nsga2 import NSGA2

from .._bo_utils import Result
from .._pymoo_utils import pymoo_main, run_pymoo


def optimize_problem(
    problem, n_init: int | None = None, iters: int = 50, seed: int = 0
) -> Result:
    """Constrained NSGA-II (pymoo); running feasible hypervolume in ``per_iteration_value``."""
    return run_pymoo(
        problem,
        "con_nsga2",
        lambda pop_size: NSGA2(pop_size=pop_size),
        n_init,
        iters,
        seed,
        constrained=True,
    )


def main() -> None:
    pymoo_main(__doc__, "con_nsga2", optimize_problem)


if __name__ == "__main__":
    main()
