"""NSGA-II for multi-objective problems -- a thin adapter around **pymoo**.

This file does not reimplement NSGA-II: it drives pymoo's official
``pymoo.algorithms.moo.nsga2.NSGA2`` on a BoCoDe multi-objective problem, purely to
showcase that BoCoDe's problems interface with pymoo unchanged. The elitist
non-dominated sorting, crowding-distance selection, SBX crossover and polynomial
mutation all live in pymoo; this adapter only supplies the search-space bridge
(BoCoDe's unit-cube/maximization/``c <= 0`` conventions -> pymoo's minimization) and
reports a running hypervolume comparable with the BO baselines. See
``algorithms/_pymoo_utils.py`` for the bridge and the budget/HV conventions.

CPU only. Run::

    python -m algorithms.multi_obj.nsga2 --problem BraninCurrin --init 10 --iters 50

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
    """NSGA-II (pymoo) over the unit cube; running hypervolume in ``per_iteration_value``."""
    return run_pymoo(
        problem,
        "nsga2",
        lambda pop_size: NSGA2(pop_size=pop_size),
        n_init,
        iters,
        seed,
        constrained=False,
    )


def main() -> None:
    pymoo_main(__doc__, "nsga2", optimize_problem)


if __name__ == "__main__":
    main()
