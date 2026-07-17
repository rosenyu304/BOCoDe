"""SPEA2 for multi-objective problems -- a thin adapter around **pymoo**.

This file does not reimplement SPEA2: it drives pymoo's official
``pymoo.algorithms.moo.spea2.SPEA2`` on a BoCoDe multi-objective problem, purely to
showcase that BoCoDe's problems interface with pymoo unchanged. The strength-based
fitness assignment, density estimation and archive truncation all live in pymoo; this
adapter only supplies the search-space bridge (BoCoDe's unit-cube/maximization/
``c <= 0`` conventions -> pymoo's minimization) and reports a running hypervolume
comparable with the BO baselines. See ``algorithms/_pymoo_utils.py`` for the bridge
and the budget/HV conventions.

CPU only. Run::

    python -m algorithms.multi_obj.spea2 --problem BraninCurrin --init 10 --iters 50

Sources:
E. Zitzler, M. Laumanns, and L. Thiele. SPEA2: Improving the Strength Pareto Evolutionary Algorithm. TIK-Report 103, ETH Zurich, 2001.
J. Blank and K. Deb. pymoo: Multi-Objective Optimization in Python. IEEE Access, 2020. https://pymoo.org
"""

from __future__ import annotations

from pymoo.algorithms.moo.spea2 import SPEA2

from .._bo_utils import Result
from .._pymoo_utils import pymoo_main, run_pymoo


def optimize_problem(
    problem, n_init: int | None = None, iters: int = 50, seed: int = 0
) -> Result:
    """SPEA2 (pymoo) over the unit cube; running hypervolume in ``per_iteration_value``."""
    return run_pymoo(
        problem,
        "spea2",
        lambda pop_size: SPEA2(pop_size=pop_size),
        n_init,
        iters,
        seed,
        constrained=False,
    )


def main() -> None:
    pymoo_main(__doc__, "spea2", optimize_problem)


if __name__ == "__main__":
    main()
