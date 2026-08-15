"""Constrained SPEA2 for constrained multi-objective problems -- pymoo adapter.

This file does not reimplement SPEA2: it drives pymoo's official
``pymoo.algorithms.moo.spea2.SPEA2`` on a *constrained* BoCoDe multi-objective
problem, purely to showcase that BoCoDe interfaces with pymoo unchanged. It is the
same SPEA2 as ``algorithms/multi_obj/spea2.py``; the only difference is that the
wrapped pymoo ``Problem`` exposes the problem's inequality constraints
(``n_ieq_constr > 0``), so pymoo's constraint handling drives the search toward the
feasible region. BoCoDe's constraints are feasible when ``c <= 0`` and so are pymoo's,
so they pass straight through. The reported ``per_iteration_value`` is the running
**feasible** hypervolume (only points satisfying every constraint count), matching the
constrained qNEHVI baseline. See ``algorithms/_pymoo_utils.py``.

CPU only. Run::

    python -m algorithms.multi_obj_constrained.spea2 --problem WeldedBeam --init 12 --iters 50

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
    """Constrained SPEA2 (pymoo); running feasible hypervolume in ``per_iteration_value``."""
    return run_pymoo(
        problem,
        "con_spea2",
        lambda pop_size: SPEA2(pop_size=pop_size),
        n_init,
        iters,
        seed,
        constrained=True,
    )


def main() -> None:
    pymoo_main(__doc__, "con_spea2", optimize_problem)


if __name__ == "__main__":
    main()
