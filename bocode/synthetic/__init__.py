"""Synthetic test functions for examples and algorithm testing.

These are classic analytical benchmarks (not real-world problems), kept separate
from the main suite. Access them as ``bocode.synthetic.Ackley()`` or via
``bocode.get_problem("Ackley")``; they are intentionally excluded from
``bocode.list_problems()`` and ``CATEGORIZATION.md``.

Single-objective: Ackley, Rosenbrock, Levy, Powell (all 10-D), Branin (2-D).
Multi-objective: BraninCurrin (2 obj), DTLZ1 (3 obj).
"""

from .Ackley import Ackley
from .Branin import Branin
from .BraninCurrin import BraninCurrin
from .DTLZ1 import DTLZ1
from .Levy import Levy
from .Powell import Powell
from .Rosenbrock import Rosenbrock

#: name -> class, used by bocode.get_problem as a fallback registry.
SYNTHETIC_PROBLEMS = {
    "Ackley": Ackley,
    "Rosenbrock": Rosenbrock,
    "Levy": Levy,
    "Powell": Powell,
    "Branin": Branin,
    "BraninCurrin": BraninCurrin,
    "DTLZ1": DTLZ1,
}

__all__ = [*SYNTHETIC_PROBLEMS.keys(), "SYNTHETIC_PROBLEMS"]
