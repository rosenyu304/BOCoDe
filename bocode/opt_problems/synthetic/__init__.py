"""Synthetic test functions for examples and algorithm testing.

These are classic analytical benchmarks (not real-world problems), kept separate
from the main suite. Access them as ``bocode.synthetic.Ackley()`` or via
``bocode.get_problem("Ackley")``; they are intentionally excluded from
``bocode.list_problems()`` and ``CATEGORIZATION.md``.

Single-objective, unconstrained: Ackley, Rosenbrock, Levy, Powell, DixonPrice,
Griewank, Rastrigin, StyblinskiTang (10-D); Hartmann (6-D); Branin, EggHolder,
HolderTable, SixHumpCamel (2-D).
Single-objective, constrained: KeaneBumpFunction (10-D, 2), ConstrainedHartmann (6-D, 1).
Multi-objective: BraninCurrin (2 obj), DTLZ1 (3 obj).
"""

from .Ackley import Ackley
from .Branin import Branin
from .BraninCurrin import BraninCurrin
from .ConstrainedHartmann import ConstrainedHartmann
from .DixonPrice import DixonPrice
from .DTLZ1 import DTLZ1
from .EggHolder import EggHolder
from .Griewank import Griewank
from .Hartmann import Hartmann
from .HolderTable import HolderTable
from .KeaneBumpFunction import KeaneBumpFunction
from .Levy import Levy
from .Powell import Powell
from .Rastrigin import Rastrigin
from .Rosenbrock import Rosenbrock
from .SixHumpCamel import SixHumpCamel
from .StyblinskiTang import StyblinskiTang

#: name -> class, used by bocode.get_problem as a fallback registry.
SYNTHETIC_PROBLEMS = {
    "Ackley": Ackley,
    "Rosenbrock": Rosenbrock,
    "Levy": Levy,
    "Powell": Powell,
    "DixonPrice": DixonPrice,
    "Griewank": Griewank,
    "Rastrigin": Rastrigin,
    "StyblinskiTang": StyblinskiTang,
    "Hartmann": Hartmann,
    "Branin": Branin,
    "EggHolder": EggHolder,
    "HolderTable": HolderTable,
    "SixHumpCamel": SixHumpCamel,
    "KeaneBumpFunction": KeaneBumpFunction,
    "ConstrainedHartmann": ConstrainedHartmann,
    "BraninCurrin": BraninCurrin,
    "DTLZ1": DTLZ1,
}

__all__ = [*SYNTHETIC_PROBLEMS.keys(), "SYNTHETIC_PROBLEMS"]
