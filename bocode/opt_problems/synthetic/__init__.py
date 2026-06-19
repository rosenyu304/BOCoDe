"""Synthetic test functions for examples and algorithm testing.

These are classic analytical benchmarks (not real-world problems), kept separate
from the main suite. Access them as ``bocode.synthetic.Ackley()`` or via
``bocode.get_problem("Ackley")``; they are intentionally excluded from
``bocode.list_problems()`` and ``CATEGORIZATION.md``.

Single-objective, unconstrained: Ackley, Rosenbrock, Levy, Powell, DixonPrice,
Griewank, Rastrigin, StyblinskiTang (10-D); Hartmann (6-D); Branin, EggHolder,
HolderTable, SixHumpCamel (2-D).
Single-objective, constrained: KeaneBumpFunction (10-D, 2), ConstrainedHartmann (6-D, 1).
Multi-objective: BraninCurrin (2 obj), ZDT1/ZDT2/ZDT3 (2 obj), DTLZ1-DTLZ5 (3 obj).
"""

from .Ackley import Ackley
from .Branin import Branin
from .BraninCurrin import BraninCurrin
from .ConstrainedHartmann import ConstrainedHartmann
from .DixonPrice import DixonPrice
from .DTLZ1 import DTLZ1
from .DTLZ2 import DTLZ2
from .DTLZ3 import DTLZ3
from .DTLZ4 import DTLZ4
from .DTLZ5 import DTLZ5
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
from .ZDT1 import ZDT1
from .ZDT2 import ZDT2
from .ZDT3 import ZDT3

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
    "DTLZ2": DTLZ2,
    "DTLZ3": DTLZ3,
    "DTLZ4": DTLZ4,
    "DTLZ5": DTLZ5,
    "ZDT1": ZDT1,
    "ZDT2": ZDT2,
    "ZDT3": ZDT3,
}

__all__ = [*SYNTHETIC_PROBLEMS.keys(), "SYNTHETIC_PROBLEMS"]
