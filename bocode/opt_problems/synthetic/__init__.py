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

Shifted variants: ``<Name>Shifted`` (AckleyShifted, GriewankShifted, …) keep the same
landscape and optimal value but translate the optimum away from the box center, which
six of the centered functions hand out for free to any center-biased method. See
:class:`~._wrapper.ShiftedSingleObjSyntheticProblem`. EggHolder and HolderTable have no
shifted variant: their optima lie on the box boundary and their formulas are unbounded
below outside it, so no translation preserves the reference optimum.
"""

from .Ackley import Ackley, AckleyShifted
from .Branin import Branin, BraninShifted
from .BraninCurrin import BraninCurrin
from .ConstrainedHartmann import ConstrainedHartmann
from .DixonPrice import DixonPrice, DixonPriceShifted
from .DTLZ1 import DTLZ1
from .DTLZ2 import DTLZ2
from .DTLZ3 import DTLZ3
from .DTLZ4 import DTLZ4
from .DTLZ5 import DTLZ5
from .EggHolder import EggHolder
from .Griewank import Griewank, GriewankShifted
from .Hartmann import Hartmann, HartmannShifted
from .HolderTable import HolderTable
from .KeaneBumpFunction import KeaneBumpFunction
from .Levy import Levy, LevyShifted
from .Powell import Powell, PowellShifted
from .Rastrigin import Rastrigin, RastriginShifted
from .Rosenbrock import Rosenbrock, RosenbrockShifted
from .SixHumpCamel import SixHumpCamel, SixHumpCamelShifted
from .StyblinskiTang import StyblinskiTang, StyblinskiTangShifted
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
    "AckleyShifted": AckleyShifted,
    "RosenbrockShifted": RosenbrockShifted,
    "LevyShifted": LevyShifted,
    "PowellShifted": PowellShifted,
    "DixonPriceShifted": DixonPriceShifted,
    "GriewankShifted": GriewankShifted,
    "RastriginShifted": RastriginShifted,
    "StyblinskiTangShifted": StyblinskiTangShifted,
    "HartmannShifted": HartmannShifted,
    "BraninShifted": BraninShifted,
    "SixHumpCamelShifted": SixHumpCamelShifted,
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
