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
from .BNH import BNH
from .Branin import Branin, BraninShifted
from .BraninCurrin import BraninCurrin
from .C2DTLZ2 import C2DTLZ2
from .ConstrainedBraninCurrin import ConstrainedBraninCurrin
from .ConstrainedGramacy import ConstrainedGramacy
from .ConstrainedHartmann import ConstrainedHartmann
from .ConstrainedHartmannSmooth import ConstrainedHartmannSmooth
from .CONSTR import CONSTR
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
from .Michalewicz import Michalewicz
from .MW7 import MW7
from .OSY import OSY
from .Powell import Powell, PowellShifted
from .Rastrigin import Rastrigin, RastriginShifted
from .Rosenbrock import Rosenbrock, RosenbrockShifted
from .SixHumpCamel import SixHumpCamel, SixHumpCamelShifted
from .SRN import SRN
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
    "Michalewicz": Michalewicz,
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
    "ConstrainedGramacy": ConstrainedGramacy,
    "ConstrainedHartmannSmooth": ConstrainedHartmannSmooth,
    "C2DTLZ2": C2DTLZ2,
    "ConstrainedBraninCurrin": ConstrainedBraninCurrin,
    "BNH": BNH,
    "SRN": SRN,
    "OSY": OSY,
    "CONSTR": CONSTR,
    "MW7": MW7,
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

# GIT-BO high-dimensional scalable variants (<Func>_100D .. <Func>_500D), added as
# unregistered test functions so the campaign can target a fixed high dim by name.
from .gitbo_scalable import GITBO_SCALABLE_PROBLEMS  # noqa: E402

SYNTHETIC_PROBLEMS.update(GITBO_SCALABLE_PROBLEMS)

__all__ = [*SYNTHETIC_PROBLEMS.keys(), "SYNTHETIC_PROBLEMS"]


# --- Mixed-variable synthetics (merged from the former synthetic_mixed/ package,
# 2026-07-17). These ARE registered (appear in bocode.list_problems()), unlike the
# plain test functions above, so they are deliberately kept OUT of SYNTHETIC_PROBLEMS
# (the unregistered test-function fallback used by list_synthetic()). ---
from .Ackley5Mixed import Ackley5Mixed  # noqa: E402,F401
from .Ackley53 import Ackley53  # noqa: E402,F401
from .AckleyCat import AckleyCat  # noqa: E402,F401
from .BraninCategorical import BraninCategorical  # noqa: E402,F401
from .BraninLVGP import BraninLVGP  # noqa: E402,F401
from .CoCaBOFunc2C import CoCaBOFunc2C  # noqa: E402,F401
from .CoCaBOFunc3C import CoCaBOFunc3C  # noqa: E402,F401
from .GoldsteinLVGP import GoldsteinLVGP  # noqa: E402,F401
from .GoldsteinMixed import GoldsteinMixed  # noqa: E402,F401
from .HartmannCat import HartmannCat  # noqa: E402,F401
from .MixedAckley import MixedAckley  # noqa: E402,F401
from .RastriginCat import RastriginCat  # noqa: E402,F401
from .Rosenbrock5Mixed import Rosenbrock5Mixed  # noqa: E402,F401
from .SchwefelCat import SchwefelCat  # noqa: E402,F401
from .ShekelMixed import ShekelMixed  # noqa: E402,F401
from .StyblinskiTangCat import StyblinskiTangCat  # noqa: E402,F401
from .StyblinskiTangMixed import StyblinskiTangMixed  # noqa: E402,F401
from .WeldedBeamCategorical import WeldedBeamCategorical  # noqa: E402,F401

MIXED_SYNTHETIC_PROBLEMS = [
    "Ackley53",
    "Ackley5Mixed",
    "AckleyCat",
    "BraninCategorical",
    "BraninLVGP",
    "CoCaBOFunc2C",
    "CoCaBOFunc3C",
    "GoldsteinLVGP",
    "GoldsteinMixed",
    "HartmannCat",
    "MixedAckley",
    "RastriginCat",
    "Rosenbrock5Mixed",
    "SchwefelCat",
    "ShekelMixed",
    "StyblinskiTangCat",
    "StyblinskiTangMixed",
    "WeldedBeamCategorical",
]
__all__ += [*MIXED_SYNTHETIC_PROBLEMS, "MIXED_SYNTHETIC_PROBLEMS"]
