"""Mixed continuous + categorical/integer synthetic benchmark problems.

A family of analytical mixed-variable benchmarks drawn from the mixed-variable BO
literature: the LVGP quantitative/qualitative problems (Zhang, Apley & Chen 2020),
the CoCaBO functions (Ru et al., ICML 2020), the discretised Styblinski-Tang /
Goldstein-Price / Shekel problems, and several additive- or offset-categorical
constructions.

Every problem follows the BoCoDe conventions: ``evaluate()`` **maximizes** (so the
objective is the negation of the minimization form quoted in each docstring), and
constraints are feasible when ``<= 0``. The ``optimum`` attribute records the
reference ``f*`` in the original *minimization* sense.

Each problem declares its per-dimension ``variable_types``. Where a categorical
level has a physical value (e.g. BraninLVGP's ``x2`` in ``{0, 5, 10, 15}``) the
level list holds those values and they enter the objective directly; where the
categorical is genuinely nominal (a material, a sub-function selector) the level
list holds the level *indices* ``{0, ..., K-1}``.

Unlike ``bocode.opt_problems.synthetic`` (plain continuous test functions, kept out
of the registry), these problems are registered and appear in
``bocode.list_problems()``.
"""

from .Ackley5Mixed import Ackley5Mixed
from .Ackley53 import Ackley53
from .AckleyCat import AckleyCat
from .BraninCategorical import BraninCategorical
from .BraninLVGP import BraninLVGP
from .CoCaBOFunc2C import CoCaBOFunc2C
from .CoCaBOFunc3C import CoCaBOFunc3C
from .GoldsteinLVGP import GoldsteinLVGP
from .GoldsteinMixed import GoldsteinMixed
from .HartmannCat import HartmannCat
from .MixedAckley import MixedAckley
from .RastriginCat import RastriginCat
from .Rosenbrock5Mixed import Rosenbrock5Mixed
from .SchwefelCat import SchwefelCat
from .ShekelMixed import ShekelMixed
from .StyblinskiTangCat import StyblinskiTangCat
from .StyblinskiTangMixed import StyblinskiTangMixed
from .WeldedBeamCategorical import WeldedBeamCategorical

__all__ = [
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
