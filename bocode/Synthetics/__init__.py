from .Ackley import Ackley
from .Bukin import Bukin
from .DixonPrice import DixonPrice
from .Goldstein import Goldstein, Goldstein_Discrete
from .Griewank import Griewank
from .Levy import Levy
from .Michalewicz import Michalewicz
from .More_Synthetics import (
    Beale,
    ConstrainedGramacy,
    ConstrainedHartmann,
    ConstrainedHartmannSmooth,
    Cosine8,
    DropWave,
    EggHolder,
    Hartmann3D,
    Hartmann6D,
    HolderTable,
    PressureVessel,
    Shekel,
    Shekelm5,
    Shekelm7,
    Shekelm10,
    SixHumpCamel,
    SpeedReducer,
    TensionCompressionString,
    ThreeHumpCamel,
    WeldedBeamSO,
)
from .Powell import Powell
from .Rastrigin import Rastrigin
from .Rosenbrock import Rosenbrock
from .StyblinskiTang import StyblinskiTang
from .SVM import SVM

# Or if you want to be more explicit about what's being exported:
__all__ = [
    "Ackley",
    "Bukin",
    "DixonPrice",
    "Goldstein",
    "Goldstein_Discrete",
    "Griewank",
    "Levy",
    "Michalewicz",
    "Powell",
    "Rastrigin",
    "Rosenbrock",
    "StyblinskiTang",
    "Beale",
    "Cosine8",
    "DropWave",
    "EggHolder",
    "Hartmann3D",
    "Hartmann6D",
    "HolderTable",
    "Shekelm5",
    "Shekelm7",
    "Shekelm10",
    "Shekel",
    "SixHumpCamel",
    "ThreeHumpCamel",
    "ConstrainedGramacy",
    "ConstrainedHartmann",
    "ConstrainedHartmannSmooth",
    "PressureVessel",
    "WeldedBeamSO",
    "TensionCompressionString",
    "SpeedReducer",
    "SVM",
]
