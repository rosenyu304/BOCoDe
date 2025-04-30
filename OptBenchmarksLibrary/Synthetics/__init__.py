from .Ackley import *
from .Bukin import *
from .DixonPrice import *
from .Goldstein import *
from .Griewank import *
from .Levy import *
from .Michalewicz import *
from .Powell import *
from .Rastrigin import *
from .Rosenbrock import *
from .StyblinskiTang import *
from .More_Synthetics import *
from .SVM import *

# Or if you want to be more explicit about what's being exported:
__all__ = ['Ackley', 
           'Bukin', 
           'DixonPrice', 
           'Goldstein',
           'Goldstein_Discrete',
           'Griewank',
           'Levy',
           'Michalewicz',
           'Powell',
           'Rastrigin',
           'Rosenbrock',
           'StyblinskiTang',
           'Beale',
           'Cosine8',
           'DropWave',
           'EggHolder',
           'Hartmann3D',
           'Hartmann6D',
           'HolderTable',
           'Shekelm5',
           'Shekelm7',
           'Shekelm10',
           'Shekel',
           'SixHumpCamel',
           'ThreeHumpCamel',
           'ConstrainedGramacy',
           'ConstrainedHartmann',
           'ConstrainedHartmannSmooth',
           'PressureVessel',
           'WeldedBeamSO',
           'TensionCompressionString',
           'SpeedReducer'
          ]
