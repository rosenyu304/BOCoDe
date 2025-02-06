from .test_Ackley import *
from .test_Bukin import *
from .test_DixonPrice import *
from .test_Goldstein import *
from .test_Goldstein_Discrete import *
from .test_Griewank import *
from .test_Levy import *
from .test_Michalewicz import *
from .test_Powell import *
from .test_Rastrigin import *
from .test_Rosenbrock import *
from .test_StyblinskiTang import *

# Or if you want to be more explicit about what's being exported:
__all__ = ['test_Ackley', 
           'test_Bukin', 
           'test_DixonPrice', 
           'test_Goldstein',
           'test_Goldstein_Discrete',
           'test_Griewank',
           'test_Levy',
           'test_Michalewicz',
           'test_Powell',
           'test_Rastrigin',
           'test_Rosenbrock',
           'test_StyblinskiTang',
          ]