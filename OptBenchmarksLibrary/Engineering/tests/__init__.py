from .test_CarSideImpact import *
from .test_EulerBernoulliBeamBending import *
from .test_GearTrain import *
from .test_RobotPush import *
from .test_Truss10D import *
from .test_Truss25D import *
from .test_TwoBarTruss import *
from .test_WaterProblem import *
from .test_WaterResources import *
from .test_MOPTA08Car import *
from .test_Mazda import *

# Or if you want to be more explicit about what's being exported:
__all__ = [
              'test_CarSideImpact',
              'test_EulerBernoulliBeamBending',
              'test_GearTrain',
              'test_RobotPush',
              'test_Truss10D',
              'test_Truss25D',
              'test_TwoBarTruss',
              'test_WaterProblem',
              'test_WaterResources',
              'test_MOPTA08Car',
              'test_Mazda',
          ]
