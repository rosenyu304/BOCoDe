from .LassoDNA import *
from .LassoSyntHigh import *
from .LassoSyntMedium import *
from .LassoSyntHard import *

# Or if you want to be more explicit about what's being exported:
__all__ = ['LassoDNA', 
           'LassoDiabetes',
           'LassoBreastCancer',
           'LassoRCV1',
           'LassoLeu',
           'LassoSyntSimple',
           'LassoSyntMedium',
           'LassoSyntHigh', 
           'LassoSyntHard'
          ]
