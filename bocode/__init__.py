from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bocode")
except PackageNotFoundError:
    # package is not installed
    pass

import bocode.BBOB as BBOB
import bocode.BoTorch as BoTorch
import bocode.CEC as CEC
import bocode.DTLZ as DTLZ
import bocode.Engineering as Engineering
import bocode.LassoBench as LassoBench
import bocode.MODAct as MODAct
import bocode.NEORL as NEORL
import bocode.Synthetics as Synthetics
import bocode.WFG as WFG
import bocode.ZDT as ZDT

from .base import DataType

from .exceptions import (
    DimensionException,
    FunctionDefinitionAssertionError,
    RangeException,
    TypeException,
)
from .search_benchmarks import filter_functions

__all__ = [
    "__version__",
    "DataType",
    "BBOB",
    "BoTorch",
    "CEC",
    "DTLZ",
    "Engineering",
    "LassoBench",
    "MODAct",
    "NEORL",
    "Synthetics",
    "WFG",
    "ZDT",
    "filter_functions",
    "DimensionException",
    "FunctionDefinitionAssertionError",
    "RangeException",
    "TypeException",
]
