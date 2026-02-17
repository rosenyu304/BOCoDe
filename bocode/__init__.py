from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bocode")
except PackageNotFoundError:
    # package is not installed
    pass

import bocode.BBOB as BBOB
import bocode.BoTorch as BoTorch
import bocode.CEC as CEC
import bocode.Engineering as Engineering
import bocode.LassoBench as LassoBench
import bocode.NEORL as NEORL
import bocode.Synthetics as Synthetics

from .base import DataType

from .exceptions import (
    DimensionException,
    FunctionDefinitionAssertionError,
    RangeException,
    TypeException,
)
from .search_benchmarks import (
    filter_functions,
    get_multi_objective_constrained,
    get_multi_objective_unconstrained,
    get_single_objective_constrained,
    get_single_objective_unconstrained,
)

__all__ = [
    "__version__",
    "DataType",
    "BBOB",
    "BoTorch",
    "CEC",
    "Engineering",
    "LassoBench",
    "NEORL",
    "Synthetics",
    "filter_functions",
    "get_single_objective_unconstrained",
    "get_single_objective_constrained",
    "get_multi_objective_unconstrained",
    "get_multi_objective_constrained",
    "DimensionException",
    "FunctionDefinitionAssertionError",
    "RangeException",
    "TypeException",
]
