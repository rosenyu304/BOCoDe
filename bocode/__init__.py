"""BoCoDe: a benchmark suite of real-world optimization problems for Bayesian optimization.

Problems are accessed lazily by name so that ``import bocode`` succeeds with only
the core dependencies installed::

    import bocode

    problem = bocode.get_problem("Car")()      # registry lookup
    car = bocode.Car()                          # equivalent flat access
    names = bocode.list_problems(application="Engineering")
    meta = bocode.get_metadata("Car")

Problems that need an optional dependency raise an actionable ``ImportError``
when first accessed (see the extras in ``pyproject.toml``).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bocode")
except PackageNotFoundError:  # package is not installed
    __version__ = "unknown"

from .exceptions import (  # noqa: F401  (re-exported as public API)
    DimensionException,
    FunctionDefinitionAssertionError,
    RangeException,
    TypeException,
)
from .opt_problems import (
    synthetic,  # noqa: F401  (bocode.synthetic.<Name> test functions)
)
from .registry import (  # noqa: F401  (re-exported as public API)
    PROBLEM_REGISTRY,
    filter_functions,
    get_metadata,
    get_multi_objective_constrained,
    get_multi_objective_unconstrained,
    get_problem,
    get_single_objective_constrained,
    get_single_objective_unconstrained,
    list_metadata,
    list_problems,
    list_synthetic,
)

_API = {
    "__version__",
    "DimensionException",
    "FunctionDefinitionAssertionError",
    "RangeException",
    "TypeException",
    "PROBLEM_REGISTRY",
    "filter_functions",
    "get_metadata",
    "get_problem",
    "list_metadata",
    "list_problems",
    "list_synthetic",
    "synthetic",
    "get_multi_objective_constrained",
    "get_multi_objective_unconstrained",
    "get_single_objective_constrained",
    "get_single_objective_unconstrained",
}

__all__ = sorted(_API | set(PROBLEM_REGISTRY))


def __getattr__(name: str):
    """Lazily resolve problem classes by name via the registry (PEP 562)."""
    if name in PROBLEM_REGISTRY:
        return get_problem(name)
    raise AttributeError(f"module 'bocode' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(PROBLEM_REGISTRY))
