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
from .transforms import (  # noqa: F401  (re-exported as public API)
    ContinuousRelaxation,
    PenalizedProblem,
    ScalarizedProblem,
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
    "PenalizedProblem",
    "ScalarizedProblem",
    "ContinuousRelaxation",
    "get_multi_objective_constrained",
    "get_multi_objective_unconstrained",
    "get_single_objective_constrained",
    "get_single_objective_unconstrained",
}

__all__ = sorted(_API | set(PROBLEM_REGISTRY))


def __getattr__(name: str):
    """Lazily resolve attributes (PEP 562).

    ``bocode.synthetic`` and problem classes are imported ON DEMAND so that
    ``import bocode`` stays cheap -- it must NOT pull in the synthetic test
    functions (and their botorch dependency) for a job that only runs, say,
    ``Borehole`` or ``random_search``. On a shared cluster filesystem the eager
    import of those modules dominated per-job wall time.
    """
    if name == "synthetic":
        import bocode.opt_problems.synthetic as _synthetic

        return _synthetic
    if name in PROBLEM_REGISTRY:
        return get_problem(name)
    raise AttributeError(f"module 'bocode' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(PROBLEM_REGISTRY))
