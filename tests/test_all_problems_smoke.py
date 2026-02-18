"""
Smoke test: instantiate every problem in categorized_classes,
draw 10 random samples within bounds, and call _evaluate_implementation.
"""

import inspect

import pytest
import torch

from bocode.search_benchmarks import categorized_classes

# Categories requiring optional external packages (not installed in CI)
SKIP_CATEGORIES = {"LassoBench"}

# Problems with known upstream library bugs that we cannot fix
XFAIL_CLASSES: set[str] = set()

# Problems that require binary {0,1} inputs (not continuous)
BINARY_CLASSES = {"ZDT5"}

# Problems that require permutation inputs (city indices)
PERMUTATION_CLASSES = {"TSP_51Cities", "TSP_100Cities"}


def _collect_problems():
    """Yield (category, cls) for every class registered in categorized_classes."""
    for category, funcs in categorized_classes.items():
        if category in SKIP_CATEGORIES:
            continue
        for cls in funcs:
            yield category, cls


ALL_PROBLEMS = list(_collect_problems())
IDS = [f"{cat}/{cls.__name__}" for cat, cls in ALL_PROBLEMS]


@pytest.mark.parametrize("cat_cls", ALL_PROBLEMS, ids=IDS)
def test_smoke(cat_cls):
    category, cls = cat_cls

    if cls.__name__ in XFAIL_CLASSES:
        pytest.xfail(f"{cls.__name__}: known upstream library issue")

    # --- instantiate with defaults first, fall back to heuristic dims ---
    problem = _instantiate(cls)

    actual_dim = problem.dim
    actual_obj = problem.num_objectives

    # --- generate 10 random points in [0, 1] ---
    n_samples = 10

    if cls.__name__ in PERMUTATION_CLASSES:
        # TSP problems need permutations of city indices (1-based)
        X = torch.zeros(n_samples, actual_dim)
        for i in range(n_samples):
            X[i] = (
                torch.tensor(torch.randperm(actual_dim).tolist(), dtype=torch.float) + 1
            )
    elif cls.__name__ in BINARY_CLASSES:
        X = torch.randint(0, 2, (n_samples, actual_dim)).float()
    else:
        X = torch.rand((n_samples, actual_dim))

    # --- evaluate ---
    # Try with scaling=True first (maps [0,1] to problem bounds internally).
    # Fall back to raw call for problems that don't accept the scaling kwarg.
    result = _evaluate_with_fallback(problem, X)

    # result can be (gx, fx) or (gx, ex, fx)
    if len(result) == 2:
        gx, fx = result
    elif len(result) == 3:
        gx, _ex, fx = result
    else:
        raise AssertionError(f"Unexpected result length: {len(result)}")

    assert fx.shape[0] == n_samples, f"Expected {n_samples} rows, got {fx.shape[0]}"
    assert fx.shape[1] == actual_obj, (
        f"Expected {actual_obj} objectives, got {fx.shape[1]}"
    )

    if gx is not None and problem.num_constraints > 0:
        assert gx.shape[0] == n_samples
        assert gx.shape[1] == problem.num_constraints


def _evaluate_with_fallback(problem, X):
    """Call _evaluate_implementation, trying scaling=True first for [0,1] data."""
    sig = inspect.signature(problem._evaluate_implementation)
    if "scaling" in sig.parameters:
        return problem._evaluate_implementation(X, scaling=True)
    return problem._evaluate_implementation(X)


def _instantiate(cls):
    """Try to instantiate a problem class, handling various constructor signatures."""
    from bocode.exceptions import FunctionDefinitionAssertionError

    # First, try with no arguments (uses class defaults)
    try:
        return cls()
    except TypeError:
        pass

    # Determine dim and num_objectives from class attributes
    avail_dim = getattr(cls, "available_dimensions", None)
    num_obj = getattr(cls, "num_objectives", None)

    # pick a concrete dim — use larger defaults for scalable problems
    if avail_dim is None:
        dim = 5
    elif isinstance(avail_dim, int):
        dim = avail_dim
    elif isinstance(avail_dim, (list, set)):
        dim = sorted(avail_dim)[0]
    elif isinstance(avail_dim, tuple) and len(avail_dim) == 2:
        lo = avail_dim[0] if avail_dim[0] is not None else 3
        dim = max(lo, 3)  # at least 3 to satisfy dim > num_objectives
    else:
        dim = 5

    # pick concrete num_objectives
    if num_obj is None:
        n_obj = 2
    elif isinstance(num_obj, int):
        n_obj = num_obj
    elif isinstance(num_obj, tuple) and len(num_obj) == 2:
        n_obj = num_obj[0] if num_obj[0] is not None else 2
    else:
        n_obj = 2

    # Ensure dim > num_objectives for problems that require it (DTLZ, etc.)
    if dim <= n_obj:
        dim = n_obj + 1

    # Try (dim, num_objectives)
    try:
        return cls(dim=dim, num_objectives=n_obj)
    except (TypeError, FunctionDefinitionAssertionError, ValueError):
        pass

    # Try with just dim
    try:
        return cls(dim=dim)
    except (TypeError, FunctionDefinitionAssertionError, ValueError):
        pass

    # Last resort: no arguments
    return cls()
