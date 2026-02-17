"""
Smoke test: instantiate every problem in categorized_classes,
draw 10 random samples within bounds, and call _evaluate_implementation.
"""

import pytest
import torch

from bocode.search_benchmarks import categorized_classes


def _collect_problems():
    """Yield (category, cls) for every class registered in categorized_classes."""
    for category, funcs in categorized_classes.items():
        for cls in funcs:
            yield category, cls


ALL_PROBLEMS = list(_collect_problems())
IDS = [f"{cat}/{cls.__name__}" for cat, cls in ALL_PROBLEMS]


@pytest.mark.parametrize("cat_cls", ALL_PROBLEMS, ids=IDS)
def test_smoke(cat_cls):
    category, cls = cat_cls

    # --- determine dimensions & objectives to use for instantiation ---
    avail_dim = getattr(cls, "available_dimensions", None)
    num_obj = getattr(cls, "num_objectives", None)

    # pick a concrete dim
    if avail_dim is None:
        dim = 2
    elif isinstance(avail_dim, int):
        dim = avail_dim
    elif isinstance(avail_dim, (list, set)):
        dim = sorted(avail_dim)[0]
    elif isinstance(avail_dim, tuple) and len(avail_dim) == 2:
        dim = avail_dim[0] if avail_dim[0] is not None else 2
    else:
        dim = 2

    # pick concrete num_objectives
    if num_obj is None:
        n_obj = 2
    elif isinstance(num_obj, int):
        n_obj = num_obj
    elif isinstance(num_obj, tuple) and len(num_obj) == 2:
        n_obj = num_obj[0] if num_obj[0] is not None else 2
    else:
        n_obj = 2

    # --- instantiate ---
    try:
        # Try with (dim, num_objectives) first for multi-obj scalable problems
        problem = cls(dim=dim, num_objectives=n_obj)
    except TypeError:
        try:
            problem = cls(dim=dim)
        except TypeError:
            problem = cls()

    actual_dim = problem.dim
    actual_obj = problem.num_objectives

    # --- generate 10 random points scaled to bounds ---
    n_samples = 10
    X = torch.rand((n_samples, actual_dim))
    X = problem.scale(X)

    # --- evaluate ---
    result = problem._evaluate_implementation(X)

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
