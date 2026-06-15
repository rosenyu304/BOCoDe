"""Smoke tests for the algorithm scripts: they run and improve over the initial design.

Kept short (few iterations, small problems) so the suite stays fast. These import
from the top-level ``algorithms`` package, which is research code shipped beside
the installable ``bocode`` package.
"""

import importlib

import pytest

import bocode

SINGLE_OBJ = [
    "algorithms.single_obj.random_search",
    "algorithms.single_obj.vanilla_bo",
    "algorithms.single_obj.turbo",
    "algorithms.single_obj.standard_gp",
]


def _mod(name):
    return importlib.import_module(name)


@pytest.mark.parametrize("modname", SINGLE_OBJ)
def test_problem_optimization_runs(modname):
    mod = _mod(modname)
    problem = bocode.CompressionSpring()
    if modname.endswith("random_search"):
        res = mod.optimize_problem(problem, iters=12, seed=0)
    else:
        res = mod.optimize_problem(problem, n_init=6, iters=4, seed=0)
    assert len(res.best_history) >= 10
    # best-so-far is monotonically non-decreasing
    assert all(b <= a for b, a in zip(res.best_history, res.best_history[1:], strict=True))


@pytest.mark.parametrize("modname", SINGLE_OBJ)
def test_dataset_optimization_runs(modname):
    mod = _mod(modname)
    problem = bocode.Perovskite()
    if modname.endswith("random_search"):
        res = mod.optimize_dataset(problem, iters=15, seed=0)
    else:
        res = mod.optimize_dataset(problem, n_init=6, iters=6, seed=0)
    assert len(res.best_history) >= 10
    assert all(b <= a for b, a in zip(res.best_history, res.best_history[1:], strict=True))
    # final best cannot exceed the dataset's true maximum
    assert res.best <= problem.values.max().item() + 1e-6
