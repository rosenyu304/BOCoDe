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
    "algorithms.single_obj.single_task_gp",
    "algorithms.single_obj.vanilla_highdim_bo",
    "algorithms.single_obj.turbo",
    "algorithms.single_obj.standard_gp",
]

SINGLE_OBJ_CONSTRAINED = [
    "algorithms.single_obj_constrained.random_search",
    "algorithms.single_obj_constrained.constrained_ei",
    "algorithms.single_obj_constrained.scbo",
]

MULTI_OBJ = [
    "algorithms.multi_obj.random_search",
    "algorithms.multi_obj.qnehvi",
    "algorithms.multi_obj.qnparego",
]


def _mod(name):
    return importlib.import_module(name)


def _monotone(history):
    return all(b <= a + 1e-9 for b, a in zip(history, history[1:], strict=False))


def _check_trace(res):
    """Every per-iteration list has the same length and the npz dict has all keys."""
    n = len(res.per_iteration_value)
    assert n >= 3
    assert len(res.wall_time) == n
    assert len(res.mean) == n
    assert len(res.variance) == n
    assert len(res.per_iteration_acquisition_function_value) == n
    assert res.iterations == list(range(n))
    assert res.wall_time[0] == 0.0  # trial clock starts at iteration 0
    d = res.to_dict()
    for key in (
        "seed",
        "acquisition_function",
        "best",
        "iterations",
        "per_iteration_value",
        "wall_time",
        "mean",
        "variance",
        "per_iteration_acquisition_function_value",
    ):
        assert key in d


@pytest.mark.parametrize("modname", SINGLE_OBJ)
def test_problem_optimization_runs(modname):
    mod = _mod(modname)
    problem = bocode.CompressionSpring()
    if modname.endswith("random_search"):
        res = mod.optimize_problem(problem, iters=12, seed=0)
    else:
        res = mod.optimize_problem(problem, n_init=6, iters=4, seed=0)
    _check_trace(res)
    # best-so-far is monotonically non-decreasing
    assert _monotone(res.per_iteration_value)


@pytest.mark.parametrize("modname", SINGLE_OBJ)
def test_dataset_optimization_runs(modname):
    mod = _mod(modname)
    problem = bocode.Perovskite()
    if modname.endswith("random_search"):
        res = mod.optimize_dataset(problem, iters=15, seed=0)
    else:
        res = mod.optimize_dataset(problem, n_init=6, iters=6, seed=0)
    _check_trace(res)
    assert _monotone(res.per_iteration_value)
    # final best cannot exceed the dataset's true maximum
    assert res.best <= problem.values.max().item() + 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("modname", SINGLE_OBJ_CONSTRAINED)
def test_constrained_runs(modname):
    mod = _mod(modname)
    problem = bocode.PressureVessel()
    if modname.endswith("random_search"):
        res = mod.optimize_problem(problem, iters=30, seed=0)
    else:
        res = mod.optimize_problem(problem, n_init=8, iters=4, seed=0)
    _check_trace(res)
    assert _monotone(res.per_iteration_value)  # best feasible objective never decreases


@pytest.mark.slow
@pytest.mark.parametrize("modname", MULTI_OBJ)
def test_multi_obj_runs(modname):
    mod = _mod(modname)
    problem = bocode.Penicillin()  # 3 objectives
    if modname.endswith("random_search"):
        res = mod.optimize_problem(problem, iters=20, seed=0)
    else:
        res = mod.optimize_problem(problem, n_init=6, iters=2, seed=0)
    _check_trace(res)
    assert _monotone(res.per_iteration_value)  # hypervolume never decreases


@pytest.mark.slow
@pytest.mark.parametrize(
    "modname",
    [
        "algorithms.multi_obj_constrained.constrained_qnehvi",
        "algorithms.multi_obj_constrained.constrained_qparego",
    ],
)
def test_constrained_multi_obj_runs(modname):
    mod = _mod(modname)
    res = mod.optimize_problem(bocode.WeldedBeam(), n_init=8, iters=2, seed=0)
    _check_trace(res)
    assert _monotone(res.per_iteration_value)
