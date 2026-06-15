"""Smoke test: instantiate every registered problem and evaluate random points.

Driven by the metadata registry (``bocode.list_problems``). Problems whose
optional dependency is not installed are skipped, so this passes on a slim core
install and exercises everything when the extras are present.
"""

import importlib
import inspect

import pytest
import torch

import bocode
from bocode.registry import PROBLEM_REGISTRY

# Problems requiring permutation inputs (city indices), handled specially below.
PERMUTATION_CLASSES = {"TSP_51Cities", "TSP_100Cities"}

# Discrete dataset-lookup problems: sample from their candidate set, not [0, 1].
DATASET_CLASSES = {"AgNP", "CrossedBarrel", "P3HT", "Perovskite", "AutoAM"}

# Problems with known upstream issues (see docs/AUDIT_findings_2026_06.md).
XFAIL_CLASSES: set[str] = set()

# Problems whose evaluation downloads a dataset (network) and/or is heavy; skipped
# in the smoke test. The LassoBench problems fetch from OpenML / fetch_rcv1.
NETWORK_CLASSES = {
    "LassoDiabetes", "LassoBreastCancer", "LassoDNA", "LassoLeukemia", "LassoRCV1",
}


def _extra_available(extra: str | None) -> bool:
    """Return True if the optional dependency backing ``extra`` is importable."""
    if extra is None:
        return True
    probe = {
        "mujoco": "gymnasium",
        "control": "gymnasium",
        "modact": "modact",
        "hpo": "sklearn",
        "mazda": "openpyxl",
        "neorl": "onnxruntime",
        "box2d": "Box2D",
        "truss": "slientruss3d",
    }.get(extra, extra)
    return importlib.util.find_spec(probe) is not None


ALL_PROBLEMS = sorted(PROBLEM_REGISTRY)


@pytest.mark.parametrize("name", ALL_PROBLEMS)
def test_smoke(name):
    extra = PROBLEM_REGISTRY[name][1]
    if not _extra_available(extra):
        pytest.skip(f"{name}: optional dependency '{extra}' not installed")
    if name in NETWORK_CLASSES:
        pytest.skip(f"{name}: evaluation downloads a dataset (network); skipped in smoke")
    if name in XFAIL_CLASSES:
        pytest.xfail(f"{name}: known upstream issue (see AUDIT_findings)")

    cls = bocode.get_problem(name)
    problem = cls()

    n_samples = 10
    dim = problem.dim

    if name in PERMUTATION_CLASSES:
        X = torch.stack(
            [torch.randperm(dim).float() + 1 for _ in range(n_samples)]
        )
    elif name in DATASET_CLASSES:
        idx = torch.randint(0, problem.candidates.shape[0], (n_samples,))
        X = problem.candidates[idx]
    else:
        X = torch.rand((n_samples, dim))

    objectives = _evaluate_objectives(problem, X)

    assert objectives.shape[0] == n_samples
    assert objectives.shape[1] == problem.num_objectives


def _evaluate_objectives(problem, X):
    """Return the objective tensor, scaling [0, 1] inputs internally when supported.

    ``_evaluate_implementation`` returns ``(constraints, objectives)`` or
    ``(eq, ineq, objectives)``; the objective tensor is always the last element.
    """
    sig = inspect.signature(problem._evaluate_implementation)
    if "scaling" in sig.parameters:
        result = problem._evaluate_implementation(X, scaling=True)
    else:
        result = problem._evaluate_implementation(X)
    return result[-1]
