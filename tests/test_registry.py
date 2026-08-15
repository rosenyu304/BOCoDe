"""Registry and metadata integrity tests (no problem instantiation)."""

import importlib

import pytest

import bocode
from bocode.registry import PROBLEM_REGISTRY


def test_registry_nonempty():
    assert len(PROBLEM_REGISTRY) > 100


def test_every_problem_has_metadata():
    """Each registered problem must have a matching metadata JSON (drift guard)."""
    for name in PROBLEM_REGISTRY:
        meta = bocode.get_metadata(name)
        assert meta["name"] == name
        for field in ("num_objectives", "num_constraints", "application", "source"):
            assert field in meta, f"{name} metadata missing {field!r}"


def test_metadata_modules_importable_paths():
    """The module recorded for each problem matches its registry entry."""
    for name, (module_suffix, _extra) in PROBLEM_REGISTRY.items():
        meta = bocode.get_metadata(name)
        assert meta["module"] == f"bocode.{module_suffix}"


def test_filter_helpers_partition_objectives():
    so_u = set(bocode.get_single_objective_unconstrained())
    so_c = set(bocode.get_single_objective_constrained())
    mo_u = set(bocode.get_multi_objective_unconstrained())
    mo_c = set(bocode.get_multi_objective_constrained())
    # the four groups are disjoint
    groups = [so_u, so_c, mo_u, mo_c]
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert not (a & b)


def test_unknown_problem_raises():
    with pytest.raises(KeyError):
        bocode.get_problem("NoSuchProblem")


def test_missing_extra_gives_actionable_error(monkeypatch):
    """A problem whose extra is absent raises ImportError naming the extra."""
    # pick any problem that declares an extra
    name = next(n for n, (_m, e) in PROBLEM_REGISTRY.items() if e is not None)
    module_suffix, extra = PROBLEM_REGISTRY[name]

    real_import = importlib.import_module

    def fake_import(modname, *a, **k):
        if modname == f"bocode.{module_suffix}":
            raise ImportError("simulated missing dependency")
        return real_import(modname, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError) as exc:
        bocode.get_problem(name)
    assert extra in str(exc.value)
