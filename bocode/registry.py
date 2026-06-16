"""Central registry for BoCoDe optimization problems.

The registry maps each problem name to the module that defines it and the
optional-dependency *extra* (if any) required to use it. Problem classes are
imported lazily so that ``import bocode`` works with only the core
dependencies installed; a problem that needs an extra raises an actionable
``ImportError`` when it is first accessed.

Public API:
    list_problems(...)   -> list[str] of problem names, optionally filtered
    get_problem(name)    -> the problem class (lazy import)
    get_metadata(name)   -> dict of the problem's JSON metadata
    filter_functions(...) and the get_*_objective_* helpers (metadata-driven)
"""

from __future__ import annotations

import importlib
import json
from functools import cache
from importlib import resources

__all__ = [
    "list_problems",
    "list_synthetic",
    "get_problem",
    "get_metadata",
    "list_metadata",
    "filter_functions",
    "get_single_objective_unconstrained",
    "get_single_objective_constrained",
    "get_multi_objective_unconstrained",
    "get_multi_objective_constrained",
    "PROBLEM_REGISTRY",
]


def _load_registry() -> dict[str, tuple[str, str | None]]:
    with resources.files("bocode").joinpath("_registry_data.json").open("r") as fh:
        raw = json.load(fh)
    return {name: (mod, extra) for name, (mod, extra) in raw.items()}


# name -> (module path relative to bocode, required extra or None)
PROBLEM_REGISTRY: dict[str, tuple[str, str | None]] = _load_registry()

# Human-readable install hint per extra.
_EXTRA_HINT = {
    "mujoco": "pip install 'bocode[mujoco]'",
    "control": "pip install 'bocode[control]'",
    "modact": "pip install 'bocode[modact]'",
    "hpo": "pip install 'bocode[hpo]'",
    "mazda": "pip install 'bocode[mazda]'",
    "neorl": "pip install 'bocode[neorl]'",
    "box2d": "pip install 'bocode[box2d]'",
    "truss": "pip install 'bocode[truss]'",
}


def list_problems(
    *,
    application: str | None = None,
    num_objectives: int | None = None,
    constrained: bool | None = None,
    input_type: str | None = None,
    scalable: bool | None = None,
) -> list[str]:
    """Return sorted problem names, optionally filtered by metadata fields.

    Args:
        application: keep problems whose ``application`` matches (case-insensitive).
        num_objectives: keep problems with this objective count.
        constrained: if True keep constrained problems, if False keep unconstrained.
        input_type: one of ``"continuous"``, ``"discrete"``, ``"mixed"``.
        scalable: keep problems whose ``scalable`` flag matches.
    """
    names = []
    for name in PROBLEM_REGISTRY:
        meta = get_metadata(name)
        if (
            application is not None
            and str(meta.get("application", "")).lower() != application.lower()
        ):
            continue
        if num_objectives is not None and meta.get("num_objectives") != num_objectives:
            continue
        if (
            constrained is not None
            and (meta.get("num_constraints", 0) > 0) != constrained
        ):
            continue
        if input_type is not None and meta.get("input_type") != input_type:
            continue
        if scalable is not None and bool(meta.get("scalable")) != scalable:
            continue
        names.append(name)
    return sorted(names)


def get_problem(name: str):
    """Import and return the problem class registered under ``name``.

    Falls back to the synthetic test functions (``bocode.synthetic``) so the
    algorithm CLIs can target e.g. ``Ackley`` or ``BraninCurrin``, even though
    those are not part of the real-world ``list_problems`` registry.
    """
    if name not in PROBLEM_REGISTRY:
        from bocode.opt_problems.synthetic import SYNTHETIC_PROBLEMS

        if name in SYNTHETIC_PROBLEMS:
            return SYNTHETIC_PROBLEMS[name]
        raise KeyError(
            f"Unknown problem {name!r}. Use bocode.list_problems() (real problems) "
            "or bocode.list_synthetic() (test functions) to see what is available."
        )
    module_suffix, extra = PROBLEM_REGISTRY[name]
    try:
        module = importlib.import_module(f"bocode.{module_suffix}")
    except ImportError as exc:
        if extra is not None:
            raise ImportError(
                f"Problem {name!r} requires the optional '{extra}' dependency. "
                f"Install it with: {_EXTRA_HINT.get(extra, f'pip install bocode[{extra}]')}"
            ) from exc
        raise
    return getattr(module, name)


def list_synthetic() -> list[str]:
    """Return the names of the synthetic test functions (separate from list_problems)."""
    from bocode.opt_problems.synthetic import SYNTHETIC_PROBLEMS

    return sorted(SYNTHETIC_PROBLEMS)


@cache
def get_metadata(name: str) -> dict:
    """Return the JSON metadata dict for ``name``."""
    if name not in PROBLEM_REGISTRY:
        raise KeyError(f"Unknown problem {name!r}.")
    with (
        resources.files("bocode.opt_problems_metadata")
        .joinpath(f"{name}.json")
        .open("r") as fh
    ):
        return json.load(fh)


def list_metadata() -> list[dict]:
    """Return metadata dicts for every registered problem."""
    return [get_metadata(n) for n in sorted(PROBLEM_REGISTRY)]


def filter_functions(
    *,
    num_objectives: int | None = None,
    constrained: bool | None = None,
    input_type: str | None = None,
    application: str | None = None,
) -> list[str]:
    """Alias of :func:`list_problems` kept for backward compatibility."""
    return list_problems(
        num_objectives=num_objectives,
        constrained=constrained,
        input_type=input_type,
        application=application,
    )


def get_single_objective_unconstrained() -> list[str]:
    """Single-objective, unconstrained problems."""
    return list_problems(num_objectives=1, constrained=False)


def get_single_objective_constrained() -> list[str]:
    """Single-objective, constrained problems."""
    return list_problems(num_objectives=1, constrained=True)


def get_multi_objective_unconstrained() -> list[str]:
    """Multi-objective (>=2), unconstrained problems."""
    return sorted(
        n
        for n in list_problems(constrained=False)
        if (get_metadata(n).get("num_objectives") or 0) >= 2
    )


def get_multi_objective_constrained() -> list[str]:
    """Multi-objective (>=2), constrained problems."""
    return sorted(
        n
        for n in list_problems(constrained=True)
        if (get_metadata(n).get("num_objectives") or 0) >= 2
    )
