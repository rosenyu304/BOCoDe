"""Generate per-problem metadata JSON files from the registered problem classes.

Run from the repo root in an environment with all extras installed::

    python tools/generate_metadata.py

For each registered problem this introspects the class attributes
(``num_objectives``, ``num_constraints``, ``available_dimensions``,
``input_type``) and, for problems that are cheap to construct, the instance
``dim`` and ``bounds``. Hand-curated fields (``application``, ``convex``,
``np_hard``, ``source``) come from ``tools/metadata_overrides.json`` and the
class/module docstring. One JSON per problem is written to
``bocode/opt_problems_metadata/``.

Problems that are expensive or side-effectful to instantiate (run a simulator,
open a window, download data, train a model) are listed in ``_NO_INSTANTIATE``;
their ``dim``/``bounds`` come from the overrides file instead.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bocode.registry import PROBLEM_REGISTRY, get_problem  # noqa: E402

META_DIR = ROOT / "bocode" / "opt_problems_metadata"
OVERRIDES_PATH = Path(__file__).resolve().parent / "metadata_overrides.json"

# Problems that should not be instantiated during metadata generation.
_NO_INSTANTIATE = {
    "SVM",  # loads a 78 MB dataset + trains an SVR
    "Mazda",
    "Mazda_SCA",  # spawn a subprocess binary
    "MOPTA08Car",  # runs a native binary
    "RobotPush",  # opens a pygame/Box2D world
}

# Application area by registry module prefix (fallback when not in overrides).
_APPLICATION_BY_PREFIX = (
    ("opt_problems.control.mujoco", "Control"),
    ("opt_problems.control", "Control"),
    ("opt_problems.hpo", "Hyperparameter Optimization"),
    ("opt_problems.materials", "Materials"),
    ("opt_problems.combinatorial", "Combinatorial"),
    ("opt_problems.modact", "Engineering"),
    ("opt_problems.cec2020_rw", "Engineering"),
    ("opt_problems.reproblems", "Engineering"),
    ("opt_problems.engineering", "Engineering"),
)


def _application(name: str, module: str, overrides: dict) -> str:
    if name in overrides and "application" in overrides[name]:
        return overrides[name]["application"]
    for prefix, app in _APPLICATION_BY_PREFIX:
        if module.startswith(prefix):
            return app
    return "Unknown"


# Benchmark suite / family by module, so grouped problems (CS1, RE21, …) are
# identifiable. Order matters: more specific module paths first.
_SUITE_BY_MODULE = (
    ("opt_problems.modact", "MODAct (actuator design)"),
    ("opt_problems.cec2020_rw", "CEC2020 RW-Constrained"),
    ("opt_problems.reproblems.RE_problems", "RE (Tanabe-Ishibuchi)"),
    ("opt_problems.reproblems.CRE_problems", "CRE (Tanabe-Ishibuchi)"),
    ("opt_problems.materials", "PV-Lab materials"),
    ("opt_problems.control.mujoco", "MuJoCo control"),
    ("opt_problems.control", "Classic control"),
    ("opt_problems.combinatorial", "TSP / NEORL"),
    ("opt_problems.hpo", "HPO (LassoBench / GP-HDBO)"),
)


def _suite(name: str, module: str, overrides: dict) -> str:
    if name in overrides and "suite" in overrides[name]:
        return overrides[name]["suite"]
    for prefix, suite in _SUITE_BY_MODULE:
        if module.startswith(prefix):
            return suite
    return "Engineering (standalone)"


def _source(cls) -> str:
    """Extract a one-line source citation from the class or module docstring.

    Prefer an explicit ``Sources:`` block (in either the class or module
    docstring); fall back to the first non-empty docstring otherwise.
    """
    docs = [cls.__doc__, sys.modules[cls.__module__].__doc__]
    for doc in docs:
        if not doc:
            continue
        m = re.search(r"[Ss]ource[s]?:\s*(.+)", doc, re.S)
        if m:
            return " ".join(m.group(1).split())[:500]
    for doc in docs:
        if doc and doc.strip():
            return " ".join(doc.split())[:500]
    return ""


def _scalable(available_dimensions) -> bool:
    # available_dimensions given as a (min, max) tuple/list signals scalability.
    return isinstance(available_dimensions, (tuple, list, set))


def _jsonify(value):
    if isinstance(value, (set,)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def build_one(name: str, overrides: dict) -> dict:
    module, extra = PROBLEM_REGISTRY[name]
    cls = get_problem(name)

    meta = {
        "name": name,
        "module": f"bocode.{module}",
        "extra": extra,
        "suite": _suite(name, module, overrides),
        "num_objectives": getattr(cls, "num_objectives", None),
        "num_constraints": getattr(cls, "num_constraints", None),
        "available_dimensions": _jsonify(getattr(cls, "available_dimensions", None)),
        "input_type": getattr(getattr(cls, "input_type", None), "value", None)
        or getattr(cls, "input_type", None),
        "scalable": _scalable(getattr(cls, "available_dimensions", None)),
        "dim": None,
        "bounds": None,
        "num_integer_variables": None,
        "application": _application(name, module, overrides),
        "convex": "unknown",
        "np_hard": "unknown",
        "f_opt": None,
        "source": _source(cls),
    }

    if name not in _NO_INSTANTIATE:
        try:
            inst = cls()
            meta["dim"] = getattr(inst, "dim", None)
            bounds = getattr(inst, "bounds", None)
            if bounds is not None:
                meta["bounds"] = [list(b) for b in bounds]
            meta["num_objectives"] = getattr(
                inst, "num_objectives", meta["num_objectives"]
            )
            meta["num_constraints"] = getattr(
                inst, "num_constraints", meta["num_constraints"]
            )
            # per-dimension variable types (all "continuous" unless the class
            # declares otherwise via variable_types); count the non-continuous ones.
            # known optimum (natural minimization sense), if the class exposes one
            opt = getattr(inst, "optimum", None)
            if isinstance(opt, (list, tuple)) and len(opt) == 1:
                opt = opt[0]
            if isinstance(opt, (int, float)):
                meta["f_opt"] = round(float(opt), 6)
            vtypes = inst.resolved_variable_types()
            meta["variable_types"] = vtypes
            meta["num_integer_variables"] = sum(1 for t in vtypes if t == "integer")
            meta["num_categorical_variables"] = sum(
                1 for t in vtypes if not isinstance(t, str)
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            meta["_instantiation_error"] = f"{type(exc).__name__}: {exc}"

    # apply hand-curated overrides last (authoritative)
    for key, val in overrides.get(name, {}).items():
        meta[key] = val

    return meta


def main() -> None:
    overrides = (
        json.loads(OVERRIDES_PATH.read_text()) if OVERRIDES_PATH.exists() else {}
    )
    META_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for name in sorted(PROBLEM_REGISTRY):
        try:
            meta = build_one(name, overrides)
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        (META_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        ok += 1
    print(f"wrote {ok} metadata files to {META_DIR}")
    if failed:
        print(f"{len(failed)} problems failed metadata generation:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
