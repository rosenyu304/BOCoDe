"""Regenerate ``bocode/_registry_data.json`` from the problem source files.

Scans ``bocode/opt_problems`` for problem classes (top-level ``class`` defs that
are not helpers/bases) and records, per problem name, the defining module and
the optional-dependency extra it needs. Run from the repo root::

    python tools/generate_registry.py

Keep this in sync with ``bocode/registry.py`` (which loads the JSON at import
time) and ``tools/generate_metadata.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "bocode" / "opt_problems"
OUT = ROOT / "bocode" / "_registry_data.json"

# Classes that are bases/helpers, not standalone benchmark problems.
SKIP = {
    "MultiObjBotorchProblem", "BaseModactProblem", "MaterialsDatasetProblem",
    "ConstantOffsetFn", "NormalizedInputFn", "Trajectory", "PointBSpline",
    "RoverDomain", "AABoxes", "NegGeom", "UnionGeom", "ConstObstacleCost",
    "ConstCost", "AdditiveCosts", "GMCost", "PushReward", "guiWorld",
    "b2WorldInterface", "end_effector",
}


def extra_for(modpath: str, cls: str):
    if "/control/mujoco/" in modpath:
        return "mujoco"
    if "/control/" in modpath:
        return "control"
    if "/modact/" in modpath:
        return "modact"
    if cls.startswith("Lasso"):
        return "lasso"
    if cls == "SVM":
        return "hpo"
    if cls in ("Mazda", "Mazda_SCA"):
        return "mazda"
    if cls == "QPowerModel":
        return "neorl"
    if cls == "RobotPush":
        return "box2d"
    if cls.startswith("Truss") and cls not in ("TwoBarTruss", "ThreeTruss"):
        return "truss"
    return None


def main() -> None:
    reg = {}
    for f in sorted(PROBLEMS.rglob("*.py")):
        if (
            "_vendor" in str(f)
            or f.name == "__init__.py"
            or f.name.startswith("_")
            or f.name == "helperFuncs.py"
        ):
            continue
        classes = re.findall(r"^class\s+([A-Za-z_0-9]+)", f.read_text(), re.M)
        modpath = str(f.relative_to(ROOT / "bocode")).replace("/", ".")[:-3]
        for c in classes:
            if c in SKIP:
                continue
            reg[c] = [modpath, extra_for(str(f), c)]
    OUT.write_text(json.dumps({k: reg[k] for k in sorted(reg)}, indent=2) + "\n")
    print(f"wrote {len(reg)} entries to {OUT}")


if __name__ == "__main__":
    main()
