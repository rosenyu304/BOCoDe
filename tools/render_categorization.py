"""Render CATEGORIZATION.md from the per-problem metadata JSONs.

The big problem table is generated so it never drifts from the metadata. Run::

    python tools/render_categorization.py

This rewrites the table between the ``<!-- TABLE:START -->`` / ``<!-- TABLE:END -->``
markers in CATEGORIZATION.md, leaving the surrounding prose (including the
NP-hardness methodology) untouched.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from bocode import list_problems  # noqa: E402
from bocode.registry import PROBLEM_REGISTRY, get_metadata  # noqa: E402

DOC = ROOT / "CATEGORIZATION.md"
START = "<!-- TABLE:START -->"
END = "<!-- TABLE:END -->"
CAT_START = "<!-- CATEGORIES:START -->"
CAT_END = "<!-- CATEGORIES:END -->"

# (heading, display string, list_problems kwargs, multi_objective?). For the
# multi-objective categories we drop num_objectives from the kwargs and post-filter
# to >= 2 objectives, so the "multi" groups capture any problem with >=2 objectives.
_CATEGORIES = [
    (
        "Single-objective, unconstrained, continuous",
        "list_problems(num_objectives=1, constrained=False, input_type='continuous')",
        {"num_objectives": 1, "constrained": False, "input_type": "continuous"},
        False,
    ),
    (
        "Single-objective, unconstrained, mixed-variable",
        "list_problems(num_objectives=1, constrained=False, input_type='mixed')",
        {"num_objectives": 1, "constrained": False, "input_type": "mixed"},
        False,
    ),
    (
        "Single-objective, constrained, continuous",
        "list_problems(num_objectives=1, constrained=True, input_type='continuous')",
        {"num_objectives": 1, "constrained": True, "input_type": "continuous"},
        False,
    ),
    (
        "Single-objective, constrained, mixed-variable",
        "list_problems(num_objectives=1, constrained=True, input_type='mixed')",
        {"num_objectives": 1, "constrained": True, "input_type": "mixed"},
        False,
    ),
    (
        "Multi-objective, unconstrained, continuous",
        "list_problems(constrained=False, input_type='continuous')  # >=2 objectives",
        {"constrained": False, "input_type": "continuous"},
        True,
    ),
    (
        "Multi-objective, constrained, continuous",
        "list_problems(constrained=True, input_type='continuous')  # >=2 objectives",
        {"constrained": True, "input_type": "continuous"},
        True,
    ),
    (
        "Multi-objective, mixed-variable (any constraints)",
        "list_problems(input_type='mixed')  # >=2 objectives",
        {"input_type": "mixed"},
        True,
    ),
]


def build_categories() -> str:
    blocks = []
    for heading, display, kwargs, multi in _CATEGORIES:
        names = list_problems(**kwargs)
        if multi:
            names = [
                n for n in names if (get_metadata(n).get("num_objectives") or 0) >= 2
            ]
        listing = ", ".join(f"`{n}`" for n in names) if names else "_(none)_"
        blocks.append(f"### {heading} ({len(names)})\n\n`{display}`\n\n{listing}\n")
    return "\n".join(blocks)


def _fmt(value) -> str:
    if value is None:
        return "?"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        return f"{value.get('min', '?')}–{value.get('max', '?')}"
    return str(value)


def _dim(meta: dict) -> str:
    if meta.get("dim") is not None:
        return str(meta["dim"])
    ad = meta.get("available_dimensions")
    if isinstance(ad, list) and len(ad) == 2:
        lo, hi = ad
        return f"{lo if lo is not None else '?'}+" if hi is None else f"{lo}–{hi}"
    return _fmt(ad)


def build_table() -> str:
    rows = []
    for name in sorted(PROBLEM_REGISTRY):
        m = get_metadata(name)
        rows.append(
            (
                name,
                _fmt(m.get("application")),
                _fmt(m.get("suite")),
                _dim(m),
                _fmt(m.get("num_objectives")),
                _fmt(m.get("num_constraints")),
                _fmt(m.get("f_opt")),
                _fmt(m.get("convex")),
                _fmt(m.get("np_hard")),
            )
        )
    # sort by application, then suite, then name
    rows.sort(key=lambda r: (r[1], r[2], r[0]))

    header = (
        "| # | Problem | Application | Suite | Dim | #Obj | #Constr | f_opt | "
        "Convex | NP-hard |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "\n".join(
        "| " + " | ".join((str(i),) + r) + " |" for i, r in enumerate(rows, start=1)
    )
    return header + body + "\n"


def _replace(text: str, start: str, end: str, body: str) -> str:
    pre, _, rest = text.partition(start)
    _, _, post = rest.partition(end)
    return f"{pre}{start}\n\n{body}\n{end}{post}"


def main() -> None:
    text = DOC.read_text()
    text = _replace(text, START, END, build_table())
    text = _replace(text, CAT_START, CAT_END, build_categories())
    DOC.write_text(text)
    print(f"updated table + categories in {DOC} ({len(PROBLEM_REGISTRY)} problems)")


if __name__ == "__main__":
    main()
