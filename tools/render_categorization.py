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

from bocode.registry import PROBLEM_REGISTRY, get_metadata  # noqa: E402

DOC = ROOT / "CATEGORIZATION.md"
START = "<!-- TABLE:START -->"
END = "<!-- TABLE:END -->"


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


def main() -> None:
    table = build_table()
    text = DOC.read_text()
    pre, _, rest = text.partition(START)
    _, _, post = rest.partition(END)
    new = f"{pre}{START}\n\n{table}\n{END}{post}"
    DOC.write_text(new)
    print(f"updated table in {DOC} ({len(PROBLEM_REGISTRY)} problems)")


if __name__ == "__main__":
    main()
