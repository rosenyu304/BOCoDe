"""Render the interactive benchmark table for the docs (MIPLIB-style).

Generates a self-contained HTML fragment (filter toggle buttons + a searchable,
sortable table) from the per-problem metadata, written to
``docs/source/_generated/benchmark_table.html`` and embedded by
``docs/source/benchmark_table.rst`` via ``.. raw:: html :file:``. Run::

    python tools/render_docs_table.py

No external JS/CSS dependencies, so it works on Read the Docs and in light/dark
Furo themes. Rerun after adding problems or regenerating metadata.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bocode import get_problem, list_problems, list_synthetic  # noqa: E402
from bocode.registry import get_metadata  # noqa: E402

OUT = ROOT / "docs" / "source" / "_generated" / "benchmark_table.html"


def _synthetic_meta(name: str) -> dict:
    """Build a metadata-like dict for a synthetic test function (not in the registry)."""
    inst = get_problem(name)()
    opt = getattr(inst, "optimum", None)
    if isinstance(opt, (list, tuple)) and len(opt) == 1:
        opt = opt[0]
    itype = getattr(getattr(inst, "input_type", None), "value", None) or "continuous"
    return {
        "name": name,
        "application": "Synthetic",
        "suite": "Synthetic test function",
        "dim": getattr(inst, "dim", None),
        "num_objectives": getattr(inst, "num_objectives", 1),
        "num_constraints": getattr(inst, "num_constraints", 0),
        "input_type": itype,
        "f_opt": round(float(opt), 6) if isinstance(opt, (int, float)) else None,
        # scalable when the class declares a dimension range (tuple/list), e.g. Ackley
        "scalable": isinstance(
            getattr(type(inst), "available_dimensions", None), (tuple, list, set)
        ),
        "convex": "unknown",
        "np_hard": "unknown",
    }


# Facets rendered as toggle-button groups (label -> per-row value accessor).
_FACETS = [
    ("Application", "application"),
    ("Objectives", "objectives"),
    ("Constraints", "constraints"),
    ("Variables", "variables"),
    ("Dimension", "dimension"),
    ("Scalable", "scalable"),
]

# Dimension buckets (low / mid / high) and the order their buttons appear in.
_DIM_ORDER = ["<=10", "11-100", ">100"]


def _dim_bucket(dim) -> str:
    if not isinstance(dim, (int, float)):
        return "unknown"
    if dim <= 10:
        return "<=10"
    if dim <= 100:
        return "11-100"
    return ">100"


def _row_facets(m: dict) -> dict:
    return {
        "application": str(m.get("application") or "Unknown"),
        "objectives": "single" if (m.get("num_objectives") or 1) == 1 else "multi",
        "constraints": "constrained"
        if (m.get("num_constraints") or 0) > 0
        else "unconstrained",
        "variables": str(m.get("input_type") or "continuous"),
        "dimension": _dim_bucket(m.get("dim")),
        "scalable": "Yes" if m.get("scalable") else "No",
    }


def _fmt(v) -> str:
    return "—" if v is None else str(v)


def build_html() -> str:
    # Real-world problems from the registry + the synthetic test functions, the
    # latter tagged Application = "Synthetic" so they can be filtered/toggled.
    rows = [(n, get_metadata(n)) for n in list_problems()]
    rows += [(n, _synthetic_meta(n)) for n in list_synthetic()]
    data = [(n, m, _row_facets(m)) for n, m in rows]

    # distinct values per facet (for the toggle buttons)
    facet_values: dict[str, list[str]] = {}
    for _, key in _FACETS:
        vals = {f[key] for _, _, f in data}
        if key == "dimension":
            facet_values[key] = [v for v in _DIM_ORDER if v in vals]
        else:
            facet_values[key] = sorted(vals)

    parts: list[str] = []
    parts.append(_CSS)
    parts.append('<div class="bocode-table">')
    parts.append(
        "<p class='bc-intro'>An interactive, filterable table of every problem in BoCoDe. "
        "Toggle the buttons to filter by application, objectives, constraints, variable "
        "type, dimension, and scalability (one selection per category); type to search by "
        "name; click a column header to sort.</p>"
    )
    parts.append(
        '<input type="text" id="bc-search" placeholder="Search by problem name…" />'
    )

    # filter buttons
    for label, key in _FACETS:
        parts.append('<div class="bc-facet">')
        parts.append(f"<span class='bc-facet-label'>{html.escape(label)}:</span>")
        for val in facet_values[key]:
            parts.append(
                f"<button class='bc-btn' data-facet='{key}' data-value='{html.escape(val)}'>"
                f"{html.escape(val)}</button>"
            )
        parts.append("</div>")

    parts.append(
        f'<p class="bc-count"><span id="bc-shown">{len(data)}</span> of {len(data)} problems shown</p>'
    )

    # table (each header carries a sort indicator so users see columns are sortable)
    headers = [
        "Problem",
        "Application",
        "Suite",
        "Dim",
        "#Obj",
        "#Constr",
        "f_opt",
        "Variables",
        "Scalable",
    ]
    header_html = "".join(
        f"<th>{html.escape(h)}<span class='bc-sort'>⇅</span></th>" for h in headers
    )
    parts.append(f"<table id='bc-table'><thead><tr>{header_html}</tr></thead><tbody>")
    for n, m, f in data:
        attrs = " ".join(f"data-{k}='{html.escape(v)}'" for k, v in f.items())
        parts.append(
            f"<tr {attrs} data-name='{html.escape(n.lower())}'>"
            f"<td><code>{html.escape(n)}</code></td>"
            f"<td>{html.escape(_fmt(m.get('application')))}</td>"
            f"<td>{html.escape(_fmt(m.get('suite')))}</td>"
            f"<td>{_fmt(m.get('dim'))}</td>"
            f"<td>{_fmt(m.get('num_objectives'))}</td>"
            f"<td>{_fmt(m.get('num_constraints'))}</td>"
            f"<td>{_fmt(m.get('f_opt'))}</td>"
            f"<td>{html.escape(_fmt(m.get('input_type')))}</td>"
            f"<td>{'Yes' if m.get('scalable') else 'No'}</td></tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append(_JS)
    return "\n".join(parts)


_CSS = """<style>
.bocode-table { font-size: 0.9em; }
/* Widen furo's fixed 46em content column for THIS page only (via :has) so the intro
   text and filter buttons are as wide as the table. This expands rightward into the
   margin/ToC space; it never shifts left under the sidebar. */
.content:has(.bocode-table) { width: min(65em, 92vw); }
.bocode-table #bc-search { width: 100%; max-width: 420px; padding: 6px 10px; margin: 0 0 12px;
  border: 1px solid var(--color-background-border, #ccc); border-radius: 6px; background: inherit; color: inherit; }
.bocode-table .bc-facet { margin: 6px 0; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.bocode-table .bc-facet-label { font-weight: 600; min-width: 92px; }
.bocode-table .bc-btn { padding: 6px 14px; border: 2px solid var(--color-background-border, #bbb);
  border-radius: 16px; background: var(--color-background-primary, #fff); color: inherit; cursor: pointer;
  font-size: 0.9em; font-weight: 500; line-height: 1.2; transition: background 0.15s, border-color 0.15s; }
.bocode-table .bc-btn:hover { border-color: var(--color-brand-primary, #2980b9);
  background: var(--color-background-secondary, #f0f0f0); }
.bocode-table .bc-btn.active { background: var(--color-brand-primary, #2980b9); color: #fff;
  border-color: var(--color-brand-primary, #2980b9); }
.bocode-table .bc-count { margin: 10px 0; opacity: 0.8; }
.bocode-table table { border-collapse: collapse; width: 100%; }
.bocode-table th, .bocode-table td { border: 1px solid var(--color-background-border, #ddd);
  padding: 4px 8px; text-align: left; }
.bocode-table th { cursor: pointer; position: sticky; top: 0; white-space: nowrap;
  background: var(--color-background-secondary, #f5f5f5); }
.bocode-table th:hover { background: var(--color-background-hover, #ececec); }
.bocode-table .bc-sort { margin-left: 5px; opacity: 0.45; font-size: 0.8em; }
.bocode-table th.bc-sorted .bc-sort { opacity: 0.95; color: var(--color-brand-primary, #2980b9); }
.bocode-table tr.bc-hidden { display: none; }
</style>"""

_JS = """<script>
(function () {
  var root = document.querySelector('.bocode-table');
  if (!root) return;
  var search = root.querySelector('#bc-search');
  var buttons = root.querySelectorAll('.bc-btn');
  var rows = Array.prototype.slice.call(root.querySelectorAll('#bc-table tbody tr'));
  var shown = root.querySelector('#bc-shown');
  var active = {};  // facet -> single active value (radio: one per category)

  function apply() {
    var q = (search.value || '').toLowerCase();
    var n = 0;
    rows.forEach(function (r) {
      var ok = r.getAttribute('data-name').indexOf(q) !== -1;
      Object.keys(active).forEach(function (facet) {
        if (active[facet] && r.getAttribute('data-' + facet) !== active[facet]) ok = false;
      });
      r.classList.toggle('bc-hidden', !ok);
      if (ok) n++;
    });
    shown.textContent = n;
  }

  buttons.forEach(function (b) {
    var facet = b.getAttribute('data-facet'), val = b.getAttribute('data-value');
    b.addEventListener('click', function () {
      if (active[facet] === val) {
        // clicking the active button again clears this category's filter
        active[facet] = null;
        b.classList.remove('active');
      } else {
        // single-select: deselect the others in this category, select this one
        active[facet] = val;
        buttons.forEach(function (o) {
          if (o.getAttribute('data-facet') === facet) o.classList.remove('active');
        });
        b.classList.add('active');
      }
      apply();
    });
  });
  search.addEventListener('input', apply);

  // click a header to sort by that column (with ▲/▼ direction indicators)
  var headers = root.querySelectorAll('#bc-table th');
  headers.forEach(function (h, idx) {
    var asc = true;
    h.addEventListener('click', function () {
      var tbody = root.querySelector('#bc-table tbody');
      rows.sort(function (a, b) {
        var x = a.children[idx].textContent.trim(), y = b.children[idx].textContent.trim();
        var nx = parseFloat(x), ny = parseFloat(y);
        if (!isNaN(nx) && !isNaN(ny)) return asc ? nx - ny : ny - nx;
        return asc ? x.localeCompare(y) : y.localeCompare(x);
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
      headers.forEach(function (hh) {
        hh.classList.remove('bc-sorted');
        var s = hh.querySelector('.bc-sort');
        if (s) s.textContent = '⇅';
      });
      h.classList.add('bc-sorted');
      var ind = h.querySelector('.bc-sort');
      if (ind) ind.textContent = asc ? '▲' : '▼';
      asc = !asc;
    });
  });
})();
</script>"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html())
    print(f"wrote {OUT} ({len(list_problems())} problems)")


if __name__ == "__main__":
    main()
