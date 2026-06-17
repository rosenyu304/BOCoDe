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

from bocode import list_problems  # noqa: E402
from bocode.registry import get_metadata  # noqa: E402

OUT = ROOT / "docs" / "source" / "_generated" / "benchmark_table.html"

# Facets rendered as toggle-button groups (label -> per-row value accessor).
_FACETS = [
    ("Application", "application"),
    ("Objectives", "objectives"),
    ("Constraints", "constraints"),
    ("Variables", "variables"),
]


def _row_facets(m: dict) -> dict:
    return {
        "application": str(m.get("application") or "Unknown"),
        "objectives": "single" if (m.get("num_objectives") or 1) == 1 else "multi",
        "constraints": "constrained"
        if (m.get("num_constraints") or 0) > 0
        else "unconstrained",
        "variables": str(m.get("input_type") or "continuous"),
    }


def _fmt(v) -> str:
    return "—" if v is None else str(v)


def build_html() -> str:
    rows = [(n, get_metadata(n)) for n in list_problems()]
    data = [(n, m, _row_facets(m)) for n, m in rows]

    # distinct values per facet (for the toggle buttons)
    facet_values: dict[str, list[str]] = {}
    for _, key in _FACETS:
        facet_values[key] = sorted({f[key] for _, _, f in data})

    parts: list[str] = []
    parts.append(_CSS)
    parts.append('<div class="bocode-table">')
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

    # table
    parts.append(
        "<table id='bc-table'><thead><tr>"
        "<th>Problem</th><th>Application</th><th>Suite</th><th>Dim</th>"
        "<th>#Obj</th><th>#Constr</th><th>f_opt</th><th>Variables</th>"
        "<th>Convex</th><th>NP-hard</th></tr></thead><tbody>"
    )
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
            f"<td>{html.escape(_fmt(m.get('convex')))}</td>"
            f"<td>{html.escape(_fmt(m.get('np_hard')))}</td></tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append(_JS)
    return "\n".join(parts)


_CSS = """<style>
.bocode-table { font-size: 0.9em; }
.bocode-table #bc-search { width: 100%; max-width: 420px; padding: 6px 10px; margin: 0 0 12px;
  border: 1px solid var(--color-background-border, #ccc); border-radius: 6px; background: inherit; color: inherit; }
.bocode-table .bc-facet { margin: 4px 0; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.bocode-table .bc-facet-label { font-weight: 600; min-width: 92px; }
.bocode-table .bc-btn { padding: 3px 10px; border: 1px solid var(--color-background-border, #ccc);
  border-radius: 14px; background: transparent; color: inherit; cursor: pointer; font-size: 0.85em; }
.bocode-table .bc-btn:hover { border-color: var(--color-brand-primary, #2980b9); }
.bocode-table .bc-btn.active { background: var(--color-brand-primary, #2980b9); color: #fff;
  border-color: var(--color-brand-primary, #2980b9); }
.bocode-table .bc-count { margin: 10px 0; opacity: 0.8; }
.bocode-table table { border-collapse: collapse; width: 100%; }
.bocode-table th, .bocode-table td { border: 1px solid var(--color-background-border, #ddd);
  padding: 4px 8px; text-align: left; }
.bocode-table th { cursor: pointer; position: sticky; top: 0; background: var(--color-background-secondary, #f5f5f5); }
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
  var active = {};  // facet -> Set of active values

  function apply() {
    var q = (search.value || '').toLowerCase();
    var n = 0;
    rows.forEach(function (r) {
      var ok = r.getAttribute('data-name').indexOf(q) !== -1;
      Object.keys(active).forEach(function (facet) {
        if (active[facet].size && !active[facet].has(r.getAttribute('data-' + facet))) ok = false;
      });
      r.classList.toggle('bc-hidden', !ok);
      if (ok) n++;
    });
    shown.textContent = n;
  }

  buttons.forEach(function (b) {
    var facet = b.getAttribute('data-facet'), val = b.getAttribute('data-value');
    active[facet] = active[facet] || new Set();
    b.addEventListener('click', function () {
      b.classList.toggle('active');
      if (active[facet].has(val)) active[facet].delete(val); else active[facet].add(val);
      apply();
    });
  });
  search.addEventListener('input', apply);

  // click a header to sort by that column
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
      asc = !asc;
      rows.forEach(function (r) { tbody.appendChild(r); });
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
