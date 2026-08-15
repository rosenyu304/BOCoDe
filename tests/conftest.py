"""Test-suite-wide hooks.

CI installs only the core/dev dependencies and runs offline, so problems that need
an optional extra (e.g. ``slientruss3d`` for the trusses, ``gym`` for TSP) or that
download data from the BoCoDe Hugging Face dataset cannot run there. Rather than
fail, any test whose *runtime* error is "this optional dependency isn't installed"
or "that remote data file is unreachable" is converted to a **skip**. When the
extras and data are present (e.g. locally) the same tests run normally.

Module-level import guards (``pytest.skip(..., allow_module_level=True)``) handle
the tests that import optional problems at the top of the file; this hook handles
the ones that only hit the missing dependency / download deep inside ``evaluate``.
"""

from __future__ import annotations

import urllib.error

import pytest


def _skip_reason(exc: BaseException) -> str | None:
    msg = str(exc)
    # Any runtime ImportError means an optional dependency isn't installed:
    # ModuleNotFoundError (slientruss3d, gym), the registry's "requires the optional
    # ..." error, or pandas-style "Import openpyxl failed". (Real import bugs surface
    # at collection, which this does not touch.)
    if isinstance(exc, ImportError):
        return f"optional dependency unavailable: {msg[:160]}"
    if isinstance(exc, RuntimeError) and (
        "Failed to download" in msg or "BOCODE_DATA_BASE_URL" in msg
    ):
        return f"remote data unavailable: {msg.splitlines()[0][:160]}"
    if isinstance(exc, urllib.error.URLError):
        return f"network unavailable: {exc}"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and call.excinfo is not None:
        reason = _skip_reason(call.excinfo.value)
        if reason:
            report.outcome = "skipped"
            # pytest's terminal reporter expects a (path, lineno, message) tuple
            # for skipped reports, not a bare string.
            relpath, lineno, _ = item.location
            report.longrepr = (relpath, (lineno or 0) + 1, f"Skipped: {reason}")
