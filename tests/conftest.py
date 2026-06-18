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
    if isinstance(exc, ModuleNotFoundError):
        return f"optional dependency not installed: {exc}"
    if isinstance(exc, ImportError) and "requires the optional" in msg:
        return f"optional extra not installed: {msg[:160]}"
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
            report.longrepr = reason
