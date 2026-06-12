"""Backward-compatibility shims for the pre-2026_06 namespace layout.

Problems used to live under ``bocode.Engineering``, ``bocode.NEORL``, and
``bocode.LassoBench``. They are now resolved by name from the central registry.
The shim modules forward attribute access to the registry and emit a
``DeprecationWarning`` so existing code keeps working for one release.
"""

from __future__ import annotations

import warnings

from .registry import PROBLEM_REGISTRY, get_problem


def deprecated_getattr(old_namespace: str):
    """Build a module ``__getattr__`` that forwards to the registry with a warning."""

    def __getattr__(name: str):
        if name in PROBLEM_REGISTRY:
            warnings.warn(
                f"bocode.{old_namespace}.{name} is deprecated; use "
                f"bocode.{name} or bocode.get_problem({name!r}) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return get_problem(name)
        raise AttributeError(f"module 'bocode.{old_namespace}' has no attribute {name!r}")

    return __getattr__
