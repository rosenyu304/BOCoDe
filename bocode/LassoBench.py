"""Deprecated namespace shim — see bocode._compat. Use bocode.<Problem> instead."""

from ._compat import deprecated_getattr

__getattr__ = deprecated_getattr("LassoBench")
