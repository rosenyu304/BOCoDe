"""BoCoDe optimization problems (opt_problems/engineering). Access problems via bocode.get_problem or bocode.<Name>."""

# Compatibility shim: slientruss3d (the FE solver used by the Truss* problems) references
# ``np.bool8``, an alias removed in numpy 2.0. Restore it when missing (no-op on numpy
# < 2.0). This package is imported before any Truss problem does its lazy slientruss3d
# import, so the alias is in place by evaluation time.
import numpy as _np

if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_
