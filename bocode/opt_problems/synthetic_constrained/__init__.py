"""Single-objective, inequality-constrained synthetic test functions.

Ported from the BO Engineering Benchmark suite
(https://github.com/rosenyu304/BOEngineeringBenchmark). The reference problems are
constrained *minimization* problems with feasibility ``g(x) <= 0``; here the
objective is negated so BoCoDe's maximizer optimizes it, and the constraint sign
convention (feasible when ``<= 0``) already matches BoCoDe. Each problem returns
``(constraints, values)`` from ``_evaluate_implementation``.
"""

from .ConstrainedAckley import (
    ConstrainedAckley2D,
    ConstrainedAckley6D,
    ConstrainedAckley10D,
)
from .GKXWC1 import GKXWC1
from .GKXWC2 import GKXWC2
from .JLH1 import JLH1
from .JLH2 import JLH2

__all__ = [
    "JLH1",
    "JLH2",
    "GKXWC1",
    "GKXWC2",
    "ConstrainedAckley2D",
    "ConstrainedAckley6D",
    "ConstrainedAckley10D",
]
