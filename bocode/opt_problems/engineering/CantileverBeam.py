"""Cantilever beam — the continuous (Bat-algorithm) form of the 5-step cantilever beam.

This is :class:`SteppedCantileverBeam` with all ten variables continuous. It is kept
as a separate registry entry under its historical name.

Sources:
X.-S. Yang, A. H. Gandomi. Bat algorithm: a novel approach for global engineering optimization. Engineering Computations 29(5):464-483, 2012.
"""

from __future__ import annotations

from .SteppedCantileverBeam import SteppedCantileverBeam


class CantileverBeam(SteppedCantileverBeam):
    """Minimize a 5-step cantilever-beam volume (10 continuous vars, 11 constraints)."""

    tags = {"single_objective", "constrained", "10D"}

    def __init__(self) -> None:
        super().__init__(is_discrete=False)
