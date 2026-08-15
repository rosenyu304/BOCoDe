"""Stepped cantilever beam — 5-segment volume minimization (continuous or mixed).

Minimize the material volume of a cantilever beam built from five equal-length
stepped segments, each with its own rectangular width ``b_i`` and height ``h_i``,
loaded by a point force ``P`` at the free tip. Segment 1 is at the wall, segment 5
at the tip. Ten design variables in the order ``[b1..b5, h1..h5]``.

With ``is_discrete=True`` (the Firefly mixed-variable form) the first three
segments use discrete/integer cross-sections; segments 4-5 stay continuous. With
``is_discrete=False`` all ten variables are continuous (the Bat-algorithm form).

Objective (minimize): ``V = 100 * sum_i b_i h_i`` (segment length l = 100 cm).
Eleven constraints (feasible <= 0): five bending-stress limits (one per segment,
moment = P * distance from the tip), a tip-deflection limit (sum of segment
compliances), and five aspect-ratio limits ``h_i / b_i <= 20``.

Reference optima: continuous ``V* ~ 61914.9`` (Bat, Gandomi & Yang); mixed
``V* ~ 63893.5`` (Firefly, Gandomi, Yang & Alavi 2011).

Sources:
A. H. Gandomi, X.-S. Yang, A. H. Alavi. Mixed variable structural optimization using Firefly Algorithm. Computers & Structures 89(23-24):2325-2336, 2011.
A. H. Gandomi, X.-S. Yang, A. H. Alavi, S. Talatahari. Bat algorithm for constrained optimization tasks. Neural Computing and Applications 22(6):1239-1255, 2013.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

_P = 50000.0  # tip load [N]
_SIGMA_D = 14000.0  # allowable bending stress [N/cm^2]
_E = 2e7  # elastic modulus [N/cm^2]
_DELTA_MAX = 2.7  # allowable tip deflection [cm]
_L_SEG = 100.0  # length of each of the 5 segments [cm]


class SteppedCantileverBeam(BenchmarkProblem):
    """Minimize a 5-step cantilever-beam volume (10 vars, 11 constraints; mixed default).

    Variables (order): widths ``b1..b5`` then heights ``h1..h5``. ``is_discrete=True``
    restricts the first three segments to discrete cross-sections (b1 integer 1-5;
    b2,b3 in {2.4,2.6,2.8,3.1}; h1,h2 in {45,50,55,60}; h3 integer 30-65).
    """

    available_dimensions = 10
    num_objectives = 1
    num_constraints = 11

    def __init__(self, is_discrete: bool = True) -> None:
        bounds = [(1.0, 5.0)] * 5 + [(30.0, 65.0)] * 5
        if is_discrete:
            self.variable_types = [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # b1
                [2.4, 2.6, 2.8, 3.1],  # b2
                [2.4, 2.6, 2.8, 3.1],  # b3
                "continuous",  # b4
                "continuous",  # b5
                [45.0, 50.0, 55.0, 60.0],  # h1
                [45.0, 50.0, 55.0, 60.0],  # h2
                "integer",  # h3
                "continuous",  # h4
                "continuous",  # h5
            ]
        else:
            self.variable_types = None
        super().__init__(
            dim=10,
            num_objectives=1,
            num_constraints=11,
            bounds=bounds,
            optimum=[63893.53] if is_discrete else [61914.87],
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        b = [x[:, i] for i in range(5)]
        h = [x[:, 5 + i] for i in range(5)]
        l = _L_SEG

        V = l * sum(b[i] * h[i] for i in range(5))

        # Bending stress per segment: moment = P * (distance from tip to segment's
        # far end) = P * k * l, with k = 1 (tip, seg 5) .. 5 (wall, seg 1).
        g = []
        for k, seg in enumerate([4, 3, 2, 1, 0]):  # segments 5,4,3,2,1
            g.append(6.0 * _P * (k + 1) * l / (b[seg] * h[seg] ** 2) - _SIGMA_D)

        # Tip deflection: sum of segment compliances with weights (1,7,19,37,61)
        # from the tip (seg 5) to the wall (seg 1); I_i = b_i h_i^3 / 12.
        inertia = [b[i] * h[i] ** 3 / 12.0 for i in range(5)]
        weights = [1.0, 7.0, 19.0, 37.0, 61.0]  # seg 5,4,3,2,1
        defl = (_P * l**3 / (3.0 * _E)) * sum(
            weights[k] / inertia[seg] for k, seg in enumerate([4, 3, 2, 1, 0])
        )
        g.append(defl - _DELTA_MAX)

        # Aspect-ratio limits h_i / b_i <= 20 (segments 5,4,3,2,1).
        for seg in [4, 3, 2, 1, 0]:
            g.append(h[seg] / b[seg] - 20.0)

        gx = torch.tensor(np.stack(g, axis=1), dtype=torch.float64)
        fx = torch.tensor(-V, dtype=torch.float64).reshape(-1, 1)  # maximize -volume
        return gx, fx
