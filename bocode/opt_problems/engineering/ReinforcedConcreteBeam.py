"""Reinforced concrete beam — single-objective design (continuous or mixed-variable).

Minimize the cost of a reinforced concrete beam (Firefly Case IV). Three design
variables: the reinforcement bar area ``As``, the beam width ``b``, and the beam
depth ``h``. With ``is_discrete=True`` (the Firefly mixed-variable form) ``As`` is
chosen from a catalog of standard rebar areas (Table 10) and the width is an
integer number of inches; the depth stays continuous. With ``is_discrete=False``
all three are continuous.

Objective (minimize): ``f = 29.4 As + 0.6 b h``.
Constraints (feasible <= 0): ``g1 = b/h - 4`` (depth/width ratio) and the simplified
ACI 318-77 strength requirement ``g2 = 180 + 7.375 As^2/h - As b`` (the variable
roles reproduce the paper's published feasible optimum).

Reference optimum (Gandomi, Yang & Alavi 2011, Table 12): ``f* ~ 359.208`` at
``As = 6.32 in^2, b = 34 in, h = 8.5 in``.

Sources:
A. H. Gandomi, X.-S. Yang, A. H. Alavi. Mixed variable structural optimization using Firefly Algorithm. Computers & Structures 89(23-24):2325-2336, 2011.
H. Amir, T. Hasegawa. Nonlinear mixed-discrete structural optimization. Journal of Structural Engineering 115(3):626-646, 1989.
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem

# Table 10: standardized reinforcing-bar areas As [in^2] (the "5#8" print artifact
# dropped); the usable, sorted value set.
_AS_VALUES = [
    0.20,
    0.40,
    0.44,
    0.60,
    0.62,
    0.79,
    0.80,
    0.88,
    0.93,
    1.00,
    1.20,
    1.24,
    1.32,
    1.40,
    1.55,
    1.58,
    1.60,
    1.76,
    1.80,
    1.86,
    2.00,
    2.17,
    2.20,
    2.37,
    2.40,
    2.48,
    2.64,
    2.79,
    2.80,
    3.00,
    3.08,
    3.10,
    3.16,
    3.41,
    3.52,
    3.60,
    3.72,
    3.95,
    3.96,
    4.00,
    4.03,
    4.20,
    4.34,
    4.40,
    4.65,
    4.74,
    4.80,
    5.00,
    5.28,
    5.40,
    5.53,
    5.72,
    6.00,
    6.16,
    6.32,
    6.60,
    7.11,
    7.20,
    7.80,
    7.90,
    8.00,
    8.40,
    8.69,
    9.00,
    9.48,
    10.27,
    11.06,
    11.85,
    12.00,
    13.00,
    14.00,
    15.00,
]


class ReinforcedConcreteBeam(BenchmarkProblem):
    """Minimize reinforced-concrete-beam cost (3 vars, 2 constraints; mixed by default).

    Variables (order): ``As`` [0.2, 15] (rebar areas from Table 10 if ``is_discrete``),
    ``b`` width [28, 40] (integer if ``is_discrete``), ``h`` depth [5, 10] (continuous).
    """

    available_dimensions = 3
    num_objectives = 1
    num_constraints = 2

    def __init__(self, is_discrete: bool = True) -> None:
        bounds = [(0.2, 15.0), (28.0, 40.0), (5.0, 10.0)]
        if is_discrete:
            self.variable_types = [list(_AS_VALUES), "integer", "continuous"]
        else:
            self.variable_types = None
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=2,
            bounds=bounds,
            optimum=[359.208],  # As=6.32, b=34, h=8.5
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)
        As = X[:, 0]
        b = X[:, 1]  # width (integer when is_discrete)
        h = X[:, 2]  # depth (continuous)

        fx = (-(29.4 * As + 0.6 * b * h)).reshape(n, 1)  # maximize -cost

        gx = torch.zeros((n, 2), dtype=X.dtype)
        gx[:, 0] = b / h - 4.0  # depth/width ratio
        gx[:, 1] = 180.0 + 7.375 * As * As / h - As * b  # ACI 318-77 strength
        return gx, fx


class ReinforcedConcreteBeamContinuous(ReinforcedConcreteBeam):
    """Fully-continuous relaxation of ReinforcedConcreteBeam: all 3 variables continuous, 2
    inequality constraints (feasible region ~17% by volume). A well-posed SO-constrained
    engineering problem."""

    tags = {"single_objective", "constrained", "3D", "continuous"}

    def __init__(self):
        super().__init__(is_discrete=False)
