"""Euler-Bernoulli beam bending — mixed-variable version (cross-section catalog).

A cantilever beam clamped at one end and loaded vertically at the other. Unlike the
continuous ``EulerBernoulliBeamBending`` problem, the cross-section here is chosen
from a *catalog* of 12 beam profiles (a categorical variable), while the length and
section area are continuous — a canonical mixed continuous/categorical benchmark
(Cuesta-Ramirez et al. 2022, built on the profile catalog of Roustant et al. 2020).

Design variables ``(x1, x2, u1)``:
- ``x1`` continuous [0, 1] -> length ``L = 10 + 10*x1`` (so L in [10, 20]);
- ``x2`` continuous [0, 1] -> section area ``S = 1 + x2`` (so S in [1, 2]);
- ``u1`` categorical, the normalized moment of inertia ``I~``, one of 12 catalog
  values (4 cross-section shapes x 3 hollowness levels).

Objective (minimize): a deflection/weight compromise
    y = P*L^3 / (3*E*S^2*I~) + alpha*L*S,   P = 600, E = 600, alpha = 60.
Unconstrained (box only). Known optimum y* ~ 1287.385 at (x1, x2, I~) = (0, 0.43,
0.380) (the hollow-square profile, the largest inertia).

Sources:
J. Cuesta-Ramirez, R. Le Riche, O. Roustant, G. Perrin, C. Durantin, A. Gliere. A comparison of mixed-variables Bayesian optimization approaches. Advances in Modeling and Simulation in Engineering Sciences, 2022. arXiv:2111.01533.
O. Roustant et al. Group kernels for Gaussian process metamodels with categorical inputs. SIAM/ASA J. Uncertainty Quantification 8(2):775-806, 2020.
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem, DataType

_P = 600.0  # vertical load [N]
_E = 600.0  # Young's modulus (normalized units of the source)
_ALPHA = 60.0  # deflection/weight trade-off weight

# 12 normalized moments of inertia (the beam-profile catalog), in level order.
_INERTIA = [
    0.083,
    0.139,
    0.380,  # square: solid -> hollow
    0.080,
    0.133,
    0.363,  # circle
    0.086,
    0.136,
    0.360,  # I-beam
    0.092,
    0.138,
    0.369,  # star/cross
]


class EulerBeamMixed(BenchmarkProblem):
    """Minimize cantilever-beam deflection+weight (3 vars, mixed; 12-profile catalog).

    Variables: ``x1`` [0,1] (length), ``x2`` [0,1] (area), ``u1`` (moment of inertia,
    12 catalog values). ``is_discrete=False`` relaxes ``u1`` to a continuous range.
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, is_discrete: bool = True) -> None:
        bounds = [(0.0, 1.0), (0.0, 1.0), (min(_INERTIA), max(_INERTIA))]
        if is_discrete:
            self.variable_types = ["continuous", "continuous", list(_INERTIA)]
            self.input_type = DataType.MIXED
        else:
            self.variable_types = None
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            bounds=bounds,
            optimum=[
                1286.97
            ],  # global at (x1, x2, I~)=(0, 0.43, 0.380); paper cites 1287.385
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        x1, x2, inertia = x[:, 0], x[:, 1], x[:, 2]

        L = 10.0 + 10.0 * x1
        S = 1.0 + x2
        deflection = _P * L**3 / (3.0 * _E * S**2 * inertia)
        y = deflection + _ALPHA * L * S

        fx = torch.tensor(-y, dtype=torch.float64).reshape(-1, 1)  # maximize -y
        return None, fx
