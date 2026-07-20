"""Welded-beam design with a nominal categorical material (4 continuous + 1 categorical, K=3).

The canonical welded-beam cost-minimization problem (continuous weld/beam geometry
``h, l, t, b``) extended with one *unordered* categorical ``material`` that swaps
the elastic moduli, the allowable stresses, and a relative material-cost factor.
The steel arm reproduces the canonical continuous welded beam (best known cost
``~1.7249``).

Objective (minimize the fabrication cost)::

    f = 1.10471 h^2 l + 0.04811 * cost_factor[m] * t b (14 + l)

Five inequality constraints, feasible when ``<= 0`` (all expressed as *relative*
violations so the constraint-active optimum is not a numerical cliff)::

    g1 = tau / tau_max[m] - 1            (weld shear stress)
    g2 = sigma / sigma_max[m] - 1        (beam bending stress)
    g3 = (h - b) / 5                     (geometry, h <= b)
    g4 = delta / 0.25 - 1                (tip deflection)
    g5 = (P - Pc) / P                    (buckling; non-positive Pc reads as strongly infeasible)

with ``P = 6000 lb``, ``L = 14 in``, and

    tau'  = P / (sqrt(2) h l)
    M     = P (L + l/2),  R = sqrt(l^2/4 + ((h + t)/2)^2)
    J     = 2 sqrt(2) h l (l^2/12 + ((h + t)/2)^2)
    tau'' = M R / J
    tau   = sqrt(tau'^2 + 2 tau' tau'' (l / (2R)) + tau''^2)
    sigma = 6 P L / (b t^2),  delta = 4 P L^3 / (E t^3 b)
    Pc    = (4.013 E sqrt(t^2 b^6 / 36) / L^2) (1 - (t / (2L)) sqrt(E / (4G)))

Materials (``E``, ``G``, ``sigma_max``, ``tau_max`` in psi, plus a relative
cost-per-volume factor): steel (A36, the canonical Coello & Montes values),
aluminum (6061-T6), titanium (Ti-6Al-4V). The elastic moduli and yield-derived
allowable stresses are standard; ``cost_factor`` is an order-of-magnitude design
proxy for relative material cost per unit volume, not a costing model — do not
draw cross-material *economic* conclusions from it.

Variables: ``h`` in ``[0.125, 5]``, ``l`` in ``[0.1, 10]``, ``t`` in ``[0.1, 10]``,
``b`` in ``[0.125, 5]`` (continuous); ``material`` categorical with K=3 nominal
levels, so the decision variable carries the level *index*.

Reference optimum ``f* ~ 1.7249`` (minimization sense) on the steel arm.
``evaluate()`` returns ``-f`` because BoCoDe maximizes; constraints are feasible
when ``<= 0``.

Sources:
C. A. Coello Coello, E. Mezura Montes. Constraint-handling in genetic algorithms through the use of dominance-based tournament selection. Advanced Engineering Informatics 16(3):193-203, 2002.
A. H. Gandomi, X.-S. Yang, A. H. Alavi. Mixed variable structural optimization using Firefly Algorithm. Computers & Structures 89(23-24):2325-2336, 2011.
"""

from __future__ import annotations

import math

import torch

from ...base import BenchmarkProblem
from ._common import cat_index

_P = 6000.0  # applied load [lb]
_L = 14.0  # cantilever length [in]
_DELTA_MAX = 0.25  # allowable tip deflection [in]
_EPS = 1e-9


class WeldedBeamCategorical(BenchmarkProblem):
    """Welded-beam cost with a 3-level nominal material categorical (5-D, mixed, 5 constraints)."""

    available_dimensions = 5
    num_objectives = 1
    num_constraints = 5

    #: material -> (E [psi], G [psi], sigma_max [psi], tau_max [psi], relative cost factor)
    MATERIALS = {
        "steel": (30.0e6, 12.0e6, 30000.0, 13600.0, 1.0),  # A36 (canonical)
        "aluminum": (10.0e6, 3.80e6, 33000.0, 15000.0, 1.1),  # 6061-T6
        "titanium": (16.5e6, 6.20e6, 100000.0, 45000.0, 8.0),  # Ti-6Al-4V
    }

    def __init__(self) -> None:
        self.variable_types = ["continuous"] * 4 + [[0.0, 1.0, 2.0]]
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=5,
            bounds=[(0.125, 5.0), (0.1, 10.0), (0.1, 10.0), (0.125, 5.0), (0.0, 2.0)],
            x_opt=[[0.2057, 3.4705, 9.0366, 0.2057, 0.0]],  # steel arm
            optimum=[1.7249],  # minimization sense; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        h = x[:, 0].clamp_min(_EPS)
        l = x[:, 1].clamp_min(_EPS)
        t = x[:, 2].clamp_min(_EPS)
        b = x[:, 3].clamp_min(_EPS)
        m = cat_index(x[:, 4], len(self.MATERIALS))

        props = torch.tensor(list(self.MATERIALS.values()), dtype=torch.float64)
        E = props[m, 0]
        G = props[m, 1]
        sigma_max = props[m, 2]
        tau_max = props[m, 3]
        cost_factor = props[m, 4]

        cost = 1.10471 * h * h * l + 0.04811 * cost_factor * t * b * (14.0 + l)

        # Weld shear stress
        tau_p = _P / (math.sqrt(2.0) * h * l)
        M = _P * (_L + l / 2.0)
        R = torch.sqrt(l * l / 4.0 + ((h + t) / 2.0) ** 2)
        J = 2.0 * (math.sqrt(2.0) * h * l * (l * l / 12.0 + ((h + t) / 2.0) ** 2))
        tau_pp = M * R / J.clamp_min(_EPS)
        tau = torch.sqrt(
            (tau_p**2 + 2.0 * tau_p * tau_pp * (l / (2.0 * R)) + tau_pp**2).clamp_min(
                0.0
            )
        )

        # Bending stress, tip deflection, critical buckling load
        sigma = 6.0 * _P * _L / (b * t * t)
        delta = 4.0 * _P * _L**3 / (E * t**3 * b)
        Pc = (4.013 * E * torch.sqrt((t * t * b**6) / 36.0) / (_L * _L)) * (
            1.0 - (t / (2.0 * _L)) * torch.sqrt(E / (4.0 * G))
        )

        g = torch.stack(
            [
                tau / tau_max - 1.0,  # g1: shear
                sigma / sigma_max - 1.0,  # g2: bending
                (h - b) / 5.0,  # g3: geometry h <= b
                delta / _DELTA_MAX - 1.0,  # g4: deflection
                (_P - Pc) / _P,  # g5: buckling
            ],
            dim=1,
        )

        fx = (-cost).reshape(-1, 1)  # maximize -cost
        return g, fx
