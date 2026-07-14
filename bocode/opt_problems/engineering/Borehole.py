"""Borehole — water flow rate through a borehole (8-D), continuous or mixed.

The classic 8-dimensional borehole function: the flow rate of water through a
borehole that connects two aquifers. Widely used as a benchmark. The mixed-variable
form (``is_discrete=True``, from the GP+ project, Bostanabad Research Group) turns
the borehole radius ``rw`` and the casing length ``L`` into categorical variables
chosen from a small set of discrete levels.

Variables (order): ``rw`` [0.05, 0.15], ``r`` [100, 50000], ``Tu`` [63070, 115600],
``Hu`` [990, 1110], ``Tl`` [63.1, 116], ``Hl`` [700, 820], ``L`` [1120, 1680],
``Kw`` [9855, 12045]. Unconstrained.

OPTIMIZATION DIRECTION: **the flow rate is MINIMIZED**, so ``_evaluate_implementation``
returns ``-flow`` (BoCoDe maximizes). This needed a judgment call, because the direction
is genuinely not fixed by the canonical source: Surjanovic & Bingham file the borehole
function under *Emulation & Prediction* test functions, not optimization test problems
(https://www.sfu.ca/~ssurjano/emulat.html), and GP+ likewise uses it only as a
regression testbed. The tie-breakers, in order:

1. The one use of borehole as a Bayesian-optimization *objective* that we could find in
   the literature MINIMIZES it: Marmin, Chevalier & Ginsbourger report a unique global
   minimum at ``x* = (0, 1, 0, 0, 0, 1, 1, 0)`` in the unit cube. Evaluating this box's
   corner ``x*`` reproduces exactly that minimizer (flow = 7.8197 m3/yr here; their
   1.1918 differs only because they use the wider ``Kw`` in [1500, 15000] variant).
2. The scenario the function was built for is a nuclear-waste repository (Harper & Gupta
   1983; Worley 1987): water flow up the borehole is the radionuclide leakage pathway,
   i.e. something to be kept SMALL.
3. It matches BoCoDe's dominant convention (minimize a physical cost; negate).

CAVEAT: the flow rate is strictly monotone in all 8 variables (verified numerically), so
the optimum is a box CORNER in either direction and this is an easy benchmark whichever
sign you pick. ``optimum`` / ``x_opt`` below are that corner, in the minimization sense.

Sources:
A. Yousefpour, Z. Zanjani Foumani, M. Shishehbor, C. Mora, R. Bostanabad. GP+: a Python library for kernel-based learning via Gaussian processes. Advances in Engineering Software, 2024. https://github.com/Bostanabad-Research-Group/GP-Plus
Borehole function (Virtual Library of Simulation Experiments): https://www.sfu.ca/~ssurjano/borehole.html
S. Marmin, C. Chevalier, D. Ginsbourger. Efficient batch-sequential Bayesian optimization with moments of truncated Gaussian vectors. arXiv:1609.02700. (Borehole as a MINIMIZATION objective; unique global min at x* = (0,1,0,0,0,1,1,0).)
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

# (lower, upper) per variable, in order rw, r, Tu, Hu, Tl, Hl, L, Kw.
_BOUNDS = [
    (0.05, 0.15),
    (100.0, 50000.0),
    (63070.0, 115600.0),
    (990.0, 1110.0),
    (63.1, 116.0),
    (700.0, 820.0),
    (1120.0, 1680.0),
    (9855.0, 12045.0),
]
# Mixed form: rw (col 0) and L (col 6) become categorical with these discrete levels.
_RW_LEVELS = list(np.linspace(0.05, 0.15, 5))
_L_LEVELS = list(np.linspace(1120.0, 1680.0, 3))

# Global minimum of the flow rate: the corner (0,1,0,0,0,1,1,0) of the unit cube. Both
# of its mixed-variable coordinates (rw = 0.05, L = 1680) are levels of _RW_LEVELS /
# _L_LEVELS, so this optimum holds for the mixed form too.
_X_OPT = [[0.05, 50000.0, 63070.0, 990.0, 63.1, 820.0, 1680.0, 9855.0]]
_OPTIMUM = [7.819676328755232]  # minimization sense (m3/yr)


class Borehole(BenchmarkProblem):
    """Minimize borehole water flow rate (8 vars). ``is_discrete`` -> mixed (rw, L)."""

    available_dimensions = 8
    num_objectives = 1
    num_constraints = 0

    def __init__(self, is_discrete: bool = False) -> None:
        if is_discrete:
            self.variable_types = ["continuous"] * 8
            self.variable_types[0] = _RW_LEVELS
            self.variable_types[6] = _L_LEVELS
        else:
            self.variable_types = None
        super().__init__(
            dim=8,
            num_objectives=1,
            num_constraints=0,
            bounds=_BOUNDS,
            x_opt=_X_OPT,
            optimum=_OPTIMUM,
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        rw, r, Tu, Hu, Tl, Hl, L, Kw = (x[:, i] for i in range(8))

        ln_r_rw = np.log(r / rw)
        flow = (2 * np.pi * Tu * (Hu - Hl)) / (
            ln_r_rw * (1.0 + 2 * L * Tu / (ln_r_rw * rw**2 * Kw) + Tu / Tl)
        )
        # The flow rate is MINIMIZED (see the module docstring); BoCoDe maximizes.
        fx = torch.tensor(-flow, dtype=torch.float64).reshape(-1, 1)
        return None, fx
